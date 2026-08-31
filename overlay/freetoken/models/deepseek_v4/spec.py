"""DSpark speculative decoding: engine-side state machine for DeepSeek-V4-Flash.

Round layout (steady state; one request, greedy, bs==1). Positions: the request has
consumed tokens through position p-1; ``x_p`` (this step's input) sits unconsumed at
position p; the DSpark rings hold main_kv through position p-1; the compressor/indexer
ring carries are SETTLED at position p-1 (the round invariant).

  draft   : carried from last round -- d1..dB drafted from [x_p, noise x B-1] at
            positions p..p+B-1 (one parallel 3-stage DSpark forward).
  verify  : ONE main-model extend of [x_p, d1..dB] (B+1 tokens, positions p..p+B) on
            the prefill path -- exact sequential attention/compressor/indexer semantics,
            on-demand MoE fetch through the slot cache. Logits rows 0..B-1 verify
            d1..dB; row a is the bonus. Per-layer attention inputs are stashed
            (batch.spec_stash) for the carry fix; the target layers' hc-mean hiddens
            land in batch.spec_hiddens for the rings.
  accept  : a = longest prefix with argmax match; emit a+1 tokens (d1..da + bonus).
  rollback: ring-carry archive (taken pre-verify at the tail window page) is restored,
            then compressor+indexer re-advance over ONLY the accepted a+1 tokens from
            the stashed inputs -- carries settle at position p+a. KV/cmp/idx pool rows
            past the accepted point are junk but unreachable: every candidate list is
            causally masked by position, and the next round's extend overwrites them
            store-first.
  rings   : main_kv written for consumed positions p..p+a from the verify hiddens;
            next round's draft starts at ring_last = p+a with the bonus token.

Any gate failure (bs>1, sampling, no allocation headroom) falls back to the normal
decode path; stale state is dropped by the (uid, device_len) key.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from typing import List

import torch

from freetoken.core import Batch


class _FakeReq:
    __slots__ = ("table_idx", "cached_len", "device_len", "linear_slot_idx",
                 "mamba_ping_pong", "mm_embeds")

    def __init__(self, table_idx: int, cached_len: int, device_len: int):
        self.table_idx = table_idx
        self.cached_len = cached_len
        self.device_len = device_len
        self.linear_slot_idx = None
        self.mamba_ping_pong = None
        self.mm_embeds = None

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len


class DsparkSpec:
    def __init__(self, engine):
        from freetoken.models.deepseek_v4.dspark import DsparkDraft

        self.eng = engine
        model = engine.model._transformer if hasattr(engine.model, "_transformer") else engine.model
        self.model = model
        self.args = model.args
        self.B = self.args.dspark_block_size          # draft block (5 -> 5 drafts)
        self.log = os.environ.get("FREETOKEN_DSV4_SPEC_LOG", "0") == "1"
        self.conf_thr = float(os.environ.get("FREETOKEN_DSV4_SPEC_CONF", "-1e30"))
        self.draft = DsparkDraft(self.args, model, engine.device)
        # compressed-layer attention modules (carry fix targets)
        self.cmp_layers = [
            blk.attn for blk in model.layers if blk.attn.compress_ratio
        ]
        # single-slot spec state, keyed by (uid, device_len)
        self.uid: int | None = None
        self.expect_dl = -1
        self.d_toks: torch.Tensor | None = None       # [B] draft tokens
        self.cache_gpu: torch.Tensor | None = None
        self.cache_pos = 0
        self.cache_len = 0
        self._last_conf: list | None = None
        self._pending = False
        self.stats = Counter()
        self.m_hist = Counter()
        self._round_t = 0.0
        self._ph = [0.0, 0.0, 0.0, 0.0]               # arch/verify/fix/draft
        # captured verify graph (lazily, once the engine serves with CUDA graphs)
        self._vg = None
        self._vg_state = None
        self._vg_failed = os.environ.get("FREETOKEN_DSV4_SPEC_EAGER", "0") == "1"

    # ------------------------------------------------------------------ entry points
    def try_step(self, batch: Batch):
        if not batch.is_decode or batch.size != 1:
            return None
        req = batch.reqs[0]
        sp = req.sampling_params
        if req.aborted or not (sp.temperature <= 0.0 or sp.top_k == 1) or not req.can_decode:
            return None
        keyed = req.uid == self.uid and req.device_len == self.expect_dl
        if keyed and self.cache_pos < self.cache_len:
            return self._pop(req)
        if (
            keyed
            and self.d_toks is not None
            and self.cache_pos >= self.cache_len
            and getattr(req, "spec_alloc_len", 0) >= req.device_len + self.B
            and req.device_len + self.B <= self.eng.max_seq_len
        ):
            return self._round(batch, req)
        if self.log and self.stats["gate_dbg"] < 6:
            self.stats["gate_dbg"] += 1
            print(f"[dspark-gate] keyed={keyed} uid={req.uid}/{self.uid}"
                  f" dl={req.device_len}/{self.expect_dl} d_toks={self.d_toks is not None}"
                  f" alloc={getattr(req, 'spec_alloc_len', 0)} need={req.device_len + self.B}",
                  flush=True)
        self._pending = True  # bootstrap off the normal forward that follows
        return None

    def wants_eager(self, batch: Batch) -> bool:
        return False  # the decode hidden buffer is model-owned; graphs write it in place

    def after_main(self, batch: Batch, next_tokens_gpu: torch.Tensor) -> None:
        """Bootstrap: after a normal forward, seed the rings from the staged target-layer
        hiddens and draft the first block. Prefill batches (bs==1) always seed -- the ring
        needs the prompt's trailing main_kv entries; decode batches seed only when a failed
        try_step marked the round pending (stale key, missing draft)."""
        if batch.size != 1:
            self._pending = False
            return
        req = batch.reqs[0]
        model = self.model
        if batch.is_decode:
            if not self._pending:
                return
            self._pending = False
            pos = req.device_len - 2  # complete_one already advanced device_len
            hid = model.spec_hidden_decode[:1]
            if req.uid != self.draft.ring_uid:
                self.draft.ring_reset(req.uid)
            self.draft.ring_write(
                torch.arange(pos, pos + 1, device=self.eng.device), hid)
        else:
            self._pending = False
            hd = getattr(batch, "spec_hiddens", None)
            if hd is None or 0 not in hd:
                if self.log and self.stats["nohid_dbg"] < 4:
                    self.stats["nohid_dbg"] += 1
                    print(f"[dspark-nohid] prefill batch without spec_hiddens ({hd})",
                          flush=True)
                return
            positions, hiddens = hd[0]  # request 0's (positions [N], hiddens [N, 3D])
            if req.uid != self.draft.ring_uid:
                self.draft.ring_reset(req.uid)
            self.draft.ring_write(positions, hiddens)
        t1 = next_tokens_gpu[:1].long()
        toks, conf = self.draft.draft(t1[0])
        self._last_conf = [round(float(v), 3) for v in conf.tolist()]
        self.d_toks = self._trim(toks, conf)
        self.uid = req.uid
        self.expect_dl = req.device_len
        self.cache_pos = self.cache_len = 0
        self.stats["bootstrap"] += 1
        if self.log and self.stats["bootstrap"] <= 4:
            print(f"[dspark-boot] #{self.stats['bootstrap']} uid={req.uid}"
                  f" dl={req.device_len} ring_last={self.draft.ring_last}"
                  f" ring_valid={self.draft.ring_valid}", flush=True)

    # ------------------------------------------------------------------ internals
    def _trim(self, toks: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
        """Confidence gating: submit the prefix with conf > threshold (default: all)."""
        if self.conf_thr <= -1e29:
            return toks
        keep = int(torch.cumprod(conf > self.conf_thr, 0).sum().item())
        return toks[: max(keep, 1)]

    def _pop(self, req):
        from freetoken.engine.engine import ForwardOutput

        tok = self.cache_gpu[self.cache_pos:self.cache_pos + 1]
        self.cache_pos += 1
        self.expect_dl += 1
        req.complete_one()
        cpu = tok.to("cpu", non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(self.eng.stream)
        self.stats["pop"] += 1
        return ForwardOutput(tok, cpu, ev)

    def _fake_verify(self, table_idx: int, start: int, toks: torch.Tensor) -> Batch:
        """n single-token co-tenant DECODE reqs at positions [start, start+n): the decode
        path natively handles arbitrary positions (no 128-alignment), rides the fully
        device-driven MoE fetch, and returns every row's logits. Per-row staggered kvlen
        gives exact causal attention (all rows' KV stores land before the attention
        kernel); the compressors chain their carry through spec_decode_chain (the
        ``spec_verify`` flag branches inside Compressor.decode_step)."""
        eng = self.eng
        n = toks.numel()
        b = Batch(reqs=[_FakeReq(table_idx, start + i, start + i + 1) for i in range(n)],
                  phase="decode")
        b.padded_reqs = b.reqs
        b.input_ids = toks
        b.positions = torch.arange(start, start + n, dtype=torch.int32, device=eng.device)
        b.out_loc = eng.page_table[table_idx, start:start + n].clone()
        b.active_table_idx = torch.full(
            (n,), table_idx, dtype=torch.int64, device=eng.device)
        eng.attn_backend.prepare_metadata(b)
        b.spec_verify = True
        return b

    def _round(self, batch: Batch, req):
        from freetoken.engine.engine import ForwardOutput

        eng, model = self.eng, self.model
        t0 = time.monotonic()
        k = self.d_toks.numel()
        p = req.device_len - 1
        x_p = batch.input_ids.view(-1)[:1].to(torch.int32)

        def _tick(i0):
            torch.cuda.synchronize()
            t = time.monotonic()
            if i0 >= 0:
                self._ph[i0] += t - _tick.last
            _tick.last = t
        _tick(-1)

        # -- carry archive (both tiers, per compressed layer) at the tail window page
        bk = eng.attn_backend
        tail_ws = int(bk.window_slots_of(req.table_idx, p - 1, p).item()) if p > 0 else None
        archives = []
        if tail_ws is not None:
            for attn in self.cmp_layers:
                c = attn.compressor
                a_blk = bk.read_carry(attn.layer_id, "attn", tail_ws, c.ring_size).clone()
                i_blk = None
                if attn.indexer is not None:
                    i_blk = bk.read_carry(
                        attn.layer_id, "idx", tail_ws, attn.indexer.compressor.ring_size
                    ).clone()
                archives.append((attn, a_blk, i_blk))
        _tick(0)

        # -- verify: k+1 staggered single-token decode rows at positions p..p+k
        toks = torch.cat([x_p, self.d_toks.to(torch.int32)])
        if self._vg is None and not self._vg_failed and k + 1 == self.B + 1:
            self._try_capture_verify(req, p, toks)
        if self._vg is not None and k + 1 == self.B + 1:
            vb, vlogits = self._replay_verify(req, p, toks)
        else:
            vb = self._fake_verify(req.table_idx, p, toks)
            vb.spec_stash = {}
            with eng.ctx.forward_batch(vb):
                vlogits = eng.model.forward()            # [k+1, V] decode path
        h_all = model.spec_hidden_decode[:k + 1].clone()  # [k+1, 3D] target-layer means
        greedy = vlogits.argmax(dim=-1)                  # [k+1]

        _tick(1)
        # -- accept: longest draft prefix + bonus
        matched = torch.cumprod((greedy[:k] == self.d_toks).int(), 0)
        a = int(matched.sum().item())
        a = min(a, req.remain_len - 1)
        emitted = torch.cat([self.d_toks[:a], greedy[a:a + 1]])  # [a+1]

        # -- carry fix (only when a draft was rejected): restore the pre-verify ring
        # carries, then re-chain over just the accepted a+1 tokens. A fully-accepted
        # round needs nothing -- the verify chain already left the sequential state.
        if a < k and tail_ws is not None:
            for attn, a_blk, i_blk in archives:
                c = attn.compressor
                bk.write_carry(attn.layer_id, "attn", tail_ws, c.ring_size, a_blk)
                if i_blk is not None:
                    bk.write_carry(attn.layer_id, "idx", tail_ws,
                                   attn.indexer.compressor.ring_size, i_blk)
            n_fix = a + 1
            pos_fix = torch.arange(p, p + n_fix, device=eng.device)
            prev_fix = bk.window_slots_of(req.table_idx, p - 1, p + n_fix - 1)
            cur_fix = bk.window_slots_of(req.table_idx, p, p + n_fix)
            rows_fix = torch.arange(n_fix, device=eng.device)
            with eng.ctx.forward_batch(vb):  # snapshot addressing for the block scatter
                for attn in self.cmp_layers:
                    x_seg = vb.spec_stash[attn.layer_id][:n_fix]
                    attn.compressor.spec_decode_chain(
                        x_seg, pos_fix, prev_fix, cur_fix, rows_fix)
                    if attn.indexer is not None:
                        attn.indexer.compressor.spec_decode_chain(
                            x_seg, pos_fix, prev_fix, cur_fix, rows_fix)
        _tick(2)

        # -- rings + next draft
        positions = torch.arange(p, p + a + 1, device=eng.device)
        self.draft.ring_write(positions, h_all[: a + 1])
        if self._last_conf is not None and self.log:
            # calibration trail: this round's accept count against the confidences the
            # PREVIOUS draft assigned to the tokens just verified
            try:
                with open("/tmp/dspark_conf.jsonl", "a") as f:
                    f.write(f'{{"a": {a}, "conf": {self._last_conf}}}\n')
            except OSError:
                pass
        toks_next, conf = self.draft.draft(emitted[-1])
        self._last_conf = [round(float(v), 3) for v in conf.tolist()]
        self.d_toks = self._trim(toks_next, conf)
        _tick(3)

        # -- bookkeeping: emit the first token now, cache the rest
        req.complete_one()
        self.cache_gpu = emitted[1:].to(torch.int32)
        self.cache_pos, self.cache_len = 0, a
        self.expect_dl = req.device_len

        self.stats["round"] += 1
        self.stats["accepted"] += a
        self.m_hist[a] += 1
        self._round_t += time.monotonic() - t0
        if self.log and (self.stats["round"] % 25 == 0 or self.stats["round"] == 1):
            r = self.stats["round"]
            print(
                f"[dspark] rounds={r} tok/round={(self.stats['accepted'] + r) / r:.2f}"
                f" m_hist={dict(sorted(self.m_hist.items()))}"
                f" ms/round={1000 * self._round_t / r:.1f}"
                f" phases(arch/verify/fix/draft)ms="
                f"{'/'.join(f'{1000 * x / r:.0f}' for x in self._ph)}"
                f" pops={self.stats['pop']} boots={self.stats['bootstrap']}",
                flush=True,
            )
        out = emitted[:1].to(torch.int32)
        cpu = out.to("cpu", non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(eng.stream)
        return ForwardOutput(out, cpu, ev)


    # ------------------------------------------------------------------ verify graph
    def _try_capture_verify(self, req, p: int, toks: torch.Tensor) -> None:
        """Capture the verify decode forward as a standalone CUDA graph (once per boot),
        reusing the engine graphs' memory pool. The capture runs on the DUMMY slot so it
        has zero side effects on live requests; replays stage the real request's
        addressing into the same static buffers. On any failure: eager forever."""
        eng = self.eng
        runner = eng.graph_runner
        n = self.B + 1
        if not runner.graph_map or runner.max_graph_bs < n:
            self._vg_failed = True
            return
        try:
            dev = eng.device
            st: dict = {}
            st["input_ids"] = torch.zeros(n, dtype=torch.int32, device=dev)
            st["positions"] = torch.zeros(n, dtype=torch.int32, device=dev)
            st["out_loc"] = torch.zeros(n, dtype=torch.int32, device=dev)
            st["active"] = torch.zeros(n, dtype=torch.int64, device=dev)
            b = Batch(reqs=[_FakeReq(req.table_idx, p + i, p + i + 1) for i in range(n)],
                      phase="decode")
            b.padded_reqs = b.reqs
            b.input_ids = st["input_ids"]
            b.positions = st["positions"]
            b.out_loc = st["out_loc"]
            b.active_table_idx = st["active"]
            eng.attn_backend.prepare_metadata(b)
            b.spec_verify = True
            b.spec_stash = {}
            st["batch"] = b
            dummy = eng.dummy_req
            self._stage_verify(st, dummy, 0, torch.zeros_like(toks))
            model = eng.model
            g = torch.cuda.CUDAGraph()
            pool = next(iter(runner.graph_map.values())).pool()
            with eng.ctx.forward_batch(b):
                warm = model.forward()                 # warmup + triton compile
                st["logits"] = torch.empty_like(warm)
                b.spec_stash.clear()
                with torch.cuda.graph(g, pool=pool, stream=eng.stream):
                    st["logits"].copy_(model.forward())
            self._vg = g
            self._vg_state = st
            print(f"[dspark] verify graph captured (bs={n})", flush=True)
        except Exception as e:  # noqa: BLE001
            self._vg_failed = True
            print(f"[dspark] verify graph capture FAILED ({e!r}); staying eager",
                  flush=True)

    def _stage_verify(self, st: dict, req, p: int, toks: torch.Tensor) -> None:
        n = self.B + 1
        eng = self.eng
        st["input_ids"].copy_(toks)
        st["positions"].copy_(
            torch.arange(p, p + n, dtype=torch.int32, device=eng.device))
        st["out_loc"].copy_(eng.page_table[req.table_idx, p:p + n])
        st["active"].fill_(req.table_idx)
        b = st["batch"]
        for i, r in enumerate(b.reqs):
            r.table_idx = req.table_idx
            r.cached_len = p + i
            r.device_len = p + i + 1
        # stage rows/snapshot into the captured buffers (replaces b.attn_metadata with
        # the capture view -- same call the engine's own replays use)
        eng.attn_backend.prepare_for_replay(b)

    def _replay_verify(self, req, p: int, toks: torch.Tensor):
        st = self._vg_state
        self._stage_verify(st, req, p, toks)
        self._vg.replay()
        return st["batch"], st["logits"]


__all__ = ["DsparkSpec"]
