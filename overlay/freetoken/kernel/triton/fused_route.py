"""Fused MoE router epilogue (2026-08-28 tail-fusion pass).

After the (kept, cublas) fp32 gate GEMV this replaces the eager chain
sigmoid -> +bias -> torch.topk (gather+bitonic) -> gather -> sum -> div -> mul
-> int32 cast (~8 launches x 42 MoE layers/step) with ONE kernel per token batch.

Semantics mirror ``Glm4MoeSparseBlock`` routing with n_group<=1 exactly:
  scores = sigmoid(logits); s4c = scores + bias; ids = top-8 of s4c (descending,
  first-index tie-break); w = scores[ids]; w /= (sum(w)+1e-20) [if RENORM];
  w *= scaling.  The renorm sum runs in the same descending order as the eager
  gather->sum, so rounding matches; only sigmoid may differ from ATen by <=1 ulp.
CUDA-graph safe: fixed shapes, no host sync, no atomics (deterministic)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_route_kernel(
    logits_ptr, bias_ptr, w_ptr, id_ptr,
    stride_lm, scaling,
    E: tl.constexpr, TOPK: tl.constexpr, RENORM: tl.constexpr, BLOCK: tl.constexpr,
    ACT: tl.constexpr,
):
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < E
    lg = tl.load(logits_ptr + m * stride_lm + offs, mask=mask, other=0.0).to(tl.float32)
    if ACT == 1:
        # sqrt(softplus), overflow-stable: max(x,0) + log(1 + exp(-|x|)) -- the log
        # argument sits in (1, 2], full fp32 precision (<=1ulp vs ATen threshold form)
        sp = tl.maximum(lg, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(lg)))
        sc = tl.sqrt(sp)
    else:
        sc = tl.sigmoid(lg)
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s4c = tl.where(mask, sc + bias, float("-inf"))
    wsum = 0.0
    for k in tl.static_range(TOPK):
        idx = tl.argmax(s4c, axis=0)  # first-index tie-break, like sorted topk
        val = tl.sum(tl.where(offs == idx, sc, 0.0), axis=0)
        tl.store(id_ptr + m * TOPK + k, idx.to(tl.int32))
        tl.store(w_ptr + m * TOPK + k, val)
        s4c = tl.where(offs == idx, float("-inf"), s4c)
        wsum += val
    for k in tl.static_range(TOPK):
        v = tl.load(w_ptr + m * TOPK + k)  # same-thread reload: program-ordered
        if RENORM:
            v = v / (wsum + 1e-20)
        tl.store(w_ptr + m * TOPK + k, v * scaling)


def fused_route(
    logits: torch.Tensor,   # [M, E] fp32 (router gemv output)
    bias: torch.Tensor,     # [E] (e_score_correction_bias; any float dtype)
    top_k: int,
    renorm: bool,
    scaling: float,
    act: str = "sigmoid",
):
    M, E = logits.shape
    w = torch.empty(M, top_k, dtype=torch.float32, device=logits.device)
    ids = torch.empty(M, top_k, dtype=torch.int32, device=logits.device)
    _fused_route_kernel[(M,)](
        logits, bias, w, ids, logits.stride(0), scaling,
        E=E, TOPK=top_k, RENORM=renorm, BLOCK=triton.next_power_of_2(E),
        ACT=1 if act == "sqrtsoftplus" else 0,
        num_warps=4,
    )
    return w, ids


@triton.jit
def _fused_route_ext_kernel(
    logits_ptr, bias_ptr, w_ptr, id_ptr, rank_id_ptr, rank_sc_ptr,
    stride_lm, scaling,
    E: tl.constexpr, TOPK: tl.constexpr, NRANK: tl.constexpr, RENORM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``_fused_route_kernel`` (sigmoid) plus the ranked list the soft cache policy
    feeds on: the top-NRANK ids by selection score with their raw sigmoid scores
    (ranks < TOPK are exactly the selected experts, same order and tie-break)."""
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < E
    lg = tl.load(logits_ptr + m * stride_lm + offs, mask=mask, other=0.0).to(tl.float32)
    sc = tl.sigmoid(lg)
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s4c = tl.where(mask, sc + bias, float("-inf"))
    wsum = 0.0
    for k in tl.static_range(NRANK):
        idx = tl.argmax(s4c, axis=0)
        val = tl.sum(tl.where(offs == idx, sc, 0.0), axis=0)
        tl.store(rank_id_ptr + m * NRANK + k, idx.to(tl.int32))
        tl.store(rank_sc_ptr + m * NRANK + k, val)
        if k < TOPK:
            tl.store(id_ptr + m * TOPK + k, idx.to(tl.int32))
            tl.store(w_ptr + m * TOPK + k, val)
            wsum += val
        s4c = tl.where(offs == idx, float("-inf"), s4c)
    for k in tl.static_range(TOPK):
        v = tl.load(w_ptr + m * TOPK + k)
        if RENORM:
            v = v / (wsum + 1e-20)
        tl.store(w_ptr + m * TOPK + k, v * scaling)


def fused_route_ranked(logits, bias, top_k: int, renorm: bool, scaling: float, n_rank: int = 16):
    """Same routing result as ``fused_route`` (sigmoid) + (rank_ids, rank_scores)
    [M, n_rank] for the score-aware expert-cache policy."""
    M, E = logits.shape
    w = torch.empty(M, top_k, dtype=torch.float32, device=logits.device)
    ids = torch.empty(M, top_k, dtype=torch.int32, device=logits.device)
    rid = torch.empty(M, n_rank, dtype=torch.int32, device=logits.device)
    rsc = torch.empty(M, n_rank, dtype=torch.float32, device=logits.device)
    _fused_route_ext_kernel[(M,)](
        logits, bias, w, ids, rid, rsc, logits.stride(0), scaling,
        E=E, TOPK=top_k, NRANK=n_rank, RENORM=renorm, BLOCK=triton.next_power_of_2(E),
        num_warps=4,
    )
    return w, ids, rid, rsc
