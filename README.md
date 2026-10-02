# freetoken-ox-boost

GLM-5.3-Flash adaptation + single-GPU offload performance patch set for
[FreeToken](https://github.com/FlashML-org/FreeToken) v0.1.2, shipped as a
patch plugin: unified diffs against v0.1.2 (9db1a39) plus new files. The tree
these patches reproduce is upstream main at f7c31e9 plus our changes, so some
patches also carry upstream's own post-0.1.2 edits.

Running GLM-5.3-Flash-NVFP4 (181 GB of expert weights, host-memory offload,
PCIe Gen5 x16, host DDR5 at 3600 MT/s) on a single RTX PRO 6000 Blackwell
96 GB with `examples/serve_full.sh`: **~38 tok/s single-stream and ~45 tok/s
aggregate at 2 concurrent requests**, decode flat to 128K context (~37 tok/s).
The first adaptation ran 17.8 tok/s. Every optimization is lossless: greedy
outputs match the plain configuration token for token (text, prefix-cache
hits, images), and the model itself matches the HF reference implementation
48/48 steps. Vision (image and video input) is supported end-to-end — see
below.

Why concurrency does not scale here: GLM-5.3-Flash routing is extremely flat
(the 8 experts each token activates barely overlap between tokens), so
concurrent requests' expert-fetch bytes add up linearly per stream while also
competing for the cache — and the PCIe link is already saturated. Batching
cannot amortize the one thing this machine actually bills for: bytes.

## Measured metrics

Speed is sensitive to context length and cache warmth; each row states its
measurement conditions. Rows marked 2026-10-02 use `examples/serve_full.sh`
(memory ratio 0.96, max-extend 4096, 8 resident layers, soft expert cache)
with ~0.9K-token prompts, 1024 generated tokens and greedy decoding.

| Metric | Measured | Conditions |
|---|---|---|
| Single-stream decode | **~38 tok/s** (37.7-38.3) | 2026-10-02 |
| 2-way concurrent aggregate | **~45 tok/s** (22.5 per stream) | 2026-10-02 |
| Decode at long context | 38.7 / 37.4 / 36.8 tok/s | 2026-10-02, 16K / 64K / 128K context, 256 generated |
| Long-context prefill | 14 s / 38 s / 75 s | 2026-10-02, 16K / 64K / 128K (chunk 4096) |
| TTFT (~0.9K-token prompt) | 2.3-2.4 s warm, 4.1 s first request | 2026-10-02; a prefix-cache repeat 2.3 s |
| MoE offload misses | ~31 per token (miss rate 0.11) | 2026-10-02, 3337 cache slots + 8 resident layers |
| Real labeling load | 124/124 valid requests in 30 min, no loops, no swap growth | 2026-10-02, 2 workers, T=1.0, reasoning effort high |
| 4-way concurrent aggregate | ~49 tok/s (12 per stream) | 2026-08, 1-slot 256K pool; 2026-10-02 max-running 4 measured lower than 2 |
| TTFT (10-token prompt) | **0.59 s** | 2026-08, on-demand prefill path |
| TTFT (40 tokens) | 0.95 s | 2026-08, same |
| 232K long-context prefill | 232 s (~1030 tok/s) | 2026-08; needles at 10%/50%/95% all recalled |
| Decode after 232K context | 25.1 tok/s | 2026-08, 1-slot 256K config |

See `examples/serve_full.sh` for the launch (adjust paths, memory ratio and
preflight for your machine) and MANIFEST.md for what each switch does and how
much it measured.

## Vision, video and caching

The checkpoint's 0.6B ViT tower is ported and wired end to end. Images work on both the OpenAI and Anthropic-style endpoints, video on the OpenAI endpoint via PyAV at 2 fps by default. Enable with FREETOKEN_GLM5_VISION=1, off by default, costs about 1.2 GB VRAM, and the text-only path is byte-identical when off. Preprocessing is bitwise-equal to the HF image processor and tower drift stays within the BF16 noise between HF's own sdpa and eager backends. Verified on the production server for description, counting, two-image comparison, text-in-image, and motion on video. A 220-token image request has the same TTFT as a 220-token text prompt.

Prefix caching works for media requests. Image placeholder ids are replaced by a pixel-content hash in the radix cache key, so identical text plus media prefixes hit and different images never false-hit. Repeating an image request drops TTFT from 3.8 to 1.0 s, and a follow-up turn in the same image conversation from 2.8 to 1.1 s. Under production-shaped load, multi-turn agent conversations with interleaved images hold about 1 s TTFT per turn, shared system prompts and long documents hit across conversations, and LRU eviction past the 262K pool keeps the newest entries. One known edge: a re-query landing within one decode step of an identical request can miss the not-yet-inserted prefix and pay one cold prefill.

**Hybrid hit-corruption: found, root-caused, and fixed (2026-08-30).** An
overnight ground-truth soak caught radix cache-HIT requests corrupting ~10% of
the time (blank visual grounding, empty or garbled answers, text and image
alike; cold path always clean; sticky per-branch failure windows). Slot-level
checksum forensics delivered the true root cause: the KDA op never implemented
the ×64-boundary track-snapshot write that the hybrid-radix design expects
(`_write_track_snapshot` exists only in the Qwen GDN op), so every donated
reuse point carried an UNWRITTEN all-zero recurrent state — hits then ran the
prefix on the 11 full-attention layers alone, which usually survives and
intermittently derails. The fix implements the missing writer for KDA
(recompute-based: `chunk_kda` only exposes the final state, so each tracked
request's pre-boundary slice is re-run from its pre-forward state and banked;
the main path stays bit-identical), plus defense-in-depth from the
investigation: snapshot copy-on-donate (tree slots are written once and
read-only forever), full-device barriers at hit admission and donation, and a
`gdn_track_snapshots` gate so a model without a writer can never donate
unwritten state again. Verified: restore-side checksums show real boundary
state on every hit (zero-state restores eliminated), 90-round hammer clean,
same-image repeat 0.85 s / turn-2 0.99 s / long-document follow-up 1.35 s with
correct answers throughout. `radix` is the example default.

**Large images (2026-10-02).** A multimodal prompt longer than one prefill
chunk used to raise inside the scheduler and kill the backend (a 3840x2160
image, or two concurrent 1920x1080 images at chunk 4096). Such prompts now wait
for a full prefill budget, are prefilled across chunks with each chunk taking
its own image-embedding rows (`FREETOKEN_MM_CHUNKED_PREFILL=1`), or are refused
with `multimodal_prompt_too_long` when they can never fit; the server stays up
either way. The vision tower's attention runs in query-row chunks under a
memory budget, so a 3840x2160 image encodes with ~3 GB of free VRAM. Fitting
images and text give the same output token for token with these switches on.

## Layout

```
patches/    65 per-file unified diffs (against v0.1.2, git apply -p1)
overlay/    41 new files (models/glm5_next incl. vision, models/naive_n05, DSV4 extras, triton kernels,
            soft cache, spec prefetch, elastic KV, gpu_select, ...); install.sh copies the whole tree
install.sh  version check -> dry-run -> apply -> copy overlay -> compileall
examples/   serve_full.sh (GLM-5.3-Flash, every lossless switch on) and serve_dsv4.sh
MANIFEST.md feature -> files -> switches -> measured numbers
```

## License

Apache-2.0, same as upstream. This repository contains patches and new files
only, not a copy of the upstream tree.
