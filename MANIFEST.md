# Feature -> files -> switches

Baseline: FreeToken v0.1.2 (commit 9db1a39). All speed numbers measured on an
RTX PRO 6000 Blackwell 96 GB + GLM-5.3-Flash-NVFP4 (181 GB experts, host
offload), single-stream decode unless noted.

## Core adaptation (no switches; active once installed)

| Feature | Files | Notes |
|---|---|---|
| GLM5-Next model (34 KDA layers + 11 MLA/DSA layers + mHC + MoE) | `overlay: models/glm5_next/*`; `patches: models_register / models_config / models_glm_moe_dsa_args` | 48/48 steps match the HF reference token for token |
| NoPE guard (rotary_dim=0) | `patches: models_glm_moe_dsa_attention / kernel_triton_glm_dsa_sparse` | D_R=0 constexpr branches |
| DSA k-pool sparsity + pooled scoring | `patches: attention_dsa / kvcache_dsa_pool / kvcache___init__` | pooled keys bitwise-match HF; >2048-token needle recall verified |
| Clamped swiglu (routed experts on the Triton path) | `patches: moe_expert_banks / moe_fused_nvfp4 (part) / models_glm_moe_dsa_moe (part)` | b12x epilogue has no clamp; auto resolves to triton |
| MTP layer weight skip (layer 45 dropped, saves 3.9 GB) | `overlay: models/glm5_next/weight.py` | speculative decoding archived, see below |

## Performance optimizations (env-controlled; defaults in parentheses)

| Optimization | Measured | Files | Switch |
|---|---|---|---|
| Non-expert FP8 (attn/mlp) | +28%, prefill 2.7x | `models/glm5_next/{attention,mlp,weight}.py` | `FREETOKEN_GLM5_ATTN_FP8` / `MLP_FP8` (1) |
| KDA FP8 (in_proj + o_proj) | +15% | same | `FREETOKEN_GLM5_KDA_FP8` (1) |
| Hot resident layers (re-picked on coding traces: 3-6,8-11) | +29% (miss 0.24 -> 0.157); coding sim ties the 3-10 pick | `models/glm5_next/experts_resident.py` + `patches: models_glm_moe_dsa_moe` | `FREETOKEN_GLM5_RESIDENT_LAYERS` (3-6,8-11) |
| Speculative expert prefetch (layer L+1 gate scored early) | +8% (P=4 hop-1 sweet spot) | `overlay: moe/spec_prefetch.py` + `patches: layers_moe` | `FREETOKEN_MOE_SPEC_PREFETCH` (0; production uses 4) |
| Short-prompt on-demand prefill | TTFT 3.5x (10-token 2.1 -> 0.59 s) | `patches: layers_moe` | `FREETOKEN_PREFILL_ONDEMAND_TOKENS` (48; 0=off) |
| Marlin decode GEMV config 16/32/2 | +4.5% (gate_up kernel -15%) | `patches: moe_fused_nvfp4` | `FREETOKEN_MARLIN_BN/BKW/WARPS` (16/32/2) |
| KDA gate fusion (5 GEMVs -> 2 + 7 elementwise -> 1) | +1% | `overlay: kernel/triton/kda_gate.py` + `models/glm5_next/attention.py` | `FREETOKEN_KDA_FUSED_GATE` (1) |
| mHC pre-norm fusion (cast/mean/rsqrt in one kernel + atomics-free gemv) | within noise (-6 kernels/layer) | `overlay: kernel/triton/dsv4/hc_norm.py` + `models/glm5_next/model.py` | none (epsilon-level numerics) |
| Router fusion (sigmoid+bias+topk+renorm, 8 kernels -> 1) | +3.4% | `overlay: kernel/triton/fused_route.py` + `patches: models_glm_moe_dsa_moe` | `FREETOKEN_FUSED_ROUTE` (1; auto-falls back when n_group>1) |
| FP8 small-batch M-tile GEMV (2<=M<=4 skips the prefill GEMM) | conc2 +13% (M=2 kernel 3x) | `patches: kernel_triton_fp8_pertensor_linear` | none (automatic for M<=4) |
| 2-slot 256K KV pool (cache 2079 -> 2600 slots) | single +14%, conc4 +67% | `examples/serve_full.sh` | `GLM5_KV_RESERVE` (524288) |
| CPU swiglu_clamp support (hybrid backend can serve GLM-5.3) | net-zero on this VM; kernels kept | `patches: kernel_csrc...cpu_moe_ext / moe_cpu_executor / layers_moe (bs gate)` | `GLM5_MOE_BACKEND=hybrid` + `GLM5_CPU_THREADS` |

## Vision support (env-gated, default off)

| Feature | Files | Switch / requirement |
|---|---|---|
| Vision tower (0.6B ViT port of `Glm5NextVisionModel`; loads `model.visual.*` BF16 verbatim, +~1.2 GB VRAM) | `overlay: models/glm5_next/vision.py` + wiring in `config.py / weight.py / model.py` | `FREETOKEN_GLM5_VISION` (0) |
| Image preprocessing (vendored `Glm5NextImageProcessorPil`: smart resize -> bicubic -> pad -> CLIP normalize -> patchify; bitwise-equal pixel_values vs HF) | `overlay: models/glm5_next/image_process.py` | `pip install pillow` |
| Server intake: OpenAI `image_url` (data URI / http) + Anthropic base64 `image` blocks -> template placeholder expansion -> pixels over the ZMQ msgpack codec -> vision encode at admission in the scheduler process | `patches: server_generation / server_openai_api / server_anthropic_api / tokenizer_tokenize / tokenizer_server / message_tokenizer / message_backend / scheduler_scheduler` | same switch |

| Video input (OpenAI `video_url` parts; PyAV decode -> fps-2 sampling -> paired-frame temporal patches -> per-unit frame blocks with timestamps, HF-processor-faithful; per-unit vision attention segments) | `overlay: models/glm5_next/image_process.py (video half)` + `patches: server_generation / server_api_models / tokenizer_tokenize` | `pip install av`; `FREETOKEN_GLM5_VIDEO_FPS` (2) / `FREETOKEN_GLM5_MAX_VIDEO_TOKENS` (8000) / `FREETOKEN_GLM5_MAX_VIDEO_FRAMES` (64) |
| Content-hash prefix cache for media requests (image spans keyed by per-image pixel blake2b in the negative id range; same media prefix hits, different media diverges at span start; covered leading spans slice the scatter rows) | `patches: scheduler_cache / scheduler_prefill / scheduler_utils / scheduler_scheduler` | active with `--cache-type radix`; offline mm path keeps the old skip |
| Image-token budget knob | `overlay: models/glm5_next/image_process.py` | `FREETOKEN_GLM5_MAX_IMAGE_TOKENS` (8000 patches) |

Context-cache hardening battery (all on the production box): 6-turn agent
conversation ~1 s TTFT per turn with image spans cached (follow-up about a
cached image: 0.8 s); cross-conversation system-prompt reuse 4.0 -> 1.0 s;
12K-doc immediate re-query 0.8 s; conc-2 shared prefix both 1.5 s; LRU
eviction under 275K > 262K pool pressure evicts oldest / keeps newest with
zero integrity failures. Known edge: an identical-prefix re-query inside the
one-decode-step window between match and the previous request's finish-insert
pays one cold prefill (rare, benign).

KDA track-snapshot writer + hybrid-cache hardening (2026-08-30): the missing
×64-boundary snapshot write for KDA (`overlay: models/glm5_next/attention.py`,
recompute-based), `gdn_track_snapshots` model flag (`patches: models_config /
scheduler_scheduler / scheduler_cache / scheduler_prefill / attention_linear`),
snapshot copy-on-donate + admission/donate barriers, and env-gated GDN
checksum forensics (`FREETOKEN_GDN_DEBUG`, `pool.debug_checksum`).

Measured (production box, radix + `FREETOKEN_PREFILL_ONDEMAND_TOKENS=128`):
same-image repeat TTFT 3.8 -> 1.0 s; image-conversation turn-2 2.8 -> 1.1 s;
long-text shared prefix 4.1 -> 1.1 s; different image never false-hits (fresh
answer verified); needle recall correct after prefix reuse; video motion /
start-end / colors correct on a synthetic clip. Vision overhead outside the
cache: tower + vendored preprocessing < 20 ms for a 448px image (a 220-token
image request matches a 220-token text prompt's TTFT); the 24-block tower runs
one attention segment per temporal unit (t=2 parity vs HF within the bf16
noise envelope).

Validation: tower parity vs HF stage-by-stage on real weights (patch embed +
rotary tables bitwise-equal; per-layer drift within the BF16 kernel-noise
envelope of HF sdpa-vs-eager). End-to-end on the production server: shapes /
colors / positions, counting, two-image comparison, text-in-image reading, on
both APIs. Constraints: an image prompt must fit one prefill chunk;
`count_tokens` does not account for image expansion yet; Anthropic-endpoint
video blocks are not supported (no standard block type).

## Archived experiments (off by default; verdicts in commit history)

| Experiment | Verdict | Switch |
|---|---|---|
| MTP speculative decoding (full verify engine + CUDA graph) | correct but -13% under PCIe-byte billing; archived | `FREETOKEN_GLM5_SPEC` (0) |
| LFU-decay cache policy | loses on real load (drift-shaped locality) | `FREETOKEN_MOE_CACHE_POLICY` (lru) |
| Fallback-free miss cap (expert dropping) | first round -35% (order-blind drop); pending round 2 | `FREETOKEN_MOE_MISS_CAP` (-1) |
| top-k knob | top-6 gives +6% but changes the A18B spec; vetoed | `FREETOKEN_GLM5_TOPK` (unset = 8) |
| Routing trace collection | analysis tool | `FREETOKEN_ROUTE_TRACE` (off; value is the dump path) |
| Decode step profiler | analysis tool | `FREETOKEN_PROFILE_DECODE` (off) |
| Separate prefetch P/hop for bs>1 | sweep verdict: uniform P=4 / hop-1 wins everywhere | `FREETOKEN_MOE_SPEC_PREFETCH_MULTI` (-1 = follow) / `SPEC_HOP_MULTI` (1) |
| Hybrid CPU offload | ~zero net contribution (QEMU vCPU); archived | `GLM5_MOE_BACKEND` (offload) |

## Infrastructure (patch group 01; not part of this optimization line but a hard dependency of the tree)

`overlay: gpu_select.py` + `patches: engine_engine (part) / server_* / daemon_* /
checkpoint_* / moe_host_banks / moe_benchbw / moe_bench_profile / kernel_fla_* /
kernel_triton_{e4m3_compat,sampling} / models_deepseek_v4_moe / scheduler_scheduler`:
multi-GPU selection (--gpu UUID), host-bank pin classes (WSL/WDDM), per-GPU bench
profiles, assorted device-index fixes. engine.py imports gpu_select at module
level, so this group must ship with the rest.

## DeepSeek-V4-Flash (DSV4) line

Serving DSV4-Flash (ds_fp4 experts, 43 layers, DSA compressor/indexer, mHC) on the
same box. Baseline after the round-2 kernels: 55 tok/s single-stream, conc8 164 tok/s
aggregate (moe-cache-auto = 6000 experts resident, ~75 GB).

| Item | Measured | Files | Switch |
|---|---|---|---|
| Native-FP8 wo_a grouped GEMV (M<=16 single launch) + FP8 lm_head + fused act-quant GEMV | single 51.2 -> 55.0, conc2 +7% | `patches: kernel_triton_dsv4_fp8_linear / models_deepseek_v4_{attention,model,weight}` | `FREETOKEN_DSV4_WOA_FP8` / `HEAD_FP8` (1) |
| Short-extend on-demand prefill threshold 48 -> 512 | cold 60-500-token TTFT 1.7 s -> 0.5-0.8 s; warm long-prompt extends 0.73-0.8 s; quality 15/15. Threshold 1024 tested and REJECTED: the single-request crossover sits at ~1200-1500 tokens, but 500-1024-token on-demand prefills thrash the expert LRU and push cold streaming (>1024) from ~1.7 s to ~2.5 s | `patches: layers_moe` (threshold) + `examples/serve_dsv4.sh` | `FREETOKEN_PREFILL_ONDEMAND_TOKENS` (512) |
| e8m0 scale-slab index_copy fix (on-demand prefill hit path crashed the engine for any extend > threshold) | correctness fix | `patches: layers_moe` | none |
| spec_alloc_len comfort gate in tokens (page_size-aware; was permanently 0 horizon on 128-token pages) | correctness fix for any spec on paged models | `patches: scheduler_cache` | none |

### DSpark speculative decoding (built, verified, ARCHIVED -- net-negative on offload)

Full port of the checkpoint's 3-stage DSpark MTP head (block-parallel 5-token draft,
markov logit bias, confidence head, per-stage 128-slot main_kv rings): weights loaded
to VRAM (10.3 GB carved from the expert cache by moe-cache-auto), draft runs zero-PCIe;
verify = k+1 staggered co-tenant single-token decode rows with the compressor carry
chained through registers (fused ratio-4 Triton chain + vectorized ratio-128 path),
captured as a standalone CUDA graph; rollback = ring-carry archive/restore + re-chain
over the accepted prefix (all pool writes are position-addressed, junk is
overwritten-before-attended).

Measured: 3.68 tok/round accepted (33% of rounds take all 5 drafts) -- the draft head
is excellent. Net throughput NEGATIVE: ~42 tok/s vs 55 baseline. Two compounding
costs: the 10.3 GB cache carve alone drops the spec-off baseline to 43.3, and the
6-row verify pays non-amortizing PCIe expert fetches (63 ms vs 18 ms single-row).
Same verdict class as the GLM MTP attempt (-13%). Kept env-gated OFF.

| Files | Switch |
|---|---|
| `overlay: models/deepseek_v4/{dspark,spec,dspark_load}.py`, `kernel/triton/dsv4/spec_chain.py`; `patches: models_deepseek_v4_{model,attention,compress} / engine_engine / scheduler_cache` | `FREETOKEN_DSV4_SPEC` (0) + `FREETOKEN_DSV4_SPEC_CKPT` (checkpoint dir); diagnostics: `FREETOKEN_DSV4_SPEC_LOG`, `FREETOKEN_DSV4_SPEC_EAGER`, `FREETOKEN_DSV4_SPEC_CONF` |

### DSV4 round 3 (single-stream 55 -> 58+)

| Item | Measured | Files | Switch |
|---|---|---|---|
| Fused compressor decode step (register roll: read/scatter/pool/promote/write in ONE launch x 62 tier-instances, ape in-kernel; ratio-128 pool row-TILED -- the first cut's serial 128-load chain cost 52 us/launch) | +1.1 then +2.3 more once tiled (55 -> 56 -> 60.4, quality 15/15, conc2 84.4) | `overlay: kernel/triton/dsv4/comp_step.py` + `patches: models_deepseek_v4_compress` | `FREETOKEN_DSV4_FUSED_COMP` (1) |
| sqrtsoftplus fused route (softplus/sqrt/+bias/topk/gather/renorm/scale, ~8 launches -> 1 x 40 layers; stable softplus form) | +2.2 tok/s (56 -> 58.1), quality 15/15 | `overlay: kernel/triton/fused_route.py` (ACT constexpr) + `patches: models_deepseek_v4_moe` | `FREETOKEN_FUSED_ROUTE` (1) |
| memory-ratio 0.95 + cache 6200 | no gain (hit already 97.6%) | -- | rejected |
| Layer-invariant decode addressing memoized (ring block bases, compressed-block rows, compressor freqs -- ~370 int64 kernels/step -> ~10) | +1.6 tok/s (60.4 -> 62) | `patches: models_deepseek_v4_compress` | none (dctx memo) |
| mHC pre-mix: rms folded into the mix GEMV (bf16 read, redundant per-program ssq; kills the rms_cast launch + fp32 intermediate). A one-CTA full fusion was tried first and DESTROYED parallelism (39 tok/s) -- parallel grid shape is the point | +1.9 tok/s (62 -> 64) | `overlay: kernel/triton/dsv4/hc_fused.py` + `patches: models_deepseek_v4_model` | `FREETOKEN_DSV4_HC_FUSED` (1) |
| swiglu + down-side FP8 round-trip fused (QUANT constexpr in the swiglu kernel) | ~0 measured (sub-noise), kernel count -80/step | `patches: kernel_triton_dsv4_fused_moe / moe_fused_ds_fp4` | none |
| fp8 GEMV Blackwell config sweep | big shapes already at 98% of GDDR7 peak (1.77 TB/s); no config wins | -- | rejected |

Final (2026-09-01): single 64.1 (+16.5% over the round-2 55), conc2 87.5, conc4 128.6,
conc8 167.2, quality 15/15. Measurement discipline: a fresh boot underreports by up
to 6 tok/s for the first 1-2 matrix passes (expert-cache convergence) -- always run
the matrix twice and read the second pass. The one-CTA hc fusion and an early serial
128-row pool loop were both caught by this only after a false negative/positive each.

Validation (2026-09-01): 10.6K-check A/B soak (fusions on vs off, exec-verified
codegen / long-gen / deep-context retrieval / concurrency): zero failures outside
one pre-existing model mode (rare adjacent-function retrieval flip on greedy
near-ties in 60-150-function synthetic modules; present at the same order in BOTH
numerics arms, replay-correct). Tonight's kernels are behaviorally clean.
