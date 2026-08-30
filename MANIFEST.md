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
