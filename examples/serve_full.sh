#!/bin/bash
# GLM-5.3-Flash (NVFP4) full serve on one RTX PRO 6000 Blackwell 96 GB + 186 GB host RAM (DDR5 at 3600 MT/s,
# which caps host->GPU expert copies at ~47 GB/s). Adjust paths, memory ratio and preflight for your machine.
#
# Every switch below is lossless: greedy outputs were checked token-for-token against the plain config
# (text, prefix-cache hits, images). Measured 2026-10-02, ~0.9k-token prompts, 1024 generated tokens, T=0:
#   single stream ~38 tok/s, 2 concurrent ~45 tok/s aggregate; decode stays ~37 tok/s at 128K context.
#
# Memory: 8 resident layers (3-6, 8-11) live in VRAM; the other 34 MoE layers are a pinned host bank (~129 GiB)
# plus an LRU/soft expert cache in VRAM sized by --memory-ratio. 0.96 needs --max-extend-length 4096 (halves the
# prefill activation peak; 0.95 OOMed on a 15K-token prompt at 8192) and leaves ~3 GiB free after startup, which is
# why vision attention runs in budgeted query-row chunks and multimodal prompts may span prefill chunks.
set -u
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy
unset FREETOKEN_PIN_BUDGET_GB FREETOKEN_BANK_CUDA_ALLOC FREETOKEN_SKIP_BANK_PIN FREETOKEN_GLM5_MAX_LAYERS
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}          # a CUDA 13.x toolkit (JIT kernels are built at startup)
export PATH=$CUDA_HOME/bin:$PATH
export HF_HUB_OFFLINE=1

# --- model layout and precision
export FREETOKEN_GLM5_RESIDENT_LAYERS=${FREETOKEN_GLM5_RESIDENT_LAYERS:-3-6,8-11}   # hottest-fetch layers stay in VRAM
export FREETOKEN_GLM5_ATTN_FP8=${FREETOKEN_GLM5_ATTN_FP8:-1}     # non-expert FP8 W8A16 (lm_head stays BF16)
export FREETOKEN_GLM5_MLP_FP8=${FREETOKEN_GLM5_MLP_FP8:-1}
export FREETOKEN_GLM5_KDA_FP8=${FREETOKEN_GLM5_KDA_FP8:-1}
export FREETOKEN_GLM5_VISION=${FREETOKEN_GLM5_VISION:-1}         # image input (~1.2 GB VRAM)
export FREETOKEN_ELASTIC_KV=${FREETOKEN_ELASTIC_KV:-1}           # 1M logical context; KV grows by retiring expert slots

# --- expert cache and copy overlap
export FREETOKEN_MOE_SPEC_PREFETCH=${FREETOKEN_MOE_SPEC_PREFETCH:-4}      # predicted next-layer experts (P=4 best)
export FREETOKEN_MOE_SPLIT_OVERLAP=${FREETOKEN_MOE_SPLIT_OVERLAP:-1}      # cached experts compute while misses copy
export FREETOKEN_MOE_CACHE_POLICY=${FREETOKEN_MOE_CACHE_POLICY:-soft}     # score-aware policy ...
export FREETOKEN_GLM5_SOFT_RANK=${FREETOKEN_GLM5_SOFT_RANK:-1}            # ... fed by the router's near-misses
export FREETOKEN_GLM5_GATE_FP32_CACHE=${FREETOKEN_GLM5_GATE_FP32_CACHE:-1}
export FREETOKEN_GLM5_SHARED_OVERLAP=${FREETOKEN_GLM5_SHARED_OVERLAP:-1}  # shared expert runs during the miss copy

# --- fused decode kernels (bit-identical)
export FREETOKEN_FP8_GEMV_FUSED_REDUCE=${FREETOKEN_FP8_GEMV_FUSED_REDUCE:-1}
export FREETOKEN_GLM5_MLP_FUSED=${FREETOKEN_GLM5_MLP_FUSED:-1}
export FREETOKEN_GLM5_DSA_GLUE=${FREETOKEN_GLM5_DSA_GLUE:-1}
export FREETOKEN_GLM5_HC_FUSED=${FREETOKEN_GLM5_HC_FUSED:-1}              # mode 1 only (mode 2 is not bit-identical)
export FREETOKEN_NVFP4_SWIGLU_FUSED=${FREETOKEN_NVFP4_SWIGLU_FUSED:-1}

# --- prefill and multimodal
export FREETOKEN_PREFILL_ONDEMAND_TOKENS=${FREETOKEN_PREFILL_ONDEMAND_TOKENS:-128}  # short extends fetch only used experts
export FREETOKEN_GLM5_VISION_ATTN_CHUNK=${FREETOKEN_GLM5_VISION_ATTN_CHUNK:-512}    # + FREETOKEN_GLM5_VISION_ATTN_BUDGET_MB (256)
export FREETOKEN_MM_CHUNKED_PREFILL=${FREETOKEN_MM_CHUNKED_PREFILL:-1}              # large images span prefill chunks

MODEL=${GLM5_MODEL:-/path/to/GLM-5.3-Flash-NVFP4}
PORT=${GLM5_PORT:-1920}
CTX=${GLM5_CTX:-1048576}
KV_RESERVE=${GLM5_KV_RESERVE:-262144}

AVAIL=$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)
[ "$AVAIL" -ge 150 ] || { echo "FATAL: MemAvailable ${AVAIL}G < 150G (129G host bank pin + engine + margin)"; exit 1; }
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$USED" -le 2000 ] || { echo "FATAL: ${USED} MiB VRAM already in use"; exit 1; }
echo "preflight ok: MemAvailable ${AVAIL}G, ctx ${CTX}"

exec ft serve \
  --model "$MODEL" \
  --served-model-name ${GLM5_NAME:-glm5-flash} \
  --host 0.0.0.0 --port "$PORT" \
  --moe-backend offload \
  --expert-load auto \
  --nvfp4-backend auto \
  --moe-cache-auto \
  --moe-prefill-hit-d2d \
  --cache-type ${GLM5_CACHE_TYPE:-radix} \
  --memory-ratio ${GLM5_MEMRATIO:-0.96} \
  --max-extend-length ${GLM5_MAX_EXTEND:-4096} \
  --kv-reserve-tokens "$KV_RESERVE" \
  --max-seq-len-override "$CTX" \
  --max-running-requests ${GLM5_MAX_RUNNING:-2} \
  --sampling-defaults model \
  "$@"
