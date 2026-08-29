#!/bin/bash
# GLM-5.3-Flash FULL 45-layer serve -- hotness-driven resident split.
#   resident layers 20-27 (middle of the U-shaped miss curve): 8 x 3.87G = 31G VRAM,
#   zero PCIe for them. Host pin drops to ~129G -> fits the current 157G box.
#   Backend auto resolves to triton (swiglu_clamp); b12x is correctly rejected.
set -u
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy
unset FREETOKEN_PIN_BUDGET_GB FREETOKEN_BANK_CUDA_ALLOC FREETOKEN_SKIP_BANK_PIN
unset FREETOKEN_GLM5_MAX_LAYERS
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/home/zrx/miniconda3/envs/freetoken/bin:$CUDA_HOME/bin:$PATH
export HF_HOME=/home/zrx/glm5-stack/hf_home
export HF_HUB_OFFLINE=1
export FREETOKEN_GLM5_RESIDENT_LAYERS=${FREETOKEN_GLM5_RESIDENT_LAYERS:-3-6,8-11}
# Non-expert FP8: +28% decode, 2.7x prefill; quality gates passed 2026-08-28 (tool
# calls, >2048 needle, coherence). lm_head stays BF16 (rare-word protection).
export FREETOKEN_GLM5_ATTN_FP8=${FREETOKEN_GLM5_ATTN_FP8:-1}
export FREETOKEN_GLM5_MLP_FP8=${FREETOKEN_GLM5_MLP_FP8:-1}
export FREETOKEN_GLM5_KDA_FP8=${FREETOKEN_GLM5_KDA_FP8:-1}
# Speculative expert prefetch (P=4 hop-1 no-wait): +8%% single-stream, conc-neutral,
# zero quality risk (pure cache warming); sweep 2026-08-28: P=2/4/6/8 -> 22.6/22.9/22.4/20.1
export FREETOKEN_MOE_SPEC_PREFETCH=${FREETOKEN_MOE_SPEC_PREFETCH:-4}
# Vision (image input on both APIs): +~1.2 GB VRAM for the BF16 tower; needs pillow.
export FREETOKEN_GLM5_VISION=${FREETOKEN_GLM5_VISION:-1}
# 128 covers typical multi-turn extends after a radix prefix hit (turn-2 TTFT 2.84->1.13s)
export FREETOKEN_PREFILL_ONDEMAND_TOKENS=${FREETOKEN_PREFILL_ONDEMAND_TOKENS:-128}

MODEL=/home/zrx/glm5-stack/weights/GLM-5.3-Flash-NVFP4
PORT="${GLM5_PORT:-1920}"
CTX=${GLM5_CTX:-262144}
# KV pool (tokens) decoupled from the per-request cap: 4-slot 256K = pool 1M.
KV_RESERVE=${GLM5_KV_RESERVE:-262144}

systemctl --user stop dsv4-freetoken.service 2>/dev/null || true
systemctl --user stop qwen38-vllm.service 2>/dev/null || true
for p in firefox update-manager gnome-software snap-store tracker-miner tracker3; do pkill -f "$p" 2>/dev/null || true; done
sync
AVAIL=$(awk "/^MemAvailable:/{print int(\$2/1048576)}" /proc/meminfo)
[ "$AVAIL" -ge 138 ] || { echo "FATAL: MemAvailable ${AVAIL}G < 138G (129G bank pin + engine)"; exit 1; }
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$USED" -le 2000 ] || { echo "FATAL: ${USED}MiB VRAM in use"; exit 1; }
echo "preflight ok: MemAvailable ${AVAIL}G; cold load ~17G dense + 31G resident experts + 129G host banks (expect ~1h serial)"

exec ft serve \
  --model "$MODEL" \
  --served-model-name glm5-flash \
  --host 0.0.0.0 --port "$PORT" \
  --moe-backend ${GLM5_MOE_BACKEND:-offload} \
  --moe-cpu-threads ${GLM5_CPU_THREADS:-0} \
  --moe-hybrid-max-fetch ${GLM5_HYBRID_MAXFETCH:-1} \
  --expert-load serial \
  --nvfp4-backend auto \
  --moe-cache-auto \
  --moe-prefill-hit-d2d \
  --cache-type ${GLM5_CACHE_TYPE:-naive} \
  --memory-ratio ${GLM5_MEMRATIO:-0.88} \
  --kv-reserve-tokens "$KV_RESERVE" \
  --max-seq-len-override "$CTX" \
  --max-running-requests 8 \
  --sampling-defaults model
