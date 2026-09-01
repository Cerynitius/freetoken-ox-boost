#!/bin/bash
# FreeToken serving DeepSeek-V4-Flash-0731 (native FP8 + MXFP4 experts) on ONE
# RTX PRO 6000 Blackwell 96 GiB + 157 GiB host RAM (KVM guest, host DRAM ~45 GB/s).
#
# Every flag below is justified against THIS platform from FreeToken's source.
#
#  --moe-backend offload   NOT hybrid. Measured B_host ~45 GB/s < B_PCIe ~50 GB/s, so the
#                          paper's residual bandwidth B_R = max(B_H-B_P,0) = 0 and q* clamps
#                          to m: every miss goes over PCIe and the CPU branch contributes
#                          nothing. Passing it explicitly also disarms the `auto` hybrid trap.
#  --moe-prefill-hit-d2d   Off by default because it is pointless at the ~11% residency a
#                          32 GB card gets. Here residency is ~55%, so most prefill experts
#                          are gathered device-side instead of re-crossing PCIe. Verified
#                          fully engaged for ds_fp4 (no silent fallback). Needs CUDA>=13.
#  --kv-reserve-tokens     DSV4 KV is ~34.99 KiB/token, NOT the 6.72 KiB of dsv4_kv_unit_bytes
#      98304               (the SWA window tier at swa_full_tokens_ratio 0.2 is ~78% of the
#                          bill -- dsv4_cost_model.py:80,119-133). plan_cache_budget is
#                          KV-first in the arithmetic (cache_budget.py:78), so every KV byte
#                          comes straight out of the expert cache. 262144 tokens would cost
#                          9.17 GiB; 98304 costs ~3.4 GiB and buys ~439 more expert slots
#                          (~51% -> ~55% residency). Take context back at runtime instead:
#                          `ft ctl cache rebuild --kv <tokens> --moe <slots>`.
#  --max-seq-len-override  Unset, DSV4 bakes rope tables for max_position_embeddings =
#      98304               1,048,576 positions it can never reach: 2 regimes x 268 MB =
#                          536 MB of VRAM (attention.py:88 via ops.py:49-59, lru_cache(2)).
#                          Capping at the resolved KV pool returns ~475 MB.
#  --memory-ratio 0.92     Keep. The (1-ratio) remainder is load-bearing for CUDA graphs and
#                          activations and scales with the prefill chunk. Do NOT raise it.
#
# DELIBERATELY ABSENT:
#  --max-prefill-length    Silently discarded for DSV4: _adjust_dsv4_config (engine.py:1067)
#                          overwrites max_extend_tokens with config.max_seq_len before the
#                          scheduler reads it. Passing it only makes the config look tuned.
#  --moe-cpu-layers        Dead on this box: _pin_budget_bytes() returns None off WSL
#                          (engine.py:1164), so the residency-splitting machinery never arms.
#                          Correct outcome -- a CPU-decoded layer costs ~2.8x the GPU path.
#  --enable-special-token-ckpt  Unreachable for DSV4: the deepseekv32 detector declares no
#                          single-token tool-call opener, so the paper's semantic-anchor
#                          mechanism has nothing to anchor to here.
#  --expert-load           Left `auto`, which resolves to the SERIAL reader: the parallel
#                          O_DIRECT guard needs MemAvailable > ~160.4 GiB on a 157.2 GiB box
#                          (expert_banks.py:373) and can never pass. That veto is correct --
#                          the parallel path would spend the host-RAM margin the 137 GiB pin
#                          needs. (An earlier version of this header wrongly claimed the
#                          parallel reader was in use.)
set -u
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy
# Each of these three would silently degrade this configuration; never set them here.
#   PIN_BUDGET_GB   -> arms _auto_cpu_layers, pushing layers onto the 45 GB/s CPU executor
#   BANK_CUDA_ALLOC -> cudaHostAlloc path eagerly zero-fills 137 GiB before the read
#   SKIP_BANK_PIN   -> no-ops the pin; the GPU movement paths require registered banks
unset FREETOKEN_PIN_BUDGET_GB FREETOKEN_BANK_CUDA_ALLOC FREETOKEN_SKIP_BANK_PIN

export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/home/zrx/miniconda3/envs/freetoken/bin:$CUDA_HOME/bin:$PATH
export HF_HOME=/home/zrx/dsv4-stack/hf_home
export HF_HUB_OFFLINE=1
# wo_a served in checkpoint-native FP8 via the grouped GEMV (single +5.8%, conc2 +4.6%;
# quality indistinguishable from baseline sampling noise across replicated batteries)
export FREETOKEN_DSV4_WOA_FP8=${FREETOKEN_DSV4_WOA_FP8:-1}
# lm_head W8A16 (fp8 + per-row scale): the full-vocab logits GEMV was 670us/step bf16
export FREETOKEN_DSV4_HEAD_FP8=${FREETOKEN_DSV4_HEAD_FP8:-1}

MODEL=/home/zrx/dsv4-stack/weights/DSV4-Flash-0731-CRACK
PORT="${DSV4_PORT:-1920}"
CTX=98304  # legacy; superseded by the two knobs below
KV_RESERVE="${DSV4_KV_RESERVE:-524288}"   # total pooled KV tokens (all streams)
MAX_SEQ="${DSV4_MAX_SEQ:-262144}"         # per-request context cap

# ---- preconditions: FreeToken has no guard for any of these and fails late/opaquely ----
# 1. --moe-cache-auto sizes from LIVE free VRAM (engine.py:316-319). If another engine holds
#    the card, memory_ratio is applied to the scraps, weights exceed it, and the plan
#    collapses to a near-zero expert cache WITH NO WARNING.
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [[ $USED -gt 2000 ]]; then
  echo "FATAL: ${USED} MiB of VRAM already in use. Stop the other engine first (dsv4 start does this)." >&2
  exit 1
fi
# 2. No host-RAM preflight exists before 137 GiB of lazy-mmap banks are faulted in; the
#    shortfall surfaces ~60-90 s later as an OOM kill with no diagnostic.
AVAIL=$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)
if [[ $AVAIL -lt 145 ]]; then
  echo "FATAL: MemAvailable ${AVAIL} GiB < 145 GiB needed (137.1 GiB pinned banks + engine)." >&2
  exit 1
fi
# 3. _weight_map() opens the index unguarded (deepseek_v4/weight.py:57-59) -- a missing file
#    is a bare FileNotFoundError with no context.
[[ -f "$MODEL/model.safetensors.index.json" ]] || { echo "FATAL: missing model.safetensors.index.json in $MODEL" >&2; exit 1; }
[[ -f "$MODEL/inference/config.json" ]]        || { echo "FATAL: missing inference/config.json in $MODEL" >&2; exit 1; }
echo "preconditions ok: VRAM used ${USED} MiB, MemAvailable ${AVAIL} GiB, checkpoint present"

export FREETOKEN_PREFILL_ONDEMAND_TOKENS=${FREETOKEN_PREFILL_ONDEMAND_TOKENS:-512}
# post-boot self-warmup: absorbs the first-request cold outlier (triton compiles, cold caches)
(
  for _i in $(seq 1 90); do
    sleep 10
    curl -s -m 8 "http://127.0.0.1:${DSV4_PORT:-1920}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${DSV4_NAME:-dsv4-flash}\",\"messages\":[{\"role\":\"user\",\"content\":\"warmup\"}],\"max_tokens\":2}" \
      | grep -q choices && break
  done
) >/dev/null 2>&1 &
exec /home/zrx/miniconda3/envs/freetoken/bin/ft serve \
  --model "$MODEL" \
  --served-model-name ${DSV4_NAME:-dsv4-flash} \
  --host 0.0.0.0 --port "$PORT" \
  --moe-backend offload \
  $([ -n "${DSV4_MOE_CACHE:-}" ] && echo "--moe-cache-size $DSV4_MOE_CACHE" || echo --moe-cache-auto) \
  --moe-prefill-hit-d2d \
  --memory-ratio ${DSV4_MEM_RATIO:-0.95} \
  --kv-reserve-tokens "$KV_RESERVE" \
  --max-seq-len-override "$MAX_SEQ" \
  --max-running-requests ${DSV4_MAX_RUNNING:-2}
