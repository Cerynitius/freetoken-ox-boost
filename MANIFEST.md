# 功能 → 文件 → 开关 对照表

基线:FreeToken v0.1.2(commit 9db1a39)。所有速度数字为 RTX PRO 6000 Blackwell 96G
单卡 + GLM-5.3-Flash-NVFP4(181G 专家,host offload)实测,单流 decode。

## 核心适配(无开关,装上即生效)

| 功能 | 文件 | 说明 |
|---|---|---|
| GLM5-Next 模型(KDA 34 层 + MLA/DSA 11 层 + mHC + MoE) | `overlay: models/glm5_next/*`;`patches: models_register / models_config / models_glm_moe_dsa_args` | 48/48 步 vs HF 参考逐 token 一致 |
| NoPE guard(rotary_dim=0) | `patches: models_glm_moe_dsa_attention / kernel_triton_glm_dsa_sparse` | D_R=0 constexpr 分支 |
| DSA k-pool 稀疏 + 池化打分 | `patches: attention_dsa / kvcache_dsa_pool / kvcache___init__ / models_glm5_next(config)` | 池化 key 逐位=HF;>2048 needle 实战通过 |
| clamped swiglu(routed 专家走 triton) | `patches: moe_expert_banks / moe_fused_nvfp4(部分) / models_glm_moe_dsa_moe(部分)` | b12x epilogue 无 clamp,auto→triton |
| MTP 层权重跳过(层 45 丢弃省 3.9G) | `overlay: models/glm5_next/weight.py` | 投机已归档,见下 |

## 性能优化(env 可控,括号内为默认)

| 优化 | 实测 | 文件 | 开关 |
|---|---|---|---|
| 非专家 FP8(attn/mlp) | +28%,prefill 2.7× | `models/glm5_next/{attention,mlp,weight}.py` | `FREETOKEN_GLM5_ATTN_FP8`/`MLP_FP8`(1) |
| KDA FP8(in_proj+o_proj) | +15% | 同上 | `FREETOKEN_GLM5_KDA_FP8`(1) |
| 热度常驻层 3-10 | +29%(miss 0.24→0.157) | `models/glm5_next/experts_resident.py` + `patches: models_glm_moe_dsa_moe` | `FREETOKEN_GLM5_RESIDENT_LAYERS`(3-10) |
| 投机专家预取(L+1 gate 提前打分) | +8%(P=4 hop-1 甜点) | `overlay: moe/spec_prefetch.py` + `patches: layers_moe` | `FREETOKEN_MOE_SPEC_PREFETCH`(0;生产设 4) |
| 短 prompt 按需 prefill | TTFT 3.5×(10-tok 2.1→0.59s) | `patches: layers_moe` | `FREETOKEN_PREFILL_ONDEMAND_TOKENS`(48;0=关) |
| marlin decode GEMV 配置 16/32/2 | +4.5%(gate_up 核 −15%) | `patches: moe_fused_nvfp4` | `FREETOKEN_MARLIN_BN/BKW/WARPS`(16/32/2) |
| KDA gate 融合(5 GEMV→2 + 7 散核→1) | +1% | `overlay: kernel/triton/kda_gate.py` + `models/glm5_next/attention.py` | `FREETOKEN_KDA_FUSED_GATE`(1) |
| mHC 前置融合(cast/mean/rsqrt 合一 + 免原子 gemv) | 噪声内(核数 −6/层) | `overlay: kernel/triton/dsv4/hc_norm.py` + `models/glm5_next/model.py` | 无开关(数值 epsilon 级) |
| 路由融合(sigmoid+bias+topk+renorm 8 核→1) | +3.4% | `overlay: kernel/triton/fused_route.py` + `patches: models_glm_moe_dsa_moe` | `FREETOKEN_FUSED_ROUTE`(1;n_group>1 自动回退) |
| FP8 小批 M-tile GEMV(2≤M≤4 免入 prefill GEMM) | conc2 +13%(M=2 核 3×) | `patches: kernel_triton_fp8_pertensor_linear` | 无开关(M≤4 自动) |
| 2 槽 256K KV 池(缓存 2079→2600 槽) | 单流 +14%,conc4 +67% | `examples/serve_full.sh` | `GLM5_KV_RESERVE`(524288) |
| CPU swiglu_clamp 支持(hybrid 后端可跑 GLM-5.3) | 本机 VM 判负,内核保留 | `patches: kernel_csrc…cpu_moe_ext / moe_cpu_executor / layers_moe(bs 门控)` | `GLM5_MOE_BACKEND=hybrid` + `GLM5_CPU_THREADS` |

## 实验存档(默认关;判决见 README)

| 实验 | 判决 | 开关 |
|---|---|---|
| MTP 投机解码(完整 verify 引擎 + CUDA graph) | 正确但 PCIe 计费下 −13%,归档 | `FREETOKEN_GLM5_SPEC`(0) |
| LFU-decay 缓存策略 | 真负载反输(漂移型局部性) | `FREETOKEN_MOE_CACHE_POLICY`(lru) |
| fallback-free miss-cap(专家丢弃) | 首测 −35%(盲序 drop),待二审 | `FREETOKEN_MOE_MISS_CAP`(-1) |
| top-k 旋钮 | top-6 +6% 但改 A18B 规格,否决 | `FREETOKEN_GLM5_TOPK`(不设=8) |
| 路由踪迹采集 | 分析工具 | `FREETOKEN_ROUTE_TRACE`(关) |
| bs>1 独立预取 P/hop | 扫参判定恒 P=4/hop-1 最优 | `FREETOKEN_MOE_SPEC_PREFETCH_MULTI`(-1=跟随)/`SPEC_HOP_MULTI`(1) |
| hybrid CPU 分担 | conc4 净贡献≈0(QEMU vCPU),判负存档 | `GLM5_MOE_BACKEND`(offload) |
| decode 步 profiler | 分析工具 | `FREETOKEN_PROFILE_DECODE`(关) |

## 基础设施(patches 01 组,非本线优化但为当前树硬依赖)

`overlay: gpu_select.py` + `patches: engine_engine(部分) / server_* / daemon_* / checkpoint_* /
moe_host_banks / moe_benchbw / moe_bench_profile / kernel_fla_* / kernel_triton_{e4m3_compat,sampling} /
models_deepseek_v4_moe / scheduler_scheduler`:多卡 GPU 选择(--gpu UUID)、host bank
pin 分级(WSL/WDDM)、per-GPU bench profile、若干 device-index 修正。engine.py 顶层
import gpu_select,故必须随包。
