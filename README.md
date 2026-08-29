# freetoken-ox-boost

[FreeToken](https://github.com/FlashML-org/FreeToken) v0.1.2 的 GLM-5.3-Flash 适配
+ 单卡 offload 性能优化补丁集(补丁插件形式,不分发上游代码)。

在 RTX PRO 6000 Blackwell 96G 单卡上跑 GLM-5.3-Flash-NVFP4(181G 专家权重,
host 内存 offload,PCIe Gen5 x16),单流 decode 从 **17.8 → 35 tok/s(约 2×)**,
2 路并发聚合 44.9(暖话题)、4 路聚合 ~49。质量闸零损
(MMLU-100:本补丁集 94 / bf16 89 / fp8 87,历次最高;
48/48 步与 HF 参考实现逐 token 一致)。

启动参考 `examples/serve_full.sh`(路径按自己机器改;4 槽 256K 上下文 + 全部优化默认开)。

## 提速账本(逐项 A/B 实测)

| 阶段 | 单流 tok/s | 手段 |
|---|---|---|
| 基线(bf16 全量 + CUDA graphs) | 17.8 | |
| + 非专家 FP8(attn/mlp,lm_head 保 bf16) | 23.6 | +28%,prefill 2.7× |
| + KDA FP8(in_proj/o_proj per-row) | 25.4 | +15%(fla in-kernel l2norm 洗掉尺度误差) |
| + 4 槽 256K 池(容量代价,后改 2 槽收回) | 21.2 | −18%(moe cache −26%) |
| + 投机专家预取 P=4(L+1 gate 提前打分) | 22.9 | +8% |
| + 常驻层重挑 3-10(5.3K 步踪迹全局 LRU 仿真) | 27.4 | +29%,全程最大单项 |
| + marlin decode 配置 16/32/2(27 配置微基准扫描) | 29.3 | +4.5% |
| + KDA gate 融合 + mHC 融合 | 29.6 | 每层 ~18 个散核并成 4 个 |
| + 路由融合(8 核链→1 核,topk 语义逐 id 一致) | 30.6 | +3.4% |
| + 2 槽 256K KV 池(缓存 2079→2600 槽) | **35.0** | conc4 26→44 |
| + FP8 小批 M-tile GEMV(M≤4 免入 prefill GEMM) | 35.0 | conc2 39.8→44.9(M=2 核 3×) |

## 目录

```
patches/    39 个 per-file unified diff(对 v0.1.2,git apply -p1)
overlay/    17 个新文件(models/glm5_next 整目录 + 4 个 triton 核 + 预取/LFU + gpu_select)
install.sh  校验版本 → dry-run → 应用 → compileall
examples/   生产启动脚本(2 槽 256K,全优化默认开)
MANIFEST.md 功能 → 文件 → 开关 → 实测数字 对照表
```

## 许可

与上游一致,Apache-2.0。本仓库只含我们编写的补丁与新文件,不含上游代码本体。
