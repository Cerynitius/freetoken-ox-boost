# freetoken-ox-boost

[FreeToken](https://github.com/FlashML-org/FreeToken) v0.1.2 的 GLM-5.3-Flash 适配
+ 单卡 offload 性能优化补丁集(补丁插件形式,不分发上游代码)。

在 RTX PRO 6000 Blackwell 96G 单卡上跑 GLM-5.3-Flash-NVFP4(181G 专家权重,
host 内存 offload,PCIe Gen5 x16),单流 decode 从 **17.8 → 35 tok/s(约 2×)**,
2 路并发聚合 44.9(暖话题)、4 路聚合 ~49。质量闸零损
(MMLU-100:本补丁集 94 / bf16 89 / fp8 87,历次最高;
48/48 步与 HF 参考实现逐 token 一致)。

启动参考 `examples/serve_full.sh`(路径按自己机器改;4 槽 256K 上下文 + 全部优化默认开)。

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
