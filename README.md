# freetoken-ox-boost

[FreeToken](https://github.com/FlashML-org/FreeToken) v0.1.2 的 GLM-5.3-Flash 适配
+ 单卡 offload 性能优化补丁集(补丁插件形式,不分发上游代码)。

在 RTX PRO 6000 Blackwell 96G 单卡上跑 GLM-5.3-Flash-NVFP4(181G 专家权重,
host 内存 offload,PCIe Gen5 x16),单流 decode 从 **17.8 → 35 tok/s(约 2×)**,
2 路并发聚合 44.9(暖话题)、4 路聚合 ~49。质量闸零损;
48/48 步与 HF 参考实现逐 token 一致)。

因为 GLM-5.3-Flash 路由太平(每 token 激活的 8 个专家彼此几乎不重叠),并发请求的专家取数字节按流数线性叠加、还互相瓜分缓存,而 PCIe 链路早已打满—，批处理摊薄不了这台机器真正计费的东西:字节。

## 实测指标一览

速度对上下文长度与缓存暖度敏感,表中标注测量条件。

| 指标 | 实测值 | 条件 |
|---|---|---|
| 单流 decode | **35.0 tok/s**(33.8-36.3) | 2 槽 256K 池,暖缓存 |
| 2 路并发聚合 | 44.9 tok/s(暖)/ 36-37(持续冷话题) | 每流 22.4 / ~18 |
| 4 路并发聚合 | ~49 tok/s | 每流 ~12 |
| MoE 缓存命中率 | **~84%**(miss 0.157) | 常驻 8 层 + P=4 预取 @2079 槽;现 2600 槽略优 |
| TTFT(10 token 短 prompt) | **0.59 s** | 按需 prefill 路径(整层流式恒 ~2.1s) |
| TTFT(40 token) | 0.95 s | 同上 |
| TTFT(2.7K token) | 3.12 s(prefill 852 tok/s) | 流式 prefill |
| 232K 长上下文 prefill | 232 s(~1030 tok/s) | 10%/50%/95% 三针 needle 全中 |
| 232K 上下文后 decode | 25.1 tok/s | 单槽 256K 配置实测,几乎无损 |
| 质量 | MMLU-100 **94**(bf16 89 / fp8 87) | 48/48 步与 HF 参考逐 token 一致 |

启动参考 `examples/serve_full.sh`(配置、路径按自己机器改)。

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
