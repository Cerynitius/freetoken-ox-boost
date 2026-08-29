# freetoken-ox-boost

[FreeToken](https://github.com/FlashML-org/FreeToken) v0.1.2 的 GLM-5.3-Flash 适配
+ 单卡 offload 性能优化补丁集(补丁插件形式,不分发上游代码)。

在 RTX PRO 6000 Blackwell 96G 单卡上跑 GLM-5.3-Flash-NVFP4(181G 专家权重,
host 内存 offload,PCIe Gen5 x16),单流 decode 从 **17.8 → 35 tok/s(约 2×)**,
2 路并发聚合 44.9(暖话题)、4 路聚合 ~49。质量闸零损;
48/48 步与 HF 参考实现逐 token 一致)。

因为 GLM-5.3-Flash 路由太平(每 token 激活的 8 个专家彼此几乎不重叠),并发请求的专家取数字节按流数线性叠加、还互相瓜分缓存,而 PCIe 链路早已打满—，批处理摊薄不了这台机器真正计费的东西:字节。

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
