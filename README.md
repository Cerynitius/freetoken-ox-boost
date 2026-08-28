# freetoken-glm5-boost

[FreeToken](https://github.com/FlashML-org/FreeToken) v0.1.2 的 GLM-5.3-Flash 适配
+ 单卡 offload 性能优化补丁集(补丁插件形式,不分发上游代码)。

在 RTX PRO 6000 Blackwell 96G 单卡上跑 GLM-5.3-Flash-NVFP4(181G 专家权重,
host 内存 offload,PCIe Gen5 x16),单流 decode 从 **17.8 → 30.6 tok/s(+72%)**,
质量闸零损(MMLU-100:本补丁集 88 / fp8 基线 87 / bf16 89,同噪声带;
48/48 步与 HF 参考实现逐 token 一致)。

## 安装

```bash
git clone https://github.com/FlashML-org/FreeToken && cd FreeToken && git checkout v0.1.2
cd .. && git clone <本仓库> && cd freetoken-glm5-boost
./install.sh --check ../FreeToken   # 先 dry-run
./install.sh ../FreeToken           # 应用 36 个补丁 + 17 个 overlay 新文件
cd ../FreeToken && pip install -e . # 或按上游 install.sh
```

启动参考 `examples/serve_full.sh`(路径按自己机器改;4 槽 256K 上下文 + 全部优化默认开)。

## 提速账本(逐项同 boot A/B 实测)

| 阶段 | 单流 tok/s | 手段 |
|---|---|---|
| 基线(bf16 全量 + CUDA graphs) | 17.8 | |
| + 非专家 FP8(attn/mlp,lm_head 保 bf16) | 23.6 | +28%,prefill 2.7× |
| + KDA FP8(in_proj/o_proj per-row) | 25.4 | +15%(fla in-kernel l2norm 洗掉尺度误差) |
| + 4 槽 256K 池(容量代价) | 21.2 | −18%(moe cache −26%,按需取舍) |
| + 投机专家预取 P=4(L+1 gate 提前打分) | 22.9 | +8% |
| + 常驻层重挑 3-10(5.3K 步踪迹全局 LRU 仿真) | 27.4 | +29%,全程最大单项 |
| + marlin decode 配置 16/32/2(27 配置微基准扫描) | 29.3 | +4.5% |
| + KDA gate 融合 + mHC 融合 | 29.6 | 每层 ~18 个散核并成 4 个 |
| + 路由融合(8 核链→1 核,topk 语义逐 id 一致) | **30.6** | +3.4% |

关键工程判断(细节见 `MANIFEST.md` 与补丁内注释):
- **本机按 PCIe 字节收费**:任何增加总搬运字节的投机都输——MTP 投机解码(接受率
  78%、KL 2.4e-3 全部做通)最终 −13% 归档;top-6 路由 +6% 因改 A18B 规格被否。
- **LRU 已近在线最优**:真实路由是漂移型时序局部性,LFU/共现/prefill 直方图/学习
  预测器全部实测判负;唯一活口是"把 L+1 层的真 gate 作用在 L 层 hidden 上"的
  结构预测(+8%)。
- **长尾融合三原则**:逐位/epsilon 对拍先行(200 批路由 id 100% 一致才上线)、
  免原子加保确定性、CUDA graph 内零 host 同步(懒构建全在 warm pass)。

## 耦合审计(2026-08-29)

- `engine.py` 顶层依赖 `gpu_select`(早期多卡基建),故基础设施补丁为硬依赖,单列一组;
- 融合 gate 输出 g/β/z 被 MTP verify 分支原样消费;融合路由输出契约
  (fp32 weights, int32 ids, contiguous)与 eager 逐字段一致,下游
  routed_forward / spec_prefetch / miss-cap 无感;
- marlin 配置常量单点定义,resident 与 offload 两路共用;
- `hc_norm` 只接入 glm5_next,deepseek_v4 路径零改动;
- 实验开关默认全安全:MTP=0、LFU=lru、miss-cap=-1、trace/profiler=关;
  融合三件默认开且各有一键回退 env。

## 目录

```
patches/    36 个 per-file unified diff(对 v0.1.2,git apply -p1)
overlay/    17 个新文件(models/glm5_next 整目录 + 4 个 triton 核 + 预取/LFU + gpu_select)
install.sh  校验版本 → dry-run → 应用 → compileall
examples/   生产启动脚本(4 槽 256K,全优化默认开)
MANIFEST.md 功能 → 文件 → 开关 → 实测数字 对照表
```

## 许可

与上游一致,Apache-2.0。本仓库只含我们编写的补丁与新文件,不含上游代码本体。
