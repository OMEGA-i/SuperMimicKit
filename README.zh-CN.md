# SuperMimicKit

**面向 Unitree G1、Unitree H2 和 Agibot A3 的 SuperMimic 非官方复现。**

[English](README.md) · 简体中文

基于 PyTorch + MuJoCo Warp 训练，使用 UMR 重定向，通过 ONNX Runtime Web / WebGPU
运行浏览器 demo。独立实现，参考 [SuperMimic](https://supermimic.github.io/)
与 [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)。

## 项目进展

- [x] G1 / H2 / A3 配置与统一机器人资产目录。
- [x] LaFAN 重定向，以及明确的 train40 / 独立 val37 划分。
- [x] FP32 训练，可选 TF32 / BF16，支持编译与 CUDA graphs。
- [x] 各机器人独立记录 SwanLab 训练指标与完整动作评测。
- [x] Policy / world model ONNX 导出与 WebGPU demo 集成；MuJoCo 物理使用 WASM。
- [x] 每 10k iterations 保存配套 policy + world model 权重，包含归一化状态。
- [ ] 发布 G1 / H2 / A3 完整 **LaFAN 100k** demo checkpoint。
- [ ] 发布对应 checkpoint 的 train40 / val37 指标与 rollout 视频。
- [ ] **BONES-SEEDs checkpoint**：正在准备重定向流程，后续训练并发布。
- [ ] **GRAIL checkpoint**：正在处理动作与场景，后续开展包含场景的训练并发布。

## 性能报告

以下为开发阶段实测，不代表已达到论文的收敛效果。

| 项目 | 结果 | 条件 |
| --- | --- | --- |
| G1 训练吞吐 | **2.18 iterations/s** · 0.459 s/iteration | H100，4,096 envs，TF32，compile + CUDA graphs |
| GPU 利用率 | **84.7%** | 同一短跑测试，33 次 GPU 采样 |
| G1 policy ONNX 最大绝对误差 | **2.63 × 10⁻⁶** | 精确 50k policy，FP32 PyTorch 对比 ONNX Runtime CPU |
| A3 policy ONNX 最大绝对误差 | **4.06 × 10⁻⁶** | 精确 50k policy，FP32 PyTorch 对比 ONNX Runtime CPU |

吞吐为 36 个测量 iteration 的均值，不含初始编译时间；不等于训练全程耗时或浏览器速度。
ONNX 每个机器人使用 16 组人工生成的输入验证，只说明这些输入下的数值一致性，
不代表运动质量或完整 world model 的一致性。详见[测量说明](docs/performance.md)。

**100k 动作跟踪指标待发布。** 每个机器人分别报告 train40 / val37 的完整动作归一化回报
与失败动作数。归一化回报为 `sum(reward) / (0.1 × (frames − 1))`，各动作等权平均；
提前终止后的剩余帧按零回报计算，触发提前终止即计为失败动作。
训练期间的单步平均回报是另一项指标。当前 40 段名单尚未确认与论文作者完全一致，
详见[评测范围](docs/reproduction_scope.md)。

## 快速开始

```bash
git clone --recurse-submodules https://github.com/OMEGA-i/SuperMimicKit.git
cd SuperMimicKit
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[mjwarp,qc,test,tracking,web-refresh]'
bash scripts/bootstrap_umr.sh
```

训练前需准备具有相应使用许可的机器人资产及重定向动作。
参考[安装、数据与训练指南](docs/getting_started.md)、
[UMR 集成说明](third_party/umr_overlay/README.md)和[浏览器 demo 指南](web/demo/README.md)。

## Checkpoint 与后续数据集

下一次公开发布以 G1、H2、A3 的 **100k iterations** 为目标。完整 demo 包需要来自同一
checkpoint 的 policy 和 world model ONNX，以及匹配的元数据与机器人资产。
目前尚未发布完整预训练 demo 包，后续见 [GitHub Releases](https://github.com/OMEGA-i/SuperMimicKit/releases)。

| 数据集 | 处理流程 | 发布计划 |
| --- | --- | --- |
| LaFAN | 原始数据 → SMPL-X → UMR → 50 Hz motions | G1 / H2 / A3 的 100k checkpoint |
| BONES-SEEDs | 120 Hz BVH → 内存中 SOMA 解析 / 50 Hz 重采样 → UMR → motions | G1 / H2 / A3 checkpoint，时间待定 |
| GRAIL | 原始数据 → SMPL-X + scenes → UMR → 配套 motions 与 scenes | G1 / H2 / A3 checkpoint，时间待定 |

BONES-SEEDs 不额外落盘一份 SOMA 数据。GRAIL 场景始终与重定向动作配套。
数据集和模型的再分发遵循各自许可。

## 致谢

感谢 [SuperMimic 作者](https://supermimic.github.io/)、
[supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)、
[UMR](https://github.com/hanyang9/UMR) 及机器人与数据集贡献者。
本项目使用的 UMR fork 为 [OMEGA-i/UMR](https://github.com/OMEGA-i/UMR)。
第三方代码、模型和数据集保留各自许可证及声明。
