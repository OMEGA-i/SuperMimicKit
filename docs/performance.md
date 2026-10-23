# Performance notes / 性能测量说明

[English](../README.md) · [中文](../README.zh-CN.md)

## Training throughput / 训练吞吐

Measured on 2026-10-23 during G1 development on NVIDIA H100, MuJoCo Warp,
LaFAN train40 and 4,096 environments. TF32, compilation and CUDA graphs enabled.
The 40-iteration short run contributes 36 measured iterations after startup.
Mean stage times: collection 0.211 s, world-model update 0.086 s,
policy update 0.163 s; total 0.459 s (2.18 iterations/s).
GPU utilization averages 84.7% across 33 samples.

This historical development measurement excludes initial compilation and is
not a fresh benchmark of the current public commit. Checkpoint I/O, evaluation,
concurrent workloads and long-run behavior can change throughput.

该结果来自开发阶段的 G1 短跑，不是当前公开提交的重新跑分。不含初始编译，
保存 checkpoint、评测及其他并发任务会影响实际训练吞吐。

## ONNX policy parity / ONNX policy 数值一致性

Exact 50k G1 and A3 policy weights were exported in FP32 and checked using
ONNX Runtime CPU against PyTorch over 16 synthetic fixtures each.
Reported errors are rounded upward. These artifacts were policy-only;
this check does not validate a full demo, world model, WebGPU execution,
or motion-tracking quality. The public complete release target is 100k.

精确 50k policy 的数值验证不代表完整 demo、world model、WebGPU 或动作跟踪评测。
完整公开版本仍以 100k 为目标。

The sanitized [measurement record](performance.json) contains settings and
numerical results, without deployment paths or private training artifacts.
100k train40 / val37 normalized returns, failure counts and videos remain TODO.
