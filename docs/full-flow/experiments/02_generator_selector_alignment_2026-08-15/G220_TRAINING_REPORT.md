# G220 GC/GM 正式训练报告

**日期：** 2026-08-16
**状态：** `COMPLETE / TRAINING PASS`
**服务器产物：** `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G220-v1/formal`
**训练代码：** commit `a22cfbd8704f678ec33683495a9d11a0ad9f38ec`；`full_flow_g220_train.py` SHA256 `8126de1ccaf685dc07350d3421694d4188da6c4e06155cfc6ad48610351d622c`

## 1. 完成边界

GC 和 GM 均按 G220-S 冻结配置完成 seeds 13/42/73。每个 run 都满足：

- 515 个 train queries、4,120 个 examples、1 epoch、515 optimizer updates；
- `max_length=2304`，observed max 为 GC 1,605、GM 2,120，截断数为 0；
- LoRA `r=8 / alpha=16 / dropout=0.05 / lr=1e-4`；
- adapter 只在 draft generation call 启用；key-fact extraction 未使用；
- claim splitter 使用 adapters disabled 的 frozen Granite base；TRUE 未参与训练；
- decision-dev、sealed600 和 system held-out 均未读取；
- 保存后的每个 adapter 均通过 fresh Granite base reload，reload 后 trainable parameters 为 0。

六个 run 的模型、G200 manifest、train cases 和 model-val cases hash 完全一致：

| 输入 | SHA256 |
|---|---|
| Granite config | `9a0e589b69e7d3ad9fb9fb2c844aa7d7156e052cb7ea4211de7de48ab7c8525c` |
| G200 data manifest | `acc376e31196952b0d195e397784dd6bc4ecc0887ecdb25c8157bb3ed698d25a` |
| Train cases | `c045fc40495d4852fdb930663257970e000f4b597fd51522c8474f4b732ff5ed` |
| Model-val cases | `26db5bd771c37849a1018e154c69baccb66057afcafaa876114cacc5425eb9a2` |

## 2. 正式运行核验

| Run | 时间 | Peak CUDA | Mean train loss | Final-100 loss | Reload | Adapter weights SHA256 |
|---|---:|---:|---:|---:|---|---|
| GC seed 13 | 50.8 min | 9.36 GB | 0.05174 | 0.00008 | PASS | `4edbf846...d4929` |
| GC seed 42 | 50.8 min | 9.36 GB | 0.04811 | 0.00003 | PASS | `51a6afe9...6db9` |
| GC seed 73 | 50.8 min | 9.36 GB | 0.04716 | 0.00016 | PASS | `a34680ff...c8a3` |
| GM seed 13 | 102.6 min | 10.09 GB | 0.07111 | 0.00750 | PASS | `8b9fd440...6d2c` |
| GM seed 42 | 102.6 min | 10.09 GB | 0.06980 | 0.01042 | PASS | `6da8e49a...d882` |
| GM seed 73 | 104.6 min | 10.09 GB | 0.06807 | 0.01176 | PASS | `662941e8...2759` |

六份 adapter weights hash 均不同，且均与各自 manifest 记录一致。归档的 `adapter_config.json` 字节 hash 因 PEFT 对 `target_modules` 集合的输出顺序不同而不同；字段集合与所有训练参数语义一致。

## 3. Model-val loss

| Run | O | TopK | Selected | O+B | O+H | O-first | O-middle | O-last |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GC 13 | 0.356 | 0.702 | 0.668 | 0.453 | 0.651 | 0.374 | 1.282 | 1.195 |
| GC 42 | 0.306 | 0.685 | 0.658 | 0.451 | 0.602 | 0.324 | 1.395 | 1.370 |
| GC 73 | 0.260 | 0.582 | 0.556 | 0.377 | 0.506 | 0.290 | 1.121 | 1.060 |
| GM 13 | 0.371 | 0.373 | 0.376 | 0.393 | 0.338 | 0.361 | 0.338 | 0.309 |
| GM 42 | 0.322 | 0.300 | 0.300 | 0.325 | 0.287 | 0.308 | 0.247 | 0.241 |
| GM 73 | 0.343 | 0.347 | 0.348 | 0.357 | 0.333 | 0.337 | 0.324 | 0.300 |

GM 的 loss 在 mixed contexts 间更均匀，GC 在 O-middle/O-last 上更高。这只说明训练目标上的拟合形态不同，不能解释为 answer-match、coverage、citation 或上下文鲁棒性已经提升。方法是否保留必须由冻结的 G230 完整 739 题和 stress diagnostics 决定。

## 4. 执行事件

GC seed 42 和 GM seed 13 的早期交互式 SSH 尝试曾因控制连接中断而停止；当时没有写出 output directory 或 manifest，随后从冻结输入和 seed 在新进程完整重跑。报告只包含六个从 0 到 4,120 examples 完成并通过 persisted-adapter reload 的正式 run。

最后两个 GM run 执行期间，本地一度因 VPN 路由未进入隧道而收到 SSH `Connection refused`。服务器没有重启，SSH 服务也未重启；两个 `nohup` 训练进程持续运行并正常完成。这一连接事件没有改变训练进程、输入或产物。

## 5. 归档

Git 仅归档每个 run 的 `training_manifest.json` 和 `adapter_config.json`：

- `artifacts/G220/formal/gc-seed{13,42,73}/`；
- `artifacts/G220/formal/gm-seed{13,42,73}/`。

模型权重保留在服务器 runtime，不提交 Git。G220 的结论是 `TRAINING PASS`，不是 Generator 方法通过开发门。
