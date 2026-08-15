# A002 基线复现与运行方差结果

**日期：** 2026-08-15
**状态：** `COMPLETE / BASELINE REPRODUCED / SAME-PROCESS STABLE`
**正式代码：** `8af44f7416c3abd11359e44643696f92fa4fe5c6`
**机器产物：** [`artifacts/A002/`](artifacts/A002/)

## 1. 执行完整性

- NIAH decision-dev：739 题，其中 Legacy Selector 改变 109 题；
- 同一进程共享 Granite 4.1-3B 与 TRUE；K/L 主轮次顺序按题轮换；
- K-first 363 题、L-first 376 题；changed 子集为 53/56；
- changed/unchanged 分层 10% 重复共 74 题；每臂 739 + 74 = 813 次调用；
- 两臂运行错误均为 0；1626 次调用的 A001 trace 均存在；
- `run` 阶段未加载 gold，生成完成后独立 `score` 才读取 reference answer；
- 正式运行没有失败重跑。两个失败 smoke 分别暴露 manifest cwd 和空分层报告问题，修复后第三个 smoke 完整 PASS；失败目录保留在服务器。

正式输入 identity SHA256 为 `bad751bd...3805`，与 F000/F001 一致；`generations.jsonl` SHA256 为 `d5ab417e...7fc8`。

## 2. 基线与 Selector 对照

| 范围 | K TopK Base | L Legacy Selector Base | L-K | 95% CI | 错→对 / 对→错 |
|---|---:|---:|---:|---:|---:|
| 全部 739 | 473/739 = 64.01% | 470/739 = 63.60% | -0.41pp | [-1.31,+0.44]pp | 4 / 7 |
| changed 109 | 64/109 = 58.72% | 61/109 = 55.96% | -2.75pp | [-8.62,+2.97]pp | 4 / 7 |
| unchanged 630 | 409/630 = 64.92% | 409/630 = 64.92% | 0.00pp | [0.00,0.00]pp | 0 / 0 |

Coverage：

| 范围 | K | L | L-K | 95% CI |
|---|---:|---:|---:|---:|
| 全部 739 | 650/739 = 87.96% | 646/739 = 87.42% | -0.54pp | [-1.25,0.00]pp |
| changed 109 | 96/109 = 88.07% | 92/109 = 84.40% | -3.67pp | [-8.11,0.00]pp |
| unchanged 630 | 554/630 = 87.94% | 554/630 = 87.94% | 0.00pp | [0.00,0.00]pp |

这两臂与 F001 的 `C_topk_verify`、`D_selector_verify` 在 739/739 题上答案、citation IDs 和完整 `GenerationResult` 逐题一致。A002 因而复现了当前 Base Verify 基线；Legacy Selector 在相同 Generator 下仍没有答案增益，CI 也不支持正式的负效应声明。

## 3. 同输入稳定性

| 检查 | K | L |
|---|---:|---:|
| 74 题重复 answer 完全一致 | 74/74 | 74/74 |
| 74 题重复 GenerationResult 完全一致 | 74/74 | 74/74 |
| answer_match 一致 | 74/74 | 74/74 |
| coverage 一致 | 74/74 | 74/74 |

630 个 Selector unchanged 题的 K/L context 完全相同；两臂虽按轮换顺序分别生成，answer 与完整 `GenerationResult` 仍为 630/630 一致。因此在本服务器、同一进程和当前执行速度下没有观察到足以改变 utility 标签方向的运行波动。历史跨 job/不同速度的差异仍是外部边界，A002 不把本次结果外推为跨作业 bitwise determinism。

## 4. Trace 归因

主轮次 final-empty reason：

| 原因 | K | L |
|---|---:|---:|
| 非空答案 | 650 | 646 |
| 全部 claims 不 faithful | 52 | 54 |
| draft 非空但 splitter 得到零 claims | 21 | 25 |
| Granite draft 为空/拒答 | 15 | 13 |
| claims 全部被移除或组装为空 | 1 | 1 |

Splitter 状态：K 为 structured 715、degraded fallback 9、empty-draft 未运行 15；L 为 718、8、13。该表只定位 A002 基线的空输出发生在哪一层，不是 B100 context matrix 的因果结论。

## 5. A002 判定

- 基线复现：PASS；
- trace 完整性：PASS；
- 同进程重复稳定性：PASS；
- Legacy Selector + 同一 Base Generator 的答案提升：未观察到；
- A002 不训练模型、不使用 sealed600、不读取 system held-out，也不构成下一方法的效果验证。
