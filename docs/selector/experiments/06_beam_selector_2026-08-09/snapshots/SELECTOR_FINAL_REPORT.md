# Selector 最终结论

**日期：** 2026-08-10

**最终决定：** 三分类 Beam Selector 未通过 M2 停止门槛；取消 M3/M4，系统使用 TopK Selector。

**模块边界：** Hybrid RRF 仍提供冻结的 Top-20；Generator 未参与本次实验，也没有被修改。

## 1. 一句话结论

三分类 Beam Selector 能明显减少误导证据，但会同时删除过多回答所必需的正确证据，因此不适合本项目“降低干扰且保留必要信息”的 Selector 目标。按照预先写明的规则，实验在单 seed 可行性阶段停止，不再追加训练或正式测试。

## 2. 实验进行到哪里

| 阶段 | 结果 | 含义 |
|---|---|---|
| M0 数据与代码冻结 | PASS | 候选池、划分和泄漏检查合格 |
| M1 32 题最小正确性 | PASS | 代码、标签和训练链路能够运行 |
| M2 seed 13 完整开发集评估 | **FAIL** | 16 组阈值均超过必要证据损失上限 |
| M3 三 seed 稳定性 | 取消 | M2 未通过，不应继续消耗算力 |
| M4 sealed600/2Wiki 正式测试 | 取消 | 失败方法不得查看正式测试集后再调参 |
| M5 最终处理 | TopK cutover | 保留负结果，删除失败方法的运行入口 |

M1 的高训练准确率只证明模型学会了训练标签，不能证明真实 Selector 有效。是否采用方法由 M2 与同候选池 TopK 的真实开发集比较决定。

## 3. M2 核心结果

下表展示 16 组中“必要证据损失最小”的配置（`required_threshold=0.4`、`reject_threshold=0.9`）。它仍不合格，只用于说明失败原因。

| 指标 | TopK | 三分类 Beam | Beam − TopK | 判断 |
|---|---:|---:|---:|---|
| NIAH harmful-in-context (%) ↓ | 88.13 | 11.64 | **−76.49 pp** | 过滤误导证据有效；95% CI 为 [−78.91, −74.07] pp |
| NIAH required-evidence recall (%) ↑ | 81.51 | 64.21 | **−17.30 pp** | 远超允许的 5 pp 损失，失败 |
| NIAH evidence precision (%) ↑ | 42.14 | 69.59 | +27.44 pp | 提高来自更激进的删除 |
| NIAH selected evidence (#) | 10.00 | 4.85 | −5.15 | 平均删掉一半以上候选 |
| 2Wiki supporting recall (%) ↑ | 74.49 | 75.03 | +0.54 pp | 没有下降，但不能抵消 NIAH 失败 |
| 2Wiki evidence precision (%) ↑ | 19.04 | 93.89 | +74.85 pp | 2Wiki 只标 supporting documents，不能单独证明抗误导能力 |

全部 16 组阈值的共同趋势：

- harmful-in-context 下降 76.11–81.48 pp；
- NIAH required recall 同时下降 17.30–20.08 pp；
- 合格配置数为 0，正式裁决为 `FAIL`。

因此，这不是“完全不会过滤”，而是“过滤过强、可靠性与完整性的取舍不合格”。继续运行另外两个 seed 只能重复验证一个已经远超失败边界的结果，不能把 17.30 pp 的损失变成允许的 5 pp。

## 4. 旧方法的最终记录

| 指标 | TopK | Reliability-MIS |
|---|---:|---:|
| Harmful-in-context (%) ↓ | 91.77 | 31.35 |
| Required-evidence recall (%) ↑ | 90.97 | 53.80 |
| Evidence precision (%) ↑ | 33.38 | 23.72 |
| 2Wiki supporting recall (%) ↑ | 74.49 | 63.30 |

MIS 能发现冲突，却不能可靠判断冲突双方哪一方正确，因此也会严重误删必要证据。更早的 corroboration/Graph 路线存在相同方向的问题，并且实现复杂度更高。两条旧路线均不进入系统。

## 5. 最终系统中的 Selector

最终保留 `TopKSelector`：它不是训练模型，而是按 Hybrid RRF 的冻结顺序，把 Top-20 中排名最高的 10 条原始证据交给后续模块。选择 TopK 不是证明它能过滤错误证据，而是在两个改造方案均未达到“减少误导且不显著丢失必要证据”后，选择当前证据支持下最稳妥、接口最稳定的回退方案。

这意味着：

- 三模块系统可以继续集成；
- Retriever 与 Generator 的既有接口不变；
- 论文或报告不得声称 Selector 已经带来性能提升；
- Selector 部分应如实报告负结果和最终 TopK 回退。

## 6. 可复核材料

- `results/selector-beam-v1/m0/M0_REPORT.json`：数据冻结与泄漏审计；
- `results/selector-beam-v1/m1/M1_REPORT.json`：最小正确性结果；
- `results/selector-beam-v1/m2/M2_TRAIN_REPORT.json`：seed 13 训练记录；
- `results/selector-beam-v1/m2/M2_EVALUATION_REPORT.json`：16 组阈值的完整配对评估；
- `results/selector-beam-v1/m2/checksums.sha256`：M2 两份报告的完整性校验。

大型模型权重和逐题开发输出保留在教师服务器 scratch，未提交 Git。Git 只保存足以复核决定的小型报告。

## 7. M5 完成验证

- M2 结果归档提交：`64cab11`；
- TopK cutover 与旧运行代码清理提交：`0c710e2`；
- Selector 生产目录只保留 `top_k.py`、数据构造仍需的 `answer_norm.py` 和包初始化文件；
- 全部实验配置的 Selector 均为 `top-k`；
- 本地完整测试、Ruff、Mypy 全部通过；
- 教师服务器已同步到 `0c710e2`，服务器完整测试通过；
- `docs/selector/` 只保留本报告，旧计划与重复总结由 Git 历史保存；
- 用户自己的 `docs/presentation/` 未纳入提交，也未被修改。
