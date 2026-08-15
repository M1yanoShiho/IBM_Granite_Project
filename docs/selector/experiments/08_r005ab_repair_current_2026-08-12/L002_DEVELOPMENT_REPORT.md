# L002 训练与开发选择报告

**日期：** 2026-08-12
**状态：** `PASS / POLICY FROZEN / FINAL UNOPENED`

## 零基础结论

这一阶段回答的是：“我们能不能找到一种足够保守的删除方式，确实减少错误证据，同时几乎不伤害正确证据？”

答案是：在普通开发集上可以。最终选择为 `NLI-base + q=0.99 + cap=2`。

这里的 `cap=2` 只是“最多可以删两条”，不是“每题必须删两条”。实际平均每题只删约 `0.063` 条证据，也就是大约每 100 个问题总共删 6 条，绝大多数问题保持原来的 TopK10。

## 训练是否正常

两个随机起点使用完全相同的训练方法，各完成 3 轮、1992 次参数更新：

| Seed | 第1轮总损失 | 第2轮 | 第3轮 | Protect/Harm head | 状态 |
|---:|---:|---:|---:|---|---|
| 13 | 0.529 | 0.336 | 0.240 | 都实际更新 | PASS |
| 42 | 0.538 | 0.399 | 0.274 | 都实际更新 | PASS |

这表示模型确实在学习，不是只完成了程序运行。

## 为什么选 q=0.99、cap=2

| 阈值与上限 | 平均每题删除 | Harmful 减少 seed13 / seed42 | NIAH recall 损失 seed13 / seed42 | NIAH chain 损失 seed13 / seed42 | 开发门 |
|---|---:|---:|---:|---:|---|
| q=.995, cap1 | 0.027 | 5.92% / 3.64% | 0 / 0 pp | 0 / 0 pp | PASS |
| q=.995, cap2 | 0.029 | 6.07% / 3.79% | 0 / 0 pp | 0 / 0 pp | PASS |
| q=.99, cap1 | 0.055 | 10.32% / 9.86% | 0.068 / 0 pp | 0.222 / 0 pp | PASS |
| **q=.99, cap2** | **0.063** | **10.77% / 10.47%** | **0.068 / 0 pp** | **0.222 / 0 pp** | **PASS / SELECTED** |
| q=.975, cap1 | 0.173 | 28.98% / 29.89% | 0.532 / 0.422 pp | 1.996 / 1.109 pp | FAIL：seed13 chain 过界 |
| q=.975, cap2 | 0.206 | 31.11% / 32.32% | 1.189 / 0.552 pp | 3.991 / 1.996 pp | FAIL：保护过界 |
| q=.95, cap1 | 0.330 | 42.49% / 40.82% | 1.977 / 1.760 pp | 6.430 / 4.878 pp | FAIL：保护过界 |
| q=.95, cap2 | 0.520 | 48.71% / 48.86% | 5.641 / 4.518 pp | 15.965 / 13.747 pp | FAIL：保护过界 |

两档 q=.99 都满足保护要求。cap2 的两 seed 平均 harmful reduction 比 cap1 高约 `0.531` 个百分点，刚刚超过预先规定的 `0.5` 个百分点等价范围，因此按事先规则选择 cap2，而不是结果出来后凭感觉选择。

两个 seed 在 2Wiki 上的 recall 和 chain 损失均为 `0`。seed13 的 harmful reduction 与 deletion precision 也都优于“随机删同样数量”和“从排名末尾删同样数量”的对照。

## 已冻结内容

- variant：`NLI-base`；基础版通过，因此不运行 `NLI-pair`。
- quantile / cap：`0.99 / 2`。
- checkpoint：seed13 `86622b...72bf`；seed42 `c258a9...d421`。
- 完整 final evaluator 代码：commit `4fb68faaf9f2cf263bb4c45465cf5d3a4bf3ab63`。
- development projection、final 原始输入和 Granite 4.1 3B 配置均已用 hash 固定。

## 下一步

L003 只允许做一次：在此前未查看 Selector 效果的 `decision-dev` 上，先运行固定证据门；证据门通过后才运行固定 Granite，比较同题 TopK10 与 Selector 的答案正确率。L003 中不能再调整模型、阈值或 cap。

## 机器产物

- [`artifacts/L002_DEVELOPMENT_RESULTS.json`](artifacts/L002_DEVELOPMENT_RESULTS.json)
- [`artifacts/L002_FROZEN_POLICY.json`](artifacts/L002_FROZEN_POLICY.json)
- [`artifacts/L002_SEED13_TRAINING_SUMMARY.json`](artifacts/L002_SEED13_TRAINING_SUMMARY.json)
- [`artifacts/L002_SEED42_TRAINING_SUMMARY.json`](artifacts/L002_SEED42_TRAINING_SUMMARY.json)
