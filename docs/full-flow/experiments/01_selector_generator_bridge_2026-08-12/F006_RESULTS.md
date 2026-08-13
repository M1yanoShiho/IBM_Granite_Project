# F006 小规模鲁棒训练正式开发集结果

**日期：** 2026-08-13
**状态：** `COMPLETE / DEVELOPMENT GATE PASS / F005 AUTHORISED`

## 一句话结论

Mixed-LoRA 在 109 个 Selector 实际改变证据的问题上取得 **61.47%** 的最终答案正确率，高于原始 TopK、未训练的关键事实笔记和 Clean-LoRA，因此通过了预先固定的 F005 进入门槛。

这表示“在正确证据仍存在时，专门学习如何从混有无关内容的上下文中提取关键事实”是目前最有希望的方向。它还不是最终 superiority 结论，因为 109 题上的答案差值置信区间仍跨过 0；必须在此前没有参与方法选择的独立数据上确认。

## 最终答案结果

| 路线 | Answer match | Coverage | 有关键事实笔记的题数 |
|---|---:|---:|---:|
| TopK_frozen | 58.72% | **88.07%** | 0 |
| Base_notes_frozen | 57.80% | 84.40% | 37 |
| Clean_LoRA | 58.72% | 85.32% | 99 |
| **Mixed_LoRA** | **61.47%** | 85.32% | 100 |

这里的 `Answer match` 表示最终答案是否包含参考答案要求的内容；`Coverage` 表示系统有没有给出非空答案。Mixed 的正确率最高，但作答覆盖率仍比 TopK 低 2.75 个百分点，这是独立确认中必须继续观察的限制。

## 配对比较

每一条差值都在同一道问题上比较，因此不会把题目难度差异混入结果。

| 比较 | Answer 差值 | 95% CI | 错→对 | 对→错 | 净变化 |
|---|---:|---:|---:|---:|---:|
| Clean − Base | +0.92 pp | [−6.96, +9.26] | 9 | 8 | +1 |
| **Mixed − Base** | **+3.67 pp** | [−3.67, +11.54] | 10 | 6 | +4 |
| **Mixed − Clean** | **+2.75 pp** | [−2.00, +8.18] | 5 | 2 | +3 |
| **Mixed − TopK** | **+2.75 pp** | [−6.03, +12.17] | 13 | 10 | +3 |

零基础地解释：点估计说明 Mixed 在这批题上确实多答对了一些；但 95% 置信区间都包含 0，说明样本波动仍可能把这几道题的净提升抵消。因此本结果足够选出下一步唯一候选，却不足以单独宣布稳定提升。

## 独立引用评分

引用评分由 MiniCheck 完成；生产系统内部使用的 TRUE 不参与裁判。

| 路线 | Answered | Citation precision（ALCE） | Precision（实际有引用） | Citation recall（ALCE） |
|---|---:|---:|---:|---:|
| Base_notes_frozen | 92 | 80.43% | 92.50% | 80.43% |
| Clean_LoRA | 93 | 79.03% | **94.23%** | 79.03% |
| TopK_frozen | 96 | 82.81% | 91.38% | 82.29% |
| **Mixed_LoRA** | 93 | **84.95%** | 92.94% | **84.95%** |

在双方都作答的题上，Mixed 与 TopK 的配对引用 precision/recall 差值为 0；Mixed 相对 Clean 为 +5.98 pp，95% CI 为 `[+0.60, +11.80]`。因此没有证据表明 Mixed 用更差的引用换取答案提升。

## 为什么 Mixed 优于 Clean 很重要

Clean 与 Mixed 使用相同问题、相同目标、相同训练样本数和相同更新步数。Clean 只练习从干净证据提取事实；Mixed 还练习“正确证据仍在，但周围有无关证据”的情况。

如果两组相同，收益可能只是普通微调；现在 Mixed 在训练内部的混合上下文损失、最终答案正确率和独立引用指标上都高于 Clean。这支持了更具体的解释：**有针对性的混合上下文训练比单纯 clean 微调更适合 Selector 后的 Generator。**

## 对 Selector 作用的正确解读

本轮只评估了 109 个 Selector 改变证据的开发问题，并且 Mixed 使用的是 selected evidence。它证明 Mixed-LoRA 能更好地接住 Selector 的输出，但还没有在独立数据上把 Generator 收益与 Selector 收益拆开。

F005 必须至少比较：

1. `TopK + 原有 Generator`：默认系统；
2. `TopK + Mixed-LoRA`：只改 Generator；
3. `Selector + Mixed-LoRA`：完整三模块系统。

其中第 3 组减第 2 组才代表 Selector 在同一个新 Generator 下的净作用；第 3 组减第 1 组才代表完整系统相对默认系统的总提升。

## 决定

- F006 开发门：**PASS**；
- 唯一进入 F005 的候选：**Mixed-LoRA**；
- Clean-LoRA 不进入最终确认；
- 不重新选择训练参数、阈值或案例；
- F005 只使用此前未参与方法选择且通过来源隔离检查的数据。

## 机器产物

- 答案报告：[`artifacts/F006/eval/REPORT.md`](artifacts/F006/eval/REPORT.md)
- 答案 JSON：[`artifacts/F006/eval/report.json`](artifacts/F006/eval/report.json)
- 引用报告：[`artifacts/F006/eval/CITATION_REPORT.md`](artifacts/F006/eval/CITATION_REPORT.md)
- 引用 JSON：[`artifacts/F006/eval/citation_report.json`](artifacts/F006/eval/citation_report.json)
- 109 题逐题输出：[`artifacts/F006/eval/generations.jsonl`](artifacts/F006/eval/generations.jsonl)
- 运行清单：[`artifacts/F006/eval/run_manifest.json`](artifacts/F006/eval/run_manifest.json)
