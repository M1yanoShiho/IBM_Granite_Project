# F005 独立最终确认结果

**日期：** 2026-08-13
**状态：** `COMPLETE / FINAL NO-WIN / SEALED SET RETIRED`

## 一句话结论

在 600 个此前没有参与方法选择的问题上，当前完整方案没有超过默认 TopK：两者最终答案正确率同为 **63.17%**。固定使用同一个 Mixed-LoRA Generator 后，加入 Selector 的正确率从 **63.33%** 变为 **63.17%**，即少答对 1 题。

因此，预先约定的最终成功条件没有通过，不能声称当前 Selector + Generator 方案提高了端到端答案质量。

这不等于前面的实验都没有用。Selector 删除的 82 条证据中，74 条确实是已知有害证据，删除准确率为 **90.24%**。独立实验否定的是“删对坏证据后，当前 Generator 会自然变得更好”这一假设，而不是 Selector 识别坏证据的能力。

## 三组到底比较什么

| 组别 | 实际系统 | 要回答的问题 |
|---|---|---|
| A：TopK + Base | 默认 TopK10 + 原有 Verify-and-annotate | 不做本轮改造时有多好？ |
| B：TopK + Mixed | 默认 TopK10 + Mixed-LoRA 关键事实读取 | 只改 Generator 是否有用？ |
| C：Selector + Mixed | Selector 清理证据 + 同一个 Mixed-LoRA Generator | Selector 在相同 Generator 下是否带来额外作用？ |

最重要的两条比较是：

- `C − B`：只隔离 Selector 的净作用；
- `C − A`：完整新系统相对默认系统的总作用。

## 最终答案结果

| 组别 | 答对 | Answer match | 作答数 | Coverage |
|---|---:|---:|---:|---:|
| A：TopK + Base | 379 / 600 | 63.17% | 540 / 600 | **90.00%** |
| B：TopK + Mixed | **380 / 600** | **63.33%** | 528 / 600 | 88.00% |
| C：Selector + Mixed | 379 / 600 | 63.17% | 525 / 600 | 87.50% |

Mixed-LoRA 单独相对默认系统只多答对 1 题；加入 Selector 后，这 1 题净收益又消失了。完整系统与默认系统最终完全打平，同时少回答了 15 题。

## 配对比较与置信区间

| 比较 | Answer 差值 | 95% CI | 错→对 | 对→错 | 净变化 |
|---|---:|---:|---:|---:|---:|
| B − A：只改 Generator | +0.17 pp | [−2.68, +3.15] | 42 | 41 | +1 |
| C − B：只加 Selector | −0.17 pp | [−0.84, +0.50] | 2 | 3 | −1 |
| C − A：完整系统 | +0.00 pp | [−2.84, +2.86] | 40 | 40 | 0 |

零基础解释：`+0.17 pp` 不是提升了 17%，而是 600 题中只多答对 1 题。置信区间都跨过 0，表示这些很小的差异无法排除为样本波动。更重要的是，预先约定的点估计条件 `C > A` 且 `C > B` 本身也没有满足，所以不需要依赖统计解释来判定：本轮最终确认就是未通过。

在 Selector 实际改变证据的 81 题中：

| 比较 | Answer 差值 | 95% CI | 错→对 | 对→错 |
|---|---:|---:|---:|---:|
| C − B：只加 Selector | −1.23 pp | [−6.17, +3.70] | 2 | 3 |
| C − A：完整系统 | +0.00 pp | [−8.64, +8.64] | 6 | 6 |

## Selector 到底有没有删对

| 证据层检查 | 结果 | 零基础含义 |
|---|---:|---|
| Selector 改变的问题 | 81 / 600 | 它只在认为足够安全时动作，不是每题强制删除 |
| 删除证据总数 | 82 | 80 题删 1 条，1 题删 2 条 |
| 删除已知有害证据 | 74 | 大部分删除命中了实验植入的错误证据 |
| 删除准确率 | 90.24% | 每 10 条删除里约 9 条确实有害 |
| 有害证据减少率 | 14.12% | TopK 中 524 个可见有害证据里删掉 74 个 |
| 被删除的 gold 相关文档 | 3 条，涉及 2 题 | 存在少量误删，但范围很小 |

`90.24%` 和 `14.12%` 衡量的不是同一件事：前者问“已经删除的内容有多准”，后者问“所有可见坏证据一共清掉多少”。当前策略非常保守，所以删除很准，但只覆盖一小部分坏证据。

## 为什么删对了，答案仍没有变好

`C − B` 只有 5 题发生对错转换。逐题查看后，这 5 题删掉的都是已知有害证据，并未删除 gold 文档：

| Query | 删除后的变化 | 归因 |
|---|---|---|
| 3946 | Jacob Tremblay → Jaeden Lieberher，错转对 | 删除错误演员名后成功恢复正确事实 |
| 632 | 只输出歌名 → Chaka Khan，错转对 | 删除错误歌手名后成功恢复正确事实 |
| 3891 | April 2005 → 空答案，对转错 | 正确事实和关键事实笔记仍在，但上下文变化触发空输出 |
| 4297 | Adam and Eve 的儿子 → 空答案，对转错 | 正确证据和笔记仍在，但生成路径未稳定保留答案 |
| 484 | September 2012 → January 2012，对转错 | 删除错误证据后，Generator 仍在剩余文本的多个日期间选择不稳定 |

所以，本轮净失败不能主要归因于“删除上限是 2”或“Selector 删得太激进”。在真正改变最终对错的 5 题中，删除动作本身方向正确；问题出在 Generator 对轻微上下文变化不够稳定：有时受益，有时空答，有时改选另一个细节。

## 引用结果

引用由独立的 MiniCheck 评分，生产系统内部 TRUE 不参与裁判。

| 组别 | 已作答 | Citation precision（ALCE） | Precision（实际有引用） | Citation recall（ALCE） |
|---|---:|---:|---:|---:|
| A：TopK + Base | 540 | **77.25%** | **89.91%** | **76.06%** |
| B：TopK + Mixed | 528 | 76.36% | 88.61% | 75.41% |
| C：Selector + Mixed | 525 | 75.17% | 88.49% | 74.38% |

完整系统的引用指标也没有超过默认系统。由于 C 的作答数更少，ALCE 指标会同时受到空答案影响；但即使只在双方都作答的题上配对比较，C 相对 A、C 相对 B 的点估计仍为负，置信区间跨 0。因此没有证据支持“答案持平但引用显著变好”这一替代结论。

## 前面哪些结果保留，哪些主张停止

### 可以保留

- Selector 的保守删除能力：独立集 deletion precision 为 90.24%；
- 运行时不读取 gold、开发/最终数据零 query-ID 重叠、同进程配对比较等实验边界；
- F001/F002 暴露的 Generator 上下文敏感问题；
- F006 Mixed-context 训练在开发集上的正向结果，作为“候选如何被选出”的真实开发记录。

### 不能继续当成已成立结论

- Mixed-LoRA 能稳定提高最终答案；
- Selector 能在当前 Mixed-LoRA Generator 下带来净提升；
- 当前完整三模块系统优于默认 TopK；
- 简单地继续放宽或收紧删除数量就足以解决端到端问题。

F006 在 109 题开发集上相对 TopK 为 `+2.75 pp`，但在独立 600 题上，Generator-only 只剩 `+0.17 pp`，完整系统为 `0.00 pp`。这是正常且有价值的独立验证结果：开发集帮助选候选，最终集负责阻止我们把不稳定的候选误写成可靠提升。

## 停止决定与下一步边界

本实验路线到 F005 正式结束，sealed600 从此退休，不能继续用于调 Selector 阈值、LoRA 参数或挑选新方法。

如果以后新开一条实验路线，问题应从“如何再调删除规则”改为：

> 当正确证据仍然存在时，如何让 Generator 在 TopK 与保守删减后的上下文之间保持同一个正确答案，并避免因为删掉一条错误证据而突然空答？

新的候选需要在新的开发数据上训练和选择，并使用另一份未见数据最终确认。当前 F005 不授权继续在 sealed600 上试错。

## 可复核产物

- 主报告：[`artifacts/F005/eval/REPORT.md`](artifacts/F005/eval/REPORT.md)
- 主结果 JSON：[`artifacts/F005/eval/report.json`](artifacts/F005/eval/report.json)
- 独立引用报告：[`artifacts/F005/eval/CITATION_REPORT.md`](artifacts/F005/eval/CITATION_REPORT.md)
- 独立引用 JSON：[`artifacts/F005/eval/citation_report.json`](artifacts/F005/eval/citation_report.json)
- 600 题逐题生成：[`artifacts/F005/eval/generations.jsonl`](artifacts/F005/eval/generations.jsonl)
- 600 题逐句引用评分：[`artifacts/F005/eval/citation_per_case.jsonl`](artifacts/F005/eval/citation_per_case.jsonl)
- 运行清单：[`artifacts/F005/eval/run_manifest.json`](artifacts/F005/eval/run_manifest.json)
