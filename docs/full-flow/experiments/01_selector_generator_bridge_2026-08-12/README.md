# 01 — Selector–Generator 跨阶段桥接

**日期：** 2026-08-12
**状态：** `COMPLETE / F005 FINAL-NO-WIN`

## 零基础说明

这不是开发第四个模块。研究系统仍然只有三部分：

```text
Retriever → Selector → Generator
```

本路线分三步：

1. 先把最新 Selector 与已经完成的 `VerifyAnnotateGenerator` 正式联合，观察真实问题；
2. 只根据第一步暴露的问题，在现有 Generator 内增加最小的跨阶段信息利用；
3. 只有轻量改造仍不能让 Generator 适应 Selector 输出时，才启动小规模鲁棒训练。

第二步目前只冻结“设计原则”和候选接口，不冻结最终功能组合。最后到底启用保护信号、风险信号、关键事实笔记还是运行时 `EvidenceReadiness` 估计，必须由第一步的结果决定。`EvidenceReadiness` 只能由问题和当前 selected evidence 估计，gold chain 只用于生成后的实验评分。

## 文件

- [完整实验与改造计划](PLAN.md)
- [v2 运行时信号边界修订](AMENDMENT_RUNTIME_SIGNAL_BOUNDARY_v2.md)
- [执行跟踪表](TRACKER.md)
- [证据与结果入口](EVIDENCE_INDEX.md)
- [F000 联合实验入口冻结报告](F000_IMPLEMENTATION_REPORT.md)
- [F001/F002 联合结果与问题归因](F001_F002_RESULTS.md)
- [F003A/F003B 实施报告](F003_IMPLEMENTATION_REPORT.md)
- [F004 正式结果与停止决定](F004_RESULTS.md)
- [F006 小规模鲁棒训练报告](F006_TRAINING_REPORT.md)
- [F006 正式开发集结果](F006_RESULTS.md)
- [F005 独立确认 Selector 阶段](F005_SELECTION_REPORT.md)
- [F005 独立最终确认结果](F005_RESULTS.md)

## 最终结果

F005 在独立 600 题上完成后，默认 TopK 和完整系统的答案正确率同为 63.17%；固定 Generator 后，Selector 的净作用为 −0.17 个百分点。当前路线因此没有证明端到端提升，sealed600 已退休，不能再用于调参。

但 Selector 的证据层能力仍然成立：82 条删除中有 74 条是已知有害证据。失败主要发生在 Generator 对删减后上下文不稳定，而不是 Selector 在最终答案变化的 5 题中删错了 gold 文档。完整解释见 [F005_RESULTS.md](F005_RESULTS.md)。
