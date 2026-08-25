# 01 — Corroboration reranking

**时间：** 2026-07-05 至 2026-07-07

**状态：** `COMPLETED / LIMITED POSITIVE SIGNAL / NOT A PRODUCTION SELECTOR`

## 零基础解释

这条路线不训练删除器，而是观察多条候选证据是否给出相同答案；相互印证较多的证据得到更高排序。它回答的是“互证能否给普通相关性排序增加一点信息”，还没有真正解决“该删除哪条错误证据”。

nested-CV 结果显示 needle-found@10 约提升 3.7 个百分点，但 MRR 基本不变。这个结果支持“corroboration 是一个有用信号”，不支持“它已经能安全删除 harmful evidence”。

## 为什么单独归档

2026-07-05 的实现计划、2026-07-06 的 gated 检查和 nested-CV 复验是同一条 reranking 路线的逐步验证，不拆成三次实验。它与 2026-07-20 开始的硬删除 Gated Graph 1.0 不同；后者在 [第 03 文件夹](../03_gated_graph1_2026-07-20/README.md)。

## 文件入口

- [计划导航](PLAN.md)
- [状态记录](TRACKER.md)
- [证据索引](EVIDENCE_INDEX.md)
- [来源清单](SOURCE_MANIFEST.json)
