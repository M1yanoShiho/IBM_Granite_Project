# 06 — 三分类 Beam Selector

**时间：** 2026-08-09 至 2026-08-10

**状态：** `COMPLETE / M2 FAIL / RETIRED`

## 零基础解释

这条路线把候选分成 `REQUIRED / HARMFUL / IRRELEVANT` 三类，并用 beam 逐步组合证据，希望只删除高置信 harmful/irrelevant，同时保护必要证据。

- M0 数据与代码冻结：PASS；
- M1 32 题最小正确性：PASS；
- M2 seed13 开发集：16 组阈值全部超过 recall 损失上限；
- M3 三 seed 和 M4 正式测试：按停止门取消。

最佳保护配置仍让 NIAH required recall 下降约 17.30 个百分点，因此回退 TopK。

## 历史快照注意

最后保留的计划正文仍写着“下一步为 M2”，因为 M2 结论写入了独立结果报告而没有回写计划状态行。归档以最终报告为真实终态，不修改历史原文。

## 文件入口

- [计划导航](PLAN.md)
- [状态记录](TRACKER.md)
- [证据索引](EVIDENCE_INDEX.md)
- [来源清单](SOURCE_MANIFEST.json)
