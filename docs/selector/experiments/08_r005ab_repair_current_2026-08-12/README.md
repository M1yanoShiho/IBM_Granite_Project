# 08 — 当前 Selector Lean v3

**时间：** 2026-08-12 起

**状态：** `COMPLETE / EVIDENCE PASS / ANSWER FAIL / KEEP TOPK10`

## 这次要验证什么

固定输入仍是默认 `TopK10`。Selector 不重新检索，也不补入 TopK11–20，只判断 TopK10 中哪些证据是“高 harmful、低 protect”，然后保守地删除 0 条、最多 1 或 2 条；不确定或分数冲突时保留。

旧 R005 的失败保留不变：它用 raw CLS 加随机双头，无法形成可靠的绝对删除边界。Lean v3 只替换这部分评分器，恢复预训练 NLI 信息；`NLI-pair` 仅在基础版本开发失败时启用。R001–R004 已验证的数据、TopK10、动作规则和指标继续复用。

## 当前边界

- L000 文档冻结、L001 最小实现和 L002 训练/开发选择均已通过；
- L003 已按冻结的 `NLI-base + q=.99 + cap2` 完成一次性 final；
- 最终证据门 PASS：harmful evidence 减少约 13%，required recall/chain 损失为 0；
- 最终答案门 FAIL：macro answer delta 为 −0.1353 pp，不能证明比 TopK10 提升；
- v1/v2 都从未运行，现作为被 v3 取代的历史完整保留；
- TopK10 仍是唯一默认，除非最终实验通过才讨论变更。

## 文件入口

- [计划导航](PLAN.md)
- [状态记录](TRACKER.md)
- [证据索引](EVIDENCE_INDEX.md)
- [来源清单](SOURCE_MANIFEST.json)
- [L001 最小实现报告](L001_IMPLEMENTATION_REPORT.md)
- [L002 训练与开发选择报告](L002_DEVELOPMENT_REPORT.md)
- [L003 最终盲测报告](L003_FINAL_REPORT.md)
