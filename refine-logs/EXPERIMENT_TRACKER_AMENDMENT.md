# 当前 Selector 修订实验 tracker

**当前版本：** [`EXPERIMENT_TRACKER_AMENDMENT_2026-08-12_R005AB_v3_LEAN.md`](EXPERIMENT_TRACKER_AMENDMENT_2026-08-12_R005AB_v3_LEAN.md)

**状态：** `L000–L003 COMPLETE / EVIDENCE PASS / ANSWER FAIL / KEEP TOPK10`

Lean v3 已完成。NLI-aware scorer 在 final 将 harmful evidence 减少约 13%，required recall/chain 损失为 0；但 macro answer delta 为 −0.1353pp，未达到必须大于 0 的答案门。旧 R005 保持 FAIL，v1/v2 amendment 保留为从未运行的历史，TopK10 保持唯一生产默认。完整结论见 [`../docs/selector/experiments/08_r005ab_repair_current_2026-08-12/L003_FINAL_REPORT.md`](../docs/selector/experiments/08_r005ab_repair_current_2026-08-12/L003_FINAL_REPORT.md)。
