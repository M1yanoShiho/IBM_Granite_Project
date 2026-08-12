# 状态记录

原始逐项 tracker 保存在 [snapshots/EXPERIMENT_TRACKER_g2_proto5.md](snapshots/EXPERIMENT_TRACKER_g2_proto5.md)。最终边界如下：

| 阶段 | 状态 | 解释 |
|---|---|---|
| M0/协议修订 A1–A4 | PARTIAL / EVOLVING | 多次纠错均保留，不等于最终方法通过 |
| R012 零训练 Gate 0B | FAIL | 现成关系模型没有达到联合门 |
| R013 seed13 正式训练路径 | PARTIAL SUCCESS | 训练收敛且三项训练标准满足；只有一个 seed |
| 真实 Top20 pool probe | BLOCKING NEGATIVE | false-conflict rate 约 `.99995`，平均 `13.34` 个/题，UNKNOWN 约 `.005%` |
| R014/R015 与 Selector-level C1 | NOT RUN / NO SELECTOR-LEVEL RESULT | 没有三 seed 和 sealed Selector 结论 |
| 生产注册 | NOT ALLOWED | 路线退役 |

训练阶段的成功不能覆盖真实候选池上的阻断性失败。
