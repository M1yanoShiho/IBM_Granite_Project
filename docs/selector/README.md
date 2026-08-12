# Selector 文档入口

这里统一保存 Selector 部分的历史实验路线、当前计划和最终决定。

## 先看哪一份

1. [历次实验总索引](experiments/README.md)：按时间查看每一次独立路线。
2. [当前 R005A/R005B 修复路线](experiments/08_r005ab_repair_current_2026-08-12/README.md)：当前唯一仍可能继续的路线。
3. [机器可读实验索引](experiments/MANIFEST.json)：供脚本核对八条路线和当前授权状态。
4. [2026-08-10 阶段性最终报告](SELECTOR_FINAL_REPORT.md)：解释为什么当时回退到 TopK10。
5. [refine-logs 总清单](../../refine-logs/MANIFEST.md)：近期冻结计划、机器合同、审计和正式结果的 canonical 索引。

## 当前状态

- 生产默认仍是 **TopK10**。
- Adaptive-Risk 原路线已在 R005 正式失败，R006–R015 按门停止。
- R005A/R005B recovery v2 仍在 `A000 WAITING APPROVAL`；A001、样本物化、训练和 held-out 读取均未执行。
- `docs/selector/experiments/` 是方便人阅读的归档层，不替代 `refine-logs/`、`results/` 或 Git 历史中的原始证据。

## 归档原则

- 一次独立方法/实验路线一个文件夹。
- 同一路线的修订、补充检查和失败诊断不伪装成新的独立实验。
- 被取代的草案放入同一路线的 `snapshots/superseded/`。
- 已冻结的原文件不移动、不改名；归档快照记录来源 commit、原路径和 Git blob。
- 历史文件里的状态行反映当时快照，最终状态以该文件夹 `README.md` 和 `TRACKER.md` 为准；`EVIDENCE_INDEX.md` 只负责导航证据。
