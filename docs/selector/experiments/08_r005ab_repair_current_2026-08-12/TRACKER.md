# 状态记录

- [Lean v3 tracker 快照](snapshots/EXPERIMENT_TRACKER_AMENDMENT_R005AB_v3_LEAN.md)
- [canonical tracker 固定入口](../../../../refine-logs/EXPERIMENT_TRACKER_AMENDMENT.md)

| ID | 当前状态 | 许可边界 |
|---|---|---|
| 旧 R005 | FAIL | raw-CLS 随机双头失败结论保持不变 |
| v1/v2 | SUPERSEDED / NEVER RUN | 历史计划完整保留，不作为当前入口 |
| L000 | PASS | Lean v3 冻结；最终复核 P0=0/P1=0 |
| L001 | PASS | 最小实现、本地检查与服务器真实模型 smoke 已通过 |
| L002 | PASS | 冻结 `NLI-base + q=.99 + cap2`；两 seed 方向一致且保护门通过 |
| L003 | NEXT / FINAL UNOPENED | 只允许一次最终盲测 |

归档同步不代表实验已经成功；TopK10 仍是默认。

L001 的可读报告见 [L001_IMPLEMENTATION_REPORT.md](L001_IMPLEMENTATION_REPORT.md)。
L002 的完整开发结果见 [L002_DEVELOPMENT_REPORT.md](L002_DEVELOPMENT_REPORT.md)。
