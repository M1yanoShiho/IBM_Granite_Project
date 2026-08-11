# Selector refine-logs manifest

**更新时间：** 2026-08-11

| 文件 | 用途 | 状态 |
|---|---|---|
| [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) | v2 最新计划固定副本；R001/R002 已完成，R003 正在执行 | active-v2 |
| [`EXPERIMENT_PLAN_2026-08-11_v2.md`](EXPERIMENT_PLAN_2026-08-11_v2.md) | 自适应 0–cap（cap≤3）、component-aware CRC、Hybrid-v2 冻结；R002 sample-size GO | frozen-v2-active |
| [`EXPERIMENT_PLAN_2026-08-11.md`](EXPERIMENT_PLAN_2026-08-11.md) | 初版风险控制双头 Selector 计划 | archived-v1 |
| [`EXPERIMENT_TRACKER.md`](EXPERIMENT_TRACKER.md) | v2 最新 tracker 固定副本；R001/R002 PASS、R003 RUNNING | active-v2 |
| [`EXPERIMENT_TRACKER_2026-08-11_v2.md`](EXPERIMENT_TRACKER_2026-08-11_v2.md) | v2 的 R001–R015、实现前 Gate、CRC/CI 与恢复决策日志 | frozen-v2-active |
| [`EXPERIMENT_TRACKER_2026-08-11.md`](EXPERIMENT_TRACKER_2026-08-11.md) | 初版 tracker | archived-v1 |
| [`R001_RECOVERY_AUDIT.md`](../results/selector-adaptive-risk-v1/R001/R001_RECOVERY_AUDIT.md) | 六个 Hybrid RRF Top20 pool、数据 sidecar 与模型资产的本地/原 HPC 只读恢复审计 | R001A/B-pass-evidence |
| [`R001C_GATE_EVIDENCE.md`](../results/selector-adaptive-risk-v1/R001/R001C_GATE_EVIDENCE.md) | 六池 write-once v2 freeze、独立 verify-only、测试与机器可读 manifest 索引 | R001C-pool-pass-evidence |
| [`R002_PROTOCOL_REPORT.md`](../results/selector-adaptive-risk-v1/R002/R002_PROTOCOL_REPORT.md) | 指标、component/role、CRC 样本量门及其零基础解释 | R002-sample-size-go-evidence |
| [`R002_VALIDATION_REPORT.json`](../results/selector-adaptive-risk-v1/R002/R002_VALIDATION_REPORT.json) | 24 个 artifact hash、10 项 projection、四风险 representative n 与独立验证结果 | R002-machine-readable-evidence |

## 更新规则

1. 计划发生实质变化时，新建带日期版本，不覆盖旧版本。
2. 更新固定入口链接与本 manifest。
3. 计划、tracker 和实际结果目录中的 run id 必须一致。
4. 失败结果与停止决定属于正式研究记录，不删除。
