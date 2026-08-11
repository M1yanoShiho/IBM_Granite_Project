# Selector refine-logs manifest

**更新时间：** 2026-08-12

| 文件 | 用途 | 状态 |
|---|---|---|
| [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) | v2 最新计划固定副本；R001–R003 已完成，Gate 1 PASS，R004 NEXT | active-v2 |
| [`EXPERIMENT_PLAN_2026-08-11_v2.md`](EXPERIMENT_PLAN_2026-08-11_v2.md) | 自适应 0–cap（cap≤3）、component-aware CRC、Hybrid-v2 冻结；R003 baseline/protocol PASS | frozen-v2-active |
| [`EXPERIMENT_PLAN_2026-08-11.md`](EXPERIMENT_PLAN_2026-08-11.md) | 初版风险控制双头 Selector 计划 | archived-v1 |
| [`EXPERIMENT_TRACKER.md`](EXPERIMENT_TRACKER.md) | v2 最新 tracker 固定副本；R001–R003 PASS、R004 NEXT/TODO | active-v2 |
| [`EXPERIMENT_TRACKER_2026-08-11_v2.md`](EXPERIMENT_TRACKER_2026-08-11_v2.md) | v2 的 R001–R015、实现前 Gate、CRC/CI、TopK 数量基线与决策日志 | frozen-v2-active |
| [`EXPERIMENT_TRACKER_2026-08-11.md`](EXPERIMENT_TRACKER_2026-08-11.md) | 初版 tracker | archived-v1 |
| [`R001_RECOVERY_AUDIT.md`](../results/selector-adaptive-risk-v1/R001/R001_RECOVERY_AUDIT.md) | 六个 Hybrid RRF Top20 pool、数据 sidecar 与模型资产的本地/原 HPC 只读恢复审计 | R001A/B-pass-evidence |
| [`R001C_GATE_EVIDENCE.md`](../results/selector-adaptive-risk-v1/R001/R001C_GATE_EVIDENCE.md) | 六池 write-once v2 freeze、独立 verify-only、测试与机器可读 manifest 索引 | R001C-pool-pass-evidence |
| [`R002_PROTOCOL_REPORT.md`](../results/selector-adaptive-risk-v1/R002/R002_PROTOCOL_REPORT.md) | 指标、component/role、CRC 样本量门及其零基础解释 | R002-sample-size-go-evidence |
| [`R002_VALIDATION_REPORT.json`](../results/selector-adaptive-risk-v1/R002/R002_VALIDATION_REPORT.json) | 24 个 artifact hash、10 项 projection、四风险 representative n 与独立验证结果 | R002-machine-readable-evidence |
| [`R003_PROTOCOL_REPORT.md`](../results/selector-adaptive-risk-v1/R003/R003_PROTOCOL_REPORT.md) | TopK 数量基线、固定 TopK9 的删除/保留冲突及 count-matched 协议的零基础解释 | R003-baseline-protocol-pass-evidence |
| [`R003_VALIDATION_REPORT.json`](../results/selector-adaptive-risk-v1/R003/R003_VALIDATION_REPORT.json) | 四组 TopK 结果、14 个原始 artifact hash、复验与测试状态 | R003-machine-readable-evidence |
| [`R003_CONFIG.json`](../results/selector-adaptive-risk-v1/R003/R003_CONFIG.json) | R003 数据范围、TopK/paired 统计与 100-repeat 等量删除冻结配置 | R003-frozen-config |
| [`selector_experiment_manifest.json`](../results/selector-adaptive-risk-v1/R003/selector_experiment_manifest.json) | R003 run-level 依赖 hash；明确 scores/decision trace 不适用、count-matched 结果延后 | R003-run-manifest |
| [`R003_RUN_LOG.md`](../results/selector-adaptive-risk-v1/R003/R003_RUN_LOG.md) | 正式生成、verify-only、服务器/本地一致性与测试记录 | R003-run-log |
| [`CHECKSUMS.sha256`](../results/selector-adaptive-risk-v1/R003/CHECKSUMS.sha256) | 14 个原始生成文件的 SHA-256；后写总结文件因避免自引用而不在此表 | R003-raw-artifact-checksums |

## 更新规则

1. 计划发生实质变化时，新建带日期版本，不覆盖旧版本。
2. 更新固定入口链接与本 manifest。
3. 计划、tracker 和实际结果目录中的 run id 必须一致。
4. 失败结果与停止决定属于正式研究记录，不删除。
