# Selector refine-logs manifest

**更新时间：** 2026-08-12

| 文件 | 用途 | 状态 |
|---|---|---|
| [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) | 原 v2 固定副本；R001–R004 PASS、R005 FAIL、R006–R015 CUT、TopK10 默认 | frozen-v2-stopped |
| [`EXPERIMENT_PLAN_2026-08-11_v2.md`](EXPERIMENT_PLAN_2026-08-11_v2.md) | 原 v2 带日期版本；完整保留 R005 停止路线与当前复验/审计边界 | frozen-v2-stopped |
| [`EXPERIMENT_PLAN_2026-08-11.md`](EXPERIMENT_PLAN_2026-08-11.md) | 初版风险控制双头 Selector 计划 | archived-v1 |
| [`EXPERIMENT_TRACKER.md`](EXPERIMENT_TRACKER.md) | 原 v2 tracker 固定副本；R001–R004 PASS、R005 FAIL、后续 CUT | frozen-v2-stopped |
| [`EXPERIMENT_TRACKER_2026-08-11_v2.md`](EXPERIMENT_TRACKER_2026-08-11_v2.md) | 原 v2 带日期 tracker；保留全部决定日志 | frozen-v2-stopped |
| [`EXPERIMENT_TRACKER_2026-08-11.md`](EXPERIMENT_TRACKER_2026-08-11.md) | 初版 tracker | archived-v1 |
| [`EXPERIMENT_PLAN_AMENDMENT.md`](EXPERIMENT_PLAN_AMENDMENT.md) | 当前修订固定入口 | draft-waiting-approval |
| [`EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB.md`](EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB.md) | fresh component-disjoint R005A/R005B、V0/V1/V2、two-seed one-shot confirm 完整计划 | draft-waiting-approval |
| [`EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB.json`](EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB.json) | 修订协议、样本 hash、variants、预算和门槛的机器可读冻结草案 | draft-machine-contract |
| [`EXPERIMENT_TRACKER_AMENDMENT.md`](EXPERIMENT_TRACKER_AMENDMENT.md) | 当前修订 tracker 固定入口 | draft-waiting-approval |
| [`EXPERIMENT_TRACKER_AMENDMENT_2026-08-12_R005AB.md`](EXPERIMENT_TRACKER_AMENDMENT_2026-08-12_R005AB.md) | 批准、样本、实现、R005A/B 和 R006A 条件队列 | draft-waiting-approval |
| [`R005_EXECUTION_REPORT_2026-08-12.md`](R005_EXECUTION_REPORT_2026-08-12.md) | 正式 R005 TRAINING-GATE FAIL、短路与零基础解释 | R005-formal-fail-report |
| [`R005_VERIFICATION_ATTESTATION_2026-08-12.md`](R005_VERIFICATION_ATTESTATION_2026-08-12.md) | 当前 runner/finalizer 命令、commit、退出码、输出与复验后 hash | R005-current-verification-pass |
| [`R005_VERIFICATION_ATTESTATION_2026-08-12.json`](R005_VERIFICATION_ATTESTATION_2026-08-12.json) | 当前 R005 双重复验机器凭据 | R005-current-verification-pass |
| [`R005_EXPERIMENT_INTEGRITY_AUDIT_2026-08-12.md`](R005_EXPERIMENT_INTEGRITY_AUDIT_2026-08-12.md) | 正式 bundle、代码、报告与泄漏边界的独立完整性审计 | WARN-P0-0-P1-1-history-attestation |
| [`R005_EXPERIMENT_INTEGRITY_AUDIT_2026-08-12.json`](R005_EXPERIMENT_INTEGRITY_AUDIT_2026-08-12.json) | R005 独立完整性审计机器结论 | WARN-P0-0-P1-1-history-attestation |
| [`EXPERIMENT_AUDIT.md`](EXPERIMENT_AUDIT.md) / [`EXPERIMENT_AUDIT.json`](EXPERIMENT_AUDIT.json) | 最新实验完整性审计固定入口 | current-audit-entry |
| [`R005_POSTHOC_DIAGNOSIS_2026-08-12.md`](R005_POSTHOC_DIAGNOSIS_2026-08-12.md) | 只用 train-fit 分数的阈值不可行性、margin、AUC 与架构事实诊断 | posthoc-design-evidence-only |
| [`R005_POSTHOC_DIAGNOSIS_2026-08-12.json`](R005_POSTHOC_DIAGNOSIS_2026-08-12.json) | R005 失败诊断机器摘要 | posthoc-design-evidence-only |
| [`R005AB_DESIGN_AUDIT_2026-08-12.md`](R005AB_DESIGN_AUDIT_2026-08-12.md) | 三个领域设计审计 + 两个稳定快照批准前终审 | plan-audited-P0-0-P1-0-P2-0-not-run |
| [`R005AB_DESIGN_AUDIT_2026-08-12.json`](R005AB_DESIGN_AUDIT_2026-08-12.json) | R005A/B 设计与终审机器摘要 | ready-for-A001-after-approval-not-run |
| [`.aris/traces/experiment-audit/2026-08-12_run01`](../.aris/traces/experiment-audit/2026-08-12_run01/) | R005 正式完整性审计 request、reviewer output 与 metadata | persisted-reviewer-trace |
| [`.aris/traces/experiment-audit/2026-08-12_r005ab_sample`](../.aris/traces/experiment-audit/2026-08-12_r005ab_sample/) | R005A/B 样本与隔离审计 trace | persisted-reviewer-trace |
| [`.aris/traces/experiment-audit/2026-08-12_r005ab_stats`](../.aris/traces/experiment-audit/2026-08-12_r005ab_stats/) | R005A/B 统计门与揭示顺序审计 trace | persisted-reviewer-trace |
| [`.aris/traces/experiment-audit/2026-08-12_r005ab_code`](../.aris/traces/experiment-audit/2026-08-12_r005ab_code/) | R005A/B 模型、loss 与 batch 审计 trace | persisted-reviewer-trace |
| [`.aris/traces/experiment-audit/2026-08-12_r005ab_final_state_machine`](../.aris/traces/experiment-audit/2026-08-12_r005ab_final_state_machine/) | 最终稳定快照的防泄漏、formal-fit、崩溃恢复与语义重算对抗审计 | PASS-P0-0-P1-0-P2-0 |
| [`.aris/traces/experiment-audit/2026-08-12_r005ab_final_consistency`](../.aris/traces/experiment-audit/2026-08-12_r005ab_final_consistency/) | 最终稳定快照的 MD/JSON/tracker 一致性终审 | PASS-P0-0-P1-0-P2-0 |
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
| [`R004 selector_experiment_manifest.json`](../results/selector-adaptive-risk-v1/R004/selector_experiment_manifest.json) | 80,460 标签、200q GPU preflight、输入与资源 pins | R004-resource-preflight-pass |
| [`R004 CHECKSUMS.sha256`](../results/selector-adaptive-risk-v1/R004/CHECKSUMS.sha256) | R004 正式 artifact checksums | R004-integrity-evidence |

## 更新规则

1. 计划发生实质变化时，新建带日期版本，不覆盖旧版本。
2. 更新固定入口链接与本 manifest。
3. 计划、tracker 和实际结果目录中的 run id 必须一致。
4. 失败结果与停止决定属于正式研究记录，不删除。
