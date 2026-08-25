# Selector refine-logs manifest

**更新时间：** 2026-08-12

跨阶段浏览入口：[Selector 历次实验总索引](../docs/selector/experiments/README.md)。该目录是带来源哈希的归档导航层，不替代本清单中的 canonical 文件。

| 文件 | 用途 | 状态 |
|---|---|---|
| [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) | 原 v2 固定副本；R001–R004 PASS、R005 FAIL、R006–R015 CUT、TopK10 默认 | frozen-v2-stopped |
| [`EXPERIMENT_PLAN_2026-08-11_v2.md`](EXPERIMENT_PLAN_2026-08-11_v2.md) | 原 v2 带日期版本；完整保留 R005 停止路线与当前复验/审计边界 | frozen-v2-stopped |
| [`EXPERIMENT_PLAN_2026-08-11.md`](EXPERIMENT_PLAN_2026-08-11.md) | 初版风险控制双头 Selector 计划 | archived-v1 |
| [`EXPERIMENT_TRACKER.md`](EXPERIMENT_TRACKER.md) | 原 v2 tracker 固定副本；R001–R004 PASS、R005 FAIL、后续 CUT | frozen-v2-stopped |
| [`EXPERIMENT_TRACKER_2026-08-11_v2.md`](EXPERIMENT_TRACKER_2026-08-11_v2.md) | 原 v2 带日期 tracker；保留全部决定日志 | frozen-v2-stopped |
| [`EXPERIMENT_TRACKER_2026-08-11.md`](EXPERIMENT_TRACKER_2026-08-11.md) | 初版 tracker | archived-v1 |
| [`EXPERIMENT_PLAN_AMENDMENT.md`](EXPERIMENT_PLAN_AMENDMENT.md) | 当前修订固定入口；指向已获批并冻结的 Lean v3 | current-v3-lean-L000-pass-not-run |
| [`EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB_v3_LEAN.md`](EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB_v3_LEAN.md) | TopK10 内 NLI-aware 保守删除；只保留 train/dev/final 隔离、必要对照与一次盲测 | current-v3-lean-frozen-for-implementation-not-run |
| [`EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB_v3_LEAN.json`](EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB_v3_LEAN.json) | Lean v3 数据、模型、阈值、统计与运行边界的机器合同 | current-v3-lean-contract-L000-pass-not-run |
| [`EXPERIMENT_TRACKER_AMENDMENT.md`](EXPERIMENT_TRACKER_AMENDMENT.md) | 当前修订 tracker 固定入口；Lean v3 已完成，证据门 PASS、答案门 FAIL | v3-lean-complete-keep-topk10 |
| [`EXPERIMENT_TRACKER_AMENDMENT_2026-08-12_R005AB_v3_LEAN.md`](EXPERIMENT_TRACKER_AMENDMENT_2026-08-12_R005AB_v3_LEAN.md) | Lean v3 的 L000–L003 完整执行记录 | v3-lean-L003-complete-overall-fail |
| [`EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB_v2.md`](EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB_v2.md) | 被 Lean v3 取代的重型 v2 计划；从未运行 | superseded-v2-draft-never-run |
| [`EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB_v2.json`](EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB_v2.json) | 被 Lean v3 取代的 v2 机器合同；从未运行 | superseded-v2-draft-never-run |
| [`EXPERIMENT_TRACKER_AMENDMENT_2026-08-12_R005AB_v2.md`](EXPERIMENT_TRACKER_AMENDMENT_2026-08-12_R005AB_v2.md) | 被 Lean v3 取代的 v2 tracker；从未运行 | superseded-v2-draft-never-run |
| [`EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB.md`](EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB.md) | 被 v2 取代的 R005A/R005B v1 草案；仅作历史记录 | superseded-v1-draft-never-approved-never-run |
| [`EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB.json`](EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB.json) | 被 v2 取代的 v1 机器合同；仅作历史记录 | superseded-v1-draft-never-approved-never-run |
| [`EXPERIMENT_TRACKER_AMENDMENT_2026-08-12_R005AB.md`](EXPERIMENT_TRACKER_AMENDMENT_2026-08-12_R005AB.md) | 被 v2 取代的 v1 tracker；仅作历史记录 | superseded-v1-draft-never-approved-never-run |
| [`R005_EXECUTION_REPORT_2026-08-12.md`](R005_EXECUTION_REPORT_2026-08-12.md) | 正式 R005 TRAINING-GATE FAIL、短路与零基础解释 | R005-formal-fail-report |
| [`R005_VERIFICATION_ATTESTATION_2026-08-12.md`](R005_VERIFICATION_ATTESTATION_2026-08-12.md) | 当前 runner/finalizer 命令、commit、退出码、输出与复验后 hash | R005-current-verification-pass |
| [`R005_VERIFICATION_ATTESTATION_2026-08-12.json`](R005_VERIFICATION_ATTESTATION_2026-08-12.json) | 当前 R005 双重复验机器凭据 | R005-current-verification-pass |
| [`R005_EXPERIMENT_INTEGRITY_AUDIT_2026-08-12.md`](R005_EXPERIMENT_INTEGRITY_AUDIT_2026-08-12.md) | 正式 bundle、代码、报告与泄漏边界的独立完整性审计 | WARN-P0-0-P1-1-history-attestation |
| [`R005_EXPERIMENT_INTEGRITY_AUDIT_2026-08-12.json`](R005_EXPERIMENT_INTEGRITY_AUDIT_2026-08-12.json) | R005 独立完整性审计机器结论 | WARN-P0-0-P1-1-history-attestation |
| [`EXPERIMENT_AUDIT.md`](EXPERIMENT_AUDIT.md) / [`EXPERIMENT_AUDIT.json`](EXPERIMENT_AUDIT.json) | 最新实验完整性审计固定入口 | current-audit-entry |
| [`R005_POSTHOC_DIAGNOSIS_2026-08-12.md`](R005_POSTHOC_DIAGNOSIS_2026-08-12.md) | 只用 train-fit 分数的阈值不可行性、margin、AUC 与架构事实诊断 | posthoc-design-evidence-only |
| [`R005_POSTHOC_DIAGNOSIS_2026-08-12.json`](R005_POSTHOC_DIAGNOSIS_2026-08-12.json) | R005 失败诊断机器摘要 | posthoc-design-evidence-only |
| [`R005AB_DESIGN_AUDIT_2026-08-12.md`](R005AB_DESIGN_AUDIT_2026-08-12.md) | v1 历史审计；后续复查发现四项漏报，不能作为执行授权 | historical-v1-false-negative-superseded |
| [`R005AB_DESIGN_AUDIT_2026-08-12.json`](R005AB_DESIGN_AUDIT_2026-08-12.json) | v1 历史审计机器摘要；已被纠错复审取代 | historical-v1-false-negative-superseded |
| [`R005AB_CORRECTIVE_REAUDIT_2026-08-12.md`](R005AB_CORRECTIVE_REAUDIT_2026-08-12.md) | 四项 v1 审计漏报与 v2 修复的历史纠错记录 | historical-v2-reaudit-superseded-by-v3 |
| [`R005AB_CORRECTIVE_REAUDIT_2026-08-12.json`](R005AB_CORRECTIVE_REAUDIT_2026-08-12.json) | v2 纠错复审机器摘要与稳定文件哈希 | historical-v2-reaudit-superseded-by-v3 |
| [`R005AB_A001_IMPLEMENTATION_MAP_2026-08-12.md`](R005AB_A001_IMPLEMENTATION_MAP_2026-08-12.md) | 被 Lean v3 取代的重型 v2 实现映射；从未实现 | superseded-v2-map-never-implemented |
| [`R005AB_A001_IMPLEMENTATION_MAP_2026-08-12.json`](R005AB_A001_IMPLEMENTATION_MAP_2026-08-12.json) | 被 Lean v3 取代的 v2 机器实现映射 | superseded-v2-map-never-implemented |
| [`R005AB_SERVER_READINESS_AUDIT_2026-08-12.md`](R005AB_SERVER_READINESS_AUDIT_2026-08-12.md) | 针对 v1 快照的服务器环境只读核验；能力事实可复用，但 v2 必须在 A001 前重查 | historical-v1-snapshot-environment-capability-only |
| [`R005AB_SERVER_READINESS_AUDIT_2026-08-12.json`](R005AB_SERVER_READINESS_AUDIT_2026-08-12.json) | v1 快照服务器环境机器摘要；不代表 v2 行为测试或执行就绪 | historical-v1-snapshot-environment-capability-only |
| [`.aris/traces/experiment-audit/2026-08-12_run01`](../.aris/traces/experiment-audit/2026-08-12_run01/) | R005 正式完整性审计 request、reviewer output 与 metadata | persisted-reviewer-trace |
| [`.aris/traces/experiment-audit/2026-08-12_r005ab_sample`](../.aris/traces/experiment-audit/2026-08-12_r005ab_sample/) | R005A/B 样本与隔离审计 trace | persisted-reviewer-trace |
| [`.aris/traces/experiment-audit/2026-08-12_r005ab_stats`](../.aris/traces/experiment-audit/2026-08-12_r005ab_stats/) | R005A/B 统计门与揭示顺序审计 trace | persisted-reviewer-trace |
| [`.aris/traces/experiment-audit/2026-08-12_r005ab_code`](../.aris/traces/experiment-audit/2026-08-12_r005ab_code/) | R005A/B 模型、loss 与 batch 审计 trace | persisted-reviewer-trace |
| [`.aris/traces/experiment-audit/2026-08-12_r005ab_final_state_machine`](../.aris/traces/experiment-audit/2026-08-12_r005ab_final_state_machine/) | v1 状态机历史审计；后续发现漏报，不能作为当前 PASS 或授权 | historical-v1-audit-superseded-false-negative |
| [`.aris/traces/experiment-audit/2026-08-12_r005ab_final_consistency`](../.aris/traces/experiment-audit/2026-08-12_r005ab_final_consistency/) | v1 一致性历史审计；后续发现漏报，不能作为当前 PASS 或授权 | historical-v1-audit-superseded-false-negative |
| [`.aris/traces/experiment-audit/2026-08-12_r005ab_v2_state_machine`](../.aris/traces/experiment-audit/2026-08-12_r005ab_v2_state_machine/) | 被 Lean v3 取代的 v2 状态机历史复审 | historical-v2-PASS-superseded-never-run |
| [`.aris/traces/experiment-audit/2026-08-12_r005ab_v2_consistency`](../.aris/traces/experiment-audit/2026-08-12_r005ab_v2_consistency/) | 被 Lean v3 取代的 v2 跨文档历史复审 | historical-v2-PASS-superseded-never-run |
| [`.aris/traces/experiment-audit/2026-08-12_r005ab_server_readiness`](../.aris/traces/experiment-audit/2026-08-12_r005ab_server_readiness/) | v1 amendment 目标服务器的历史只读环境审计 | historical-v1-snapshot-WARN-P0-0-P1-1-P2-0 |
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
