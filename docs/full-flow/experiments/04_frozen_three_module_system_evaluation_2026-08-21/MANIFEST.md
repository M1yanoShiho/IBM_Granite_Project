# 文件清单

| 文件 | 用途 | 状态 |
|---|---|---|
| [README.md](README.md) | 零基础入口：五个独立目标、最终矩阵和当前状态 | v4-current |
| [PLAN.md](PLAN.md) | 总路线、五个独立执行目标、停止边界、实验矩阵与统计规则 | v4-master-roadmap |
| [TRACKER.md](TRACKER.md) | 五个目标的启动条件、交付物与实际状态 | v4-current |
| [GOAL_1_HANDOFF.md](GOAL_1_HANDOFF.md) | 新任务只执行 Goal 1 的完整边界、交付物和 PASS 条件 | complete-pass-stopped |
| [RESULT_TABLES.md](RESULT_TABLES.md) | 两张已填充的英文报告表格 | complete |
| [FINAL_REPORT.md](FINAL_REPORT.md) | Experiment 04 最终结论、有效性与范围声明 | FINAL PASS |
| [artifacts/goal1_data_manifest.json](artifacts/goal1_data_manifest.json) | Goal 1 count/schema/ordered-ID/hash 与隔离机器审计 | complete-pass |
| [artifacts/goal1_scorer_validation.json](artifacts/goal1_scorer_validation.json) | synthetic/revealed 五指标和失败分母验证 | complete-pass |
| [reports/GOAL1_DATA_SCORER_READINESS.md](reports/GOAL1_DATA_SCORER_READINESS.md) | Goal 1 最终数据与评分 readiness 报告 | PASS |
| [reports/GOAL2_SYSTEM_WIRING_READINESS.md](reports/GOAL2_SYSTEM_WIRING_READINESS.md) | Goal 2 接线、smoke 与冻结配置报告 | PASS |
| [reports/GOAL3_MAIN_SYSTEM_RESULTS.md](reports/GOAL3_MAIN_SYSTEM_RESULTS.md) | Goal 3 正式主系统结果与结论 | PASS |
| [reports/GOAL4_MODULE_ABLATION_RESULTS.md](reports/GOAL4_MODULE_ABLATION_RESULTS.md) | Goal 4 三模块消融正式结果与结论 | PASS |
| [artifacts/goal3_pass_manifest.json](artifacts/goal3_pass_manifest.json) | Goal 3 覆盖、隔离、运行与结果 hash 审计 | PASS |
| [artifacts/goal3_formal_manifests/](artifacts/goal3_formal_manifests/) | 三数据集 generation/score manifests | 6/6 PASS |
| [artifacts/goal4_pass_manifest.json](artifacts/goal4_pass_manifest.json) | Goal 4 覆盖、Full 复用、运行与结果 hash 审计 | PASS |
| [artifacts/goal4_formal_manifests/](artifacts/goal4_formal_manifests/) | 三数据集 Goal 4 generation/score manifests | 6/6 PASS |
| [artifacts/goal5_final_pass_manifest.json](artifacts/goal5_final_pass_manifest.json) | Goal 5 统一统计、hash 与最终结论清单 | FINAL PASS |
| [results/TABLE1.md](results/TABLE1.md) | 冻结 Table 1 | complete |
| [results/table1.json](results/table1.json) | Table 1 机器值 | complete |
| [results/per_query_metrics.csv](results/per_query_metrics.csv) | Goal 3 的 7,700 行逐题指标 | complete |
| [results/bootstrap_ci.json](results/bootstrap_ci.json) | Goal 3 的 12 个 paired RAR CI | complete |
| [results/goal3_audit.json](results/goal3_audit.json) | Goal 3 最终输出数与 hash 审计 | PASS |
| [results/TABLE2.md](results/TABLE2.md) | 冻结 Table 2 | complete |
| [results/table2.json](results/table2.json) | Table 2 机器值 | complete |
| [results/per_query_goal4_metrics.csv](results/per_query_goal4_metrics.csv) | Goal 4 的 4,400 行 Table 2 逐题指标（含复用 Full） | complete |
| [results/bootstrap_goal4_ci.json](results/bootstrap_goal4_ci.json) | Goal 4 的 9 个 Full-minus-ablation paired RAR CI | complete |
| [results/goal4_audit.json](results/goal4_audit.json) | Goal 4 最终输出数、Full 复用与 hash 审计 | PASS |
| [results/FINAL_TABLES.md](results/FINAL_TABLES.md) | 两张最终冻结 Markdown 表 | complete |
| [results/TABLE1.tex](results/TABLE1.tex) | Table 1 LaTeX | complete |
| [results/TABLE2.tex](results/TABLE2.tex) | Table 2 LaTeX | complete |
| [results/summary_metrics.csv](results/summary_metrics.csv) | 135 行统一长格式机器指标 | complete |
| [results/final_results.json](results/final_results.json) | 两表、两组 bootstrap 与最终 claim decisions | FINAL PASS |
| [results/final_audit.json](results/final_audit.json) | 跨 Goal 3/4 来源、统计与生成物 hash 最终审计 | FINAL PASS |
| [snapshots/PLAN_v1_2026-08-21.md](snapshots/PLAN_v1_2026-08-21.md) | 旧版三系统计划快照 | archived |
| [snapshots/RESULT_TABLES_v1_2026-08-21.md](snapshots/RESULT_TABLES_v1_2026-08-21.md) | 旧版结果表快照 | archived |
| [snapshots/PLAN_v2_2026-08-22.md](snapshots/PLAN_v2_2026-08-22.md) | 旧版两个 baseline 计划快照 | archived |
| [snapshots/RESULT_TABLES_v2_2026-08-22.md](snapshots/RESULT_TABLES_v2_2026-08-22.md) | 旧版两个 baseline 表格快照 | archived |
| [snapshots/PLAN_v3_2026-08-22.md](snapshots/PLAN_v3_2026-08-22.md) | 改为分目标执行前的单一连续路线快照 | archived |
| [snapshots/TRACKER_v3_2026-08-22.md](snapshots/TRACKER_v3_2026-08-22.md) | 改为分目标执行前的逐步骤跟踪表 | archived |

Goal 1–5 已全部完成；两张冻结表、统一 CSV/JSON、LaTeX、最终报告和 final audit
均已产生并通过 hash 复核。
