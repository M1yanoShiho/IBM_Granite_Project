# Full-flow 文档清单

**更新时间：** 2026-08-18

| 文件 | 用途 | 状态 |
|---|---|---|
| [`README.md`](README.md) | Retriever、Selector、Generator 与 full-flow 文档的目录分工 | current |
| [`experiments/README.md`](experiments/README.md) | 完整证据流实验总索引 | current |
| [`experiments/01_selector_generator_bridge_2026-08-12/README.md`](experiments/01_selector_generator_bridge_2026-08-12/README.md) | 第 01 路线零基础入口 | complete-F005-final-no-win |
| [`experiments/01_selector_generator_bridge_2026-08-12/PLAN.md`](experiments/01_selector_generator_bridge_2026-08-12/PLAN.md) | 三阶段实验与改造计划 v2 | complete |
| [`experiments/01_selector_generator_bridge_2026-08-12/AMENDMENT_RUNTIME_SIGNAL_BOUNDARY_v2.md`](experiments/01_selector_generator_bridge_2026-08-12/AMENDMENT_RUNTIME_SIGNAL_BOUNDARY_v2.md) | Selector 信号、Generator 运行时估计与 gold 评测标签的严格边界 | current-v2-amendment-partially-run |
| [`experiments/01_selector_generator_bridge_2026-08-12/TRACKER.md`](experiments/01_selector_generator_bridge_2026-08-12/TRACKER.md) | F000–F006 执行跟踪表 | complete-F005-final-no-win |
| [`experiments/01_selector_generator_bridge_2026-08-12/EVIDENCE_INDEX.md`](experiments/01_selector_generator_bridge_2026-08-12/EVIDENCE_INDEX.md) | 已有上游证据与未来产物入口 | current |
| [`experiments/01_selector_generator_bridge_2026-08-12/F000_IMPLEMENTATION_REPORT.md`](experiments/01_selector_generator_bridge_2026-08-12/F000_IMPLEMENTATION_REPORT.md) | 联合输入、四臂入口与运行时 gold 隔离核对 | PASS |
| [`experiments/01_selector_generator_bridge_2026-08-12/F001_F002_RESULTS.md`](experiments/01_selector_generator_bridge_2026-08-12/F001_F002_RESULTS.md) | 现有四臂联合结果、独立引用评分与109题归因 | COMPLETE-NO-WIN |
| [`experiments/01_selector_generator_bridge_2026-08-12/F003_IMPLEMENTATION_REPORT.md`](experiments/01_selector_generator_bridge_2026-08-12/F003_IMPLEMENTATION_REPORT.md) | runtime-safe Selector sidecar 与关键事实笔记实现 | COMPLETE |
| [`experiments/01_selector_generator_bridge_2026-08-12/F004_RESULTS.md`](experiments/01_selector_generator_bridge_2026-08-12/F004_RESULTS.md) | G0/G1/G2、TopK 对照、引用质量、诊断与停止决定 | COMPLETE-NO-GATE |
| [`experiments/01_selector_generator_bridge_2026-08-12/F006_TRAINING_REPORT.md`](experiments/01_selector_generator_bridge_2026-08-12/F006_TRAINING_REPORT.md) | 小规模 clean/mixed-context LoRA 训练记录 | COMPLETE |
| [`experiments/01_selector_generator_bridge_2026-08-12/F006_RESULTS.md`](experiments/01_selector_generator_bridge_2026-08-12/F006_RESULTS.md) | Mixed-LoRA 开发集选择结果 | COMPLETE-DEV-PASS |
| [`experiments/01_selector_generator_bridge_2026-08-12/F005_SELECTION_REPORT.md`](experiments/01_selector_generator_bridge_2026-08-12/F005_SELECTION_REPORT.md) | 独立集运行前冻结选择记录 | COMPLETE |
| [`experiments/01_selector_generator_bridge_2026-08-12/F005_RESULTS.md`](experiments/01_selector_generator_bridge_2026-08-12/F005_RESULTS.md) | 独立 600 题最终答案、引用与失败归因 | COMPLETE-FINAL-NO-WIN |
| [`experiments/02_generator_selector_alignment_2026-08-15/PLAN.md`](experiments/02_generator_selector_alignment_2026-08-15/PLAN.md) | Generator-aware Selector 对齐总路线与数据/held-out 边界 | G230-complete-no-candidate / S300-blocked |
| [`experiments/02_generator_selector_alignment_2026-08-15/TRACKER.md`](experiments/02_generator_selector_alignment_2026-08-15/TRACKER.md) | A000–G230 实际执行状态及后续条件阶段 | current |
| [`experiments/02_generator_selector_alignment_2026-08-15/G230_RESULTS.md`](experiments/02_generator_selector_alignment_2026-08-15/G230_RESULTS.md) | draft LoRA answer/empty 改善与 citation gate 失败 | complete-no-candidate |
| [`experiments/03_generator_grounding_repair_2026-08-18/README.md`](experiments/03_generator_grounding_repair_2026-08-18/README.md) | 第 03 路线零基础入口、旧 Selector 外推边界与协同顺序 | revised-plan |
| [`experiments/03_generator_grounding_repair_2026-08-18/PLAN.md`](experiments/03_generator_grounding_repair_2026-08-18/PLAN.md) | Generator 修复、冻结教师、Utility Selector 和完整系统分阶段计划 | revised-draft-for-review |
| [`experiments/03_generator_grounding_repair_2026-08-18/TRACKER.md`](experiments/03_generator_grounding_repair_2026-08-18/TRACKER.md) | G000–H100 条件执行表 | revised-planned |
| [`experiments/03_generator_grounding_repair_2026-08-18/snapshots/PLAN_v1_generator_only_2026-08-18.md`](experiments/03_generator_grounding_repair_2026-08-18/snapshots/PLAN_v1_generator_only_2026-08-18.md) | 修订前的 Generator-only 严格资格门方案 | superseded-snapshot |

## 更新规则

1. 每条独立跨阶段实验使用一个单独文件夹。
2. 计划有实质变化时，在该实验文件夹中新增带版本的快照，不覆盖旧版本。
3. `PLAN.md` 始终指向当前执行版本；`TRACKER.md` 记录实际状态，不预填结果。
4. Retriever、Selector、Generator 各自的模块实验继续保留在原目录，不迁入此处。
