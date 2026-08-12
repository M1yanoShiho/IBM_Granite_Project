# Full-flow 文档清单

**更新时间：** 2026-08-12

| 文件 | 用途 | 状态 |
|---|---|---|
| [`README.md`](README.md) | Retriever、Selector、Generator 与 full-flow 文档的目录分工 | current |
| [`experiments/README.md`](experiments/README.md) | 完整证据流实验总索引 | current |
| [`experiments/01_selector_generator_bridge_2026-08-12/README.md`](experiments/01_selector_generator_bridge_2026-08-12/README.md) | 第 01 路线零基础入口 | F000-pass-F001-ready |
| [`experiments/01_selector_generator_bridge_2026-08-12/PLAN.md`](experiments/01_selector_generator_bridge_2026-08-12/PLAN.md) | 三阶段实验与改造计划 v2 | current-plan-running |
| [`experiments/01_selector_generator_bridge_2026-08-12/AMENDMENT_RUNTIME_SIGNAL_BOUNDARY_v2.md`](experiments/01_selector_generator_bridge_2026-08-12/AMENDMENT_RUNTIME_SIGNAL_BOUNDARY_v2.md) | Selector 信号、Generator 运行时估计与 gold 评测标签的严格边界 | current-v2-amendment-not-run |
| [`experiments/01_selector_generator_bridge_2026-08-12/TRACKER.md`](experiments/01_selector_generator_bridge_2026-08-12/TRACKER.md) | F000–F006 执行跟踪表 | F000-pass |
| [`experiments/01_selector_generator_bridge_2026-08-12/EVIDENCE_INDEX.md`](experiments/01_selector_generator_bridge_2026-08-12/EVIDENCE_INDEX.md) | 已有上游证据与未来产物入口 | current |
| [`experiments/01_selector_generator_bridge_2026-08-12/F000_IMPLEMENTATION_REPORT.md`](experiments/01_selector_generator_bridge_2026-08-12/F000_IMPLEMENTATION_REPORT.md) | 联合输入、四臂入口与运行时 gold 隔离核对 | PASS |

## 更新规则

1. 每条独立跨阶段实验使用一个单独文件夹。
2. 计划有实质变化时，在该实验文件夹中新增带版本的快照，不覆盖旧版本。
3. `PLAN.md` 始终指向当前执行版本；`TRACKER.md` 记录实际状态，不预填结果。
4. Retriever、Selector、Generator 各自的模块实验继续保留在原目录，不迁入此处。
