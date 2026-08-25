# 完整证据流实验索引

| # | 实验 | 状态 | 研究问题 |
|---:|---|---|---|
| 01 | [Selector–Generator 跨阶段桥接](01_selector_generator_bridge_2026-08-12/README.md) | `COMPLETE / F005 FINAL-NO-WIN` | 已完成的 Selector 与高级 Generator 联合后发生什么，以及关键事实读取能否把局部改善传到最终答案 |
| 02 | [Generator–Selector 目标对齐](02_generator_selector_alignment_2026-08-15/PLAN.md) | `G230 COMPLETE / NO CANDIDATE` | 训练真实 draft 路径并检验是否能产生可靠 Generator 教师；当前因 citation 退化未解锁 Utility Selector |
| 03 | [Generator 修复与 Selector 分阶段协同](03_generator_grounding_repair_2026-08-18/README.md) | `REVISED PLAN / NO RUN AUTHORIZED` | 先冻结职责合格 Generator 教师，再训练 Generator-aware Utility Selector，最后验证完整三模块系统；模块资格与强统计结论分开 |
| 04 | [冻结三模块系统完整评估](04_frozen_three_module_system_evaluation_2026-08-21/README.md) | `V4 / GOAL 2 PASS / GOAL 3 ACTIVE / CHECKPOINT RESTORE` | Goal 3 runner/评分与公开模型已就绪；冻结自定义 checkpoint 待恢复，尚未生成或评分 held-out |

每一次独立路线使用一个文件夹，避免把计划、改造和结果混在其他模块的历史记录中。
