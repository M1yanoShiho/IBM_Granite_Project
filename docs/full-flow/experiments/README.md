# 完整证据流实验索引

| # | 实验 | 状态 | 研究问题 |
|---:|---|---|---|
| 01 | [Selector–Generator 跨阶段桥接](01_selector_generator_bridge_2026-08-12/README.md) | `COMPLETE / F005 FINAL-NO-WIN` | 已完成的 Selector 与高级 Generator 联合后发生什么，以及关键事实读取能否把局部改善传到最终答案 |
| 02 | [Generator–Selector 目标对齐](02_generator_selector_alignment_2026-08-15/PLAN.md) | `G230 COMPLETE / NO CANDIDATE` | 训练真实 draft 路径并检验是否能产生可靠 Generator 教师；当前因 citation 退化未解锁 Utility Selector |
| 03 | [Generator 证据使用与引用修复](03_generator_grounding_repair_2026-08-18/README.md) | `PLANNED / NO NEW RUN AUTHORIZED` | 保留 G230 answer/empty 改善，同时修复 context robustness 与事实引用对应，并进行跨数据集 Generator 资格审查 |

每一次独立路线使用一个文件夹，避免把计划、改造和结果混在其他模块的历史记录中。
