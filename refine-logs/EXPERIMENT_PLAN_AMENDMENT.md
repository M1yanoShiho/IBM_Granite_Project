# 当前 Selector 实验计划修订入口

**当前版本：** [`EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB_v3_LEAN.md`](EXPERIMENT_PLAN_AMENDMENT_2026-08-12_R005AB_v3_LEAN.md)

**状态：** `USER-APPROVED LEAN SCOPE / L000 PASS / FROZEN FOR IMPLEMENTATION / NOT RUN`

当前 v3 只保留必要的数据隔离、最终盲测、阈值冻结和成对统计。方法为固定 TopK10 内的 NLI-aware protect/harm 评分与 0–cap2 保守删除；`NLI-pair` 只在 `NLI-base` 开发失败时启用。本轮 final 使用 R002 已隔离的 `decision-dev`，sealed/official heldout 保留不读。

它不会覆盖原冻结主计划或旧 R005 `FAIL`。R005A/B v1 与 v2 都从未执行，现永久保留为 superseded 历史；本固定文件只作为 Lean v3 最新入口。
