# F005 独立确认：Selector 阶段

**日期：** 2026-08-13
**状态：** `SELECTION COMPLETE / GENERATION NOT RUN`

## 零基础说明

F006 已通过开发门后，我们才解封独立的 sealed600 数据。本阶段只让已经冻结的 Lean Selector 查看问题和当前 TopK10 证据，并保存它决定删除什么；没有读取参考答案、gold 文档标签或 counterfactual 来源记录。

使用的方法没有重新选择：

- Seed-13 NLI-aware 双头 checkpoint：`86622bd9...2bf`；
- 安全阈值：`0.9212157130241394`；
- 每题最多删除 2 条，但允许删除 0 条；
- 候选池与 development 使用同一个 frozen hybrid Retriever 配置；
- sealed600 与 F006 train / validation / development 的 query ID 重叠均为 0。

## 实际结果（尚未读取 gold）

| 项目 | 数量 |
|---|---:|
| 独立问题 | 600 |
| Selector 改变的问题 | 81 |
| 删除的证据总数 | 82 |
| 未改变的问题 | 519 |

这一步只能说明 Selector 在独立数据上确实产生了足够多的动作，不能说明删得是否正确。证据正确性和最终答案结果只能在三组 Generator 全部完成后统一评分。

机器清单：[`artifacts/F005/selection/selection_manifest.json`](artifacts/F005/selection/selection_manifest.json)；逐题记录：[`artifacts/F005/selection/selection_trace.jsonl`](artifacts/F005/selection/selection_trace.jsonl)。
