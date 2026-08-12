# F000 联合实验入口冻结报告

**日期：** 2026-08-12
**状态：** `PASS / F001 READY`

## 零基础结论

F000 已经完成。它确认了下一步四组实验会使用同一批题和同一批 Retriever 候选证据，不需要重新运行 Retriever，也不需要重新训练或运行 Selector。

真正需要重新运行的只有 Generator，因为四组答案必须在同一个运行过程中产生，不能把旧实验答案和新实验答案跨作业拼接。

## 已冻结的四组

| 组 | 证据 | Generator |
|---|---|---|
| A | TopK10 | 基础 Granite |
| B | Lean v3 Selector 的保留结果 | 基础 Granite |
| C | TopK10 | 现有 Verify-and-annotate |
| D | Lean v3 Selector 的保留结果 | 现有 Verify-and-annotate |

## 真实输入核对

服务器使用冻结文件重新拼装后得到：

| 项目 | 数量 |
|---|---:|
| NIAH `decision-dev` 问题 | 739 |
| Selector 改变证据的问题 | 109 |
| Selector 没有改变证据的问题 | 630 |

这些数量与 L003 完全一致。冻结输入总指纹为：

`bad751bde256d01e204cbae85de2d43ce8e865689b4a08a9fb83500843ea3805`

## 运行时信息边界

F001 的 `run` 命令没有 gold/reference answer 参数，只读取：

- 问题文本；
- 冻结 TopK10 候选证据；
- 预先划定的 `decision-dev` 题目 ID；
- L003 已保存的 Selector 保留证据 ID。

标准答案只由生成结束后的 `score` 命令读取。因此 Generator 运行时不能偷看 reference answer、official required evidence 或 gold chain。

## 最小实现

- `scripts/full_flow_joint.py`：`prepare / run / score` 三个入口；
- `scripts/run_full_flow_f001_server.sh`：服务器完整运行入口；
- `tests/full_flow/test_full_flow_joint.py`：四臂复用、交互计算、TopK 边界和 gold 隔离测试。

## 已完成验证

- 33 个相关本地测试通过；
- ruff 通过；
- mypy 通过；
- shell 语法和 diff 检查通过；
- 服务器真实 F000 输入审计通过。

## 下一步

启动 F001 四臂生成。F001 完成前不实现 `SelectionGuidance`、关键事实笔记或 `EvidenceReadiness`。
