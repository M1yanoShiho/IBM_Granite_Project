# A002 基线复现与运行方差协议

**日期：** 2026-08-15
**状态：** `COMPLETE / PASS`
**结果：** [A002_RESULTS.md](A002_RESULTS.md)

## 1. 目的

在已揭示的 NIAH decision-dev 739 题上，使用 A001 trace 复现当前 Generator 基线，并测量同输入在同一进程内的生成波动。A002 不训练模型、不选择新方法，也不产生最终 held-out 结论。

## 2. 冻结实验臂

| Arm | Context | Generator |
|---|---|---|
| `K_topk_base` | Retriever TopK10 | 当前 Granite 4.1-3B Base Verify-and-annotate |
| `L_legacy_selector_base` | 冻结 Legacy Selector 输出 | 同一个 Granite/TRUE 实例、同一 prompt/decode |

两臂对每题都实际运行。即使 Selector 没有改变 evidence，也不复用另一个臂的答案，因为 unchanged 题正好提供“相同输入、顺序执行”的波动观测。该比较仍是平行实验，不是正式系统回答两次。

## 3. 顺序与重复

- 主轮次按 `seed=13 + query_id` 的稳定 hash 决定 `K->L` 或 `L->K`；
- 在 changed 与 unchanged 两层内分别确定性抽取 10%，正式 739 题对应 11 + 63 = 74 题；
- 重复轮次使用完全相同 context，并反转主轮次臂顺序；
- 所有调用共享一个 Granite 实例和一个 TRUE 实例，位于同一 Python 进程；
- 每次调用保存 A001 `GeneratorTrace`、最终答案、citation、错误与耗时。

主要报告完整 739 分布；109 changed 题与 630 unchanged 题分层报告。重复审计报告 exact answer/result、answer metric agreement 和 coverage agreement。unchanged 题的 K/L 一致率是额外的同输入顺序波动观测，不解释为 Selector 效应。

## 4. Runtime 与评分边界

`run` 命令只接受 query、candidate pool、role assignments、Selector decision trace 和模型 snapshots；命令行不存在 gold 参数。全部生成完成后，独立 `score` 命令才读取 `gold_cases.jsonl`。

```text
run: runtime inputs -> two context arms + repeats -> generations/traces
score: saved generations + gold -> aggregate paired report
```

Reference answer、官方相关文档和 answer_match 不进入 Generator、Selector 或 trace。错误率任一臂超过 5% 时 run 标为 invalid，不接受聚合结果。

## 5. 固定实现与产物

- Runner：`scripts/full_flow_a002.py`；
- Server wrapper：`scripts/run_full_flow_a002_server.sh`；
- 正式目录：`/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/A002-v1`；
- 产物：`generations.jsonl`、`run_manifest.json`、`report.json`、`REPORT.md`；
- 正式运行前先在独立 `A002-smoke-*` 目录做 1 题接线检查；smoke 不与正式结果合并。
