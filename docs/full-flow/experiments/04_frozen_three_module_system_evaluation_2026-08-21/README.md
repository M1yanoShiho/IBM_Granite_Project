# 冻结三模块 RAG：总体系统实验

**日期：** 2026-08-22  
**状态：** `v4 总路线已冻结 / Goal 1–5 COMPLETE / FINAL PASS`

## 一句话说明

最终实验内容没有变：用三套数据比较四个 baseline 与 `Ours`，再做三个单模块消融。执行被拆成五个防偏移目标；每个目标独立 PASS/FAIL，PASS 后自动接力，FAIL 留在当前目标修复。

## 总路线

| Goal | 做什么 | 完成后得到什么 |
|---:|---|---|
| 1 | 准备数据与评分器 | 数据无泄漏、候选池正确、五指标可用 |
| 2 | 接通五个系统并做开发集检查 | 10 个正式运行臂全部可运行 |
| 3 | 比较四个 baseline 与 Ours | Table 1，7,700 个答案 |
| 4 | 做 Retriever、Selector、Generator 消融 | Table 2，3,300 个新答案 |
| 5 | 统一统计和最终核验 | 可直接放入报告的两张英文表 |

每次只设一个 Goal 为 active。Goal 内部可以连续推进；Goal PASS 后冻结交付物并立即把下一 Goal 设为 active。

## 冻结实验矩阵

| 项目 | 内容 |
|---|---|
| 三套数据 | HotpotQA 400、MuSiQue answerable 400、RGB noise 300 |
| 五个系统 | Dense RAG、Hybrid RAG、Granite Rerank RAG、Provence RAG、Ours |
| 三个消融 | w/ Dense Retriever、w/ Top-10、w/ Direct Generator |
| 五个指标 | Ret.、Sel.、Ans.、Cit.、RAR |
| 两张表 | 完整系统主表、模块消融表 |

Ours 已有三个独立训练的 Generator checkpoint（seeds 13/42/73）。主表的答案类结果报告 mean ± sample SD；四个确定性 baseline 各跑一次。`Full`、Retriever/Selector 消融使用 GR-C seed 13；Direct Generator 消融使用 frozen base Granite。

## 现在处于哪里

Goal 1–4 均已于 2026-08-22 `PASS`。Goal 3 的 fresh attempt D 三个 jobs
`18674672`、`18674673`、`18674674` 产生 7,700 条正式输出并冻结 Table 1；结果不
支持 Ours 的 RAR 优越性。Goal 4 的三个 jobs `18674854`、`18674855`、`18674856`
又产生精确 3,300 条单模块消融答案，复用 Goal 3 Full seed13 而未重新生成，Table 2、
9 个 paired CI 与逐题/hash 审计全部 PASS。消融显示 Retriever/Selector 替换不改变
RAR，而 Direct Generator 在 HotpotQA 和 MuSiQue 上优于 Full。Goal 5 没有生成新
答案，已精确复算 mean/sample SD、21 个 paired CI、共同分母与所有来源 hash，并生成
统一 CSV/JSON、Markdown/LaTeX 两表和最终报告。五个目标现均完成并 `FINAL PASS`。
详见 [final report](FINAL_REPORT.md)、
[Goal 3 results](reports/GOAL3_MAIN_SYSTEM_RESULTS.md) 和
[Goal 4 results](reports/GOAL4_MODULE_ABLATION_RESULTS.md)。

- 完整路线：[PLAN.md](PLAN.md)
- 当前状态：[TRACKER.md](TRACKER.md)
- Goal 1 交接：[GOAL_1_HANDOFF.md](GOAL_1_HANDOFF.md)
- Goal 1 实际报告：[GOAL1_DATA_SCORER_READINESS.md](reports/GOAL1_DATA_SCORER_READINESS.md)
- Goal 2 实际报告：[GOAL2_SYSTEM_WIRING_READINESS.md](reports/GOAL2_SYSTEM_WIRING_READINESS.md)
- Goal 3 实际报告：[GOAL3_MAIN_SYSTEM_RESULTS.md](reports/GOAL3_MAIN_SYSTEM_RESULTS.md)
- Goal 4 实际报告：[GOAL4_MODULE_ABLATION_RESULTS.md](reports/GOAL4_MODULE_ABLATION_RESULTS.md)
- 最终报告：[FINAL_REPORT.md](FINAL_REPORT.md)
- 报告表格：[RESULT_TABLES.md](RESULT_TABLES.md)
- 上一版计划：[v3 snapshot](snapshots/PLAN_v3_2026-08-22.md)

若两张 A4000 可持续使用、依赖顺利且无需整包重跑，整体目标约 2–3 个自然日；安全排期为 3–4 个自然日。
