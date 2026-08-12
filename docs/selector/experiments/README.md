# Selector 历次实验总索引

这个目录解决一个实际问题：旧计划曾被新计划从原路径移除，只看当前文件夹会误以为以前没有写过计划。现在按“一次独立路线一个文件夹”恢复，并清楚区分真正运行过的实验、未运行草案和当前计划。

[机器可读总清单](MANIFEST.json)记录八条路线、当前路线和授权边界。

## 时间线

| # | 路线 | 最终状态 | 最重要的结论 |
|---:|---|---|---|
| 01 | [Corroboration reranking](01_corroboration_reranking_2026-07-05/README.md) | `COMPLETED / LIMITED POSITIVE SIGNAL / NOT A PRODUCTION SELECTOR` | nested-CV 的 needle-found@10 约提升 3.7pp，但 MRR 基本不变；没有证明能安全删除 harmful evidence |
| 02 | [ML Selector V1](02_ml_selector_v1_2026-07-10/README.md) | `COMPLETE / GATE FAIL / STOPPED` | 排序/召回部分改善，但 harmful 过滤与跨域门失败；完整 RAG 按协议未运行 |
| 03 | [Gated Corroboration / Graph 1.0](03_gated_graph1_2026-07-20/README.md) | `COMPLETE / SAFETY GATE FAIL / RETIRED` | harmful 明显下降，但 required recall 下降 4.8pp；lenient 只回收约 1.2pp，仍不安全 |
| 04 | [Graph 2.0 relation layer](04_graph2_relation_layer_2026-07-30/README.md) | `PARTIAL / NO SELECTOR-LEVEL PASS / RETIRED` | 零训练 Gate 失败；seed13 训练虽收敛，真实 Top20 probe 却产生约 13.34 个 false conflicts/题，未进入 sealed C1 |
| 05 | [Reliability-MIS](05_reliability_mis_2026-08-08/README.md) | `COMPLETE / FAIL / RETIRED` | 能删 harmful，但 required recall 下降 37.17pp、2Wiki supporting recall 下降 11.19pp |
| 06 | [三分类 Beam Selector](06_beam_selector_2026-08-09/README.md) | `COMPLETE / M2 FAIL / RETIRED` | M0/M1 PASS；M2 的 16 组阈值全部超过 recall 损失上限，M3/M4 取消 |
| 07 | [Adaptive Conservative R001–R005](07_adaptive_conservative_r001_r005_2026-08-11/README.md) | `R001–R004 PASS / R005 TRAINING-GATE FAIL / R006–R015 CUT` | 新的风险控制路线完成基础设施和基线，但双头 scorer 未通过训练资格门 |
| 08 | [R005A/R005B recovery v2](08_r005ab_repair_current_2026-08-12/README.md) | `DRAFT / WAITING A000 EXPLICIT APPROVAL / NOT IMPLEMENTED / NOT RUN` | 当前方案；相对 amendment v1 只修执行协议，相对旧 R005 同时包含方法层升级 |

## 怎样理解“八次”

这里按独立研究问题和方法路线划分，不按 commit、seed 或 run ID 计数。例如：

- R001–R005 是第 07 路线中的阶段，不是五个彼此无关的计划；
- R005A/B amendment v1 与 document revision v2 属于同一第 08 路线，v1 从未运行；
- Graph 2.0 的 A1–A4、R012b–f 是同一路线内的协议修订与诊断；
- class-weight 修正、服务器复验或文档纠错也不算新实验。

## 每个文件夹的固定结构

| 文件 | 作用 |
|---|---|
| `README.md` | 零基础摘要、实际状态、停止原因和与前后路线的关系 |
| `PLAN.md` | 本次计划的导航页，指出最终计划快照及其权威来源 |
| `TRACKER.md` | 当时 tracker 或事后状态对照；会明确二者性质 |
| `EVIDENCE_INDEX.md` | 结果、报告、审计和大文件所在位置 |
| `SOURCE_MANIFEST.json` | 来源 commit/path/blob 与快照完整性机器清单 |
| `snapshots/` | 从 Git 或 canonical 文件恢复的原文快照；不能直接作为新执行入口 |

`snapshots/` 保持原始字节不变，因此其中的旧相对链接仍按当时原路径书写，搬到归档目录后不保证可点击；请从同目录的 `PLAN.md` 或 `EVIDENCE_INDEX.md` 导航。每份快照是否与原 Git blob/本地原件一致，以 `SOURCE_MANIFEST.json` 的 `integrity` 字段为准。

## 状态词

- `COMPLETE/COMPLETED`：这条路线或指定阶段的记录已经闭合；不表示所有阶段都运行，更不表示方法成功。
- `PASS`：某一个预先规定的门通过；局部门通过不等于整个 Selector 可以上线。
- `PARTIAL`：只完成了部分阶段或 seed，没有形成完整的正式结论。
- `FAIL`：执行过，并未达到预先规定的门。
- `CUT/STOPPED/RETIRED`：因为上游门失败或路线结束，后续阶段没有运行。
- `SUPERSEDED`：文档草案被新版本取代，不表示实验失败。
- `NOT RUN`：没有运行，不能把它描述成正结果或负结果。
