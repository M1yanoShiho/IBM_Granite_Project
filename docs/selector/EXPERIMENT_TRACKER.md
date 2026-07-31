# Graph-Assisted Evidence Selector 2.0 — Experiment Tracker

**Protocol:** `g2-proto-1`（[M0_PROTOCOL_FREEZE.md](M0_PROTOCOL_FREEZE.md)，状态 DRAFT — 待 R001 的 δ 后转 FROZEN）

**Plan:** [TRAINING_PLAN.md](TRAINING_PLAN.md) + [Graph 2.0 设计](../superpowers/specs/2026-07-30-graph-2.0-relation-layer-design.md) + [Plan 1](../superpowers/plans/2026-07-30-graph-2.0-plan-1-foundation.md)

**Current state:** Plan 1 代码全部落地。已完成的纯 CPU Run：**R011**（VitaminC 去污染）、**R001b(a)**（parent 碰撞率 0.931）。等待中：R001（GPU，job 18225682 排队）、R011b（等 `runs/niah-train` 物化）、R012（等 R011b）。

**D1 = A**（只升级边与票的构造，门的四条件不动）。**D2 已消解**（D1=A 下选择器无参数，零训练路线下关系模型也不训）。

## 使用规则

- 正式运行前先填写 Git commit、seed、host/job 和 output package；运行完成后填写 Gate/metric 与 interpretation。
- `outputs` 必须指向团队可访问的精简结果包，不能只写私有服务器路径。
- 状态只使用 `TODO`、`RUNNING`、`DONE`、`FAIL`、`STOPPED`。
- 每次运行的精简结果包至少包含 `run_manifest.json`、aggregate metrics、per-query metrics、统计/Gate 结果和必要 hash。
- FinanceBench 不得出现在 V2 的 command、input、调参、失败分析或结果包中。

## Run table

| Run | M | 工作与系统 | 数据 / split | 优先级 | 状态 | Git commit | Seed | Host / Job | Outputs | Gate / metric | Interpretation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R000 | M0 | 冻结 protocol/schema/splits（`g2-proto-1`） | 全部 | MUST | RUNNING | ddb5342 | — | — | [M0_PROTOCOL_FREEZE.md](M0_PROTOCOL_FREEZE.md) | hash / leakage audit | 决定全部拍板；状态 DRAFT，等 R001 的 δ 才能转 FROZEN |
| R001 | M0 | **G-FC 固定分母基线**：带 `--dump` 跑一轮 3B+single E1，再离线 `cluster_rescore_cli` | `runs/niah-injected` | MUST | TODO | 10289bc | deterministic | — | — | 基线 false_conflict + δ | **关键路径**：工具已就位，只差一次 GPU 轮次 |
| R001b | M0 | parent 碰撞率 **(a) DONE** + Graph 1.0-lenient 的 `support_unit=parent` 基线臂 **(b) TODO** | `runs/niah-injected` | MUST | **PARTIAL** | d6607b3 / 2b946a0 | 13 | bp1 login (a) | (a) stdout，见 hpc-run-log | (a) **collision_rate 0.931**（1862/2000）；20.0 doc → 16.47 parent；unresolved 0 | (a) 缺陷普遍存在，非边角；(b) **不作方向性主张** —— 原预注册（门 fire 更少 ⇒ harm↑recall↑）只考虑了条件 3，漏了条件 4 使 `own≤cap` 更易满足，已于运行前修正 |
| R002 | M0 | ~~冻结 NIAH train 500 或 2000~~ | — | — | **N/A** | — | — | — | — | — | 已消解：D1=A + 零训练 ⇒ 无 Selector 训练 |
| R003 | M0 | paired Monte Carlo 样本量 sensitivity + G-FC 的 MDE | NIAH dev | MUST | TODO | — | frozen in run | — | — | power / MDE | 依赖 R001；δ = max(0.05, MDE) |
| R010 | M1 | ~~ContractNLI adapter sanity~~ | — | — | **DROPPED** | — | — | — | — | — | 17 假设×607 NDA，与任务无结构相似性；v1 迁移 −0.134。理由见 M0 §3.0 |
| R011 | M1 | VitaminC adapter 与 revision-family decontamination | official test（+去污染 train/dev） | MUST | **DONE** | b5dbb5d | deterministic | bp1 login | `data/gate0b/vitaminc_decontamination.json` | test 55197 **未动**；train −810 行/38 page；dev −70 行/2 page | official split 确有跨 split family 重叠（量小但非零），去污染非形式主义；test-preserving 满足 |
| R011b | M1 | Gate 0B-2 探针构造（mutation log → 四类确定性对） | NIAH **train** split | MUST | TODO | ddb5342 | deterministic | — | — | n_pairs / n_skipped | 必须用 train，不得用 dev |
| R012 | M1 | **零训练 Relation Builder sweep（三臂同场）** | VitaminC official test + 0B-2 探针 | MUST | TODO | be9dd2c | model default | — | — | 0B-1 五项 + 0B-2 两项 | **真实分叉点**：过则接门，不过则启动训练路径 |
| R013 | M1 | fine-tuned Relation Builder | 去污染 VitaminC train/dev + NIAH-train 域适配 | CONDITIONAL | TODO | — | 13 | — | — | per-class F1 | 仅当 R012 未过才启动（M0 §3.8） |
| R014 | M1 | fine-tuned Relation Builder | 同上 | CONDITIONAL | TODO | — | 42 | — | — | per-class F1 | 同上；训练路径启动时三 seed 条款恢复生效 |
| R015 | M1 | fine-tuned Relation Builder | 同上 | CONDITIONAL | TODO | — | 73 | — | — | per-class F1 | 同上 |
| R016 | M1 | claim→cluster→relation→graph 全链路 OOF | NIAH train | MUST | TODO | — | 13/42/73 | — | — | OOF coverage / hash | 按父页面和 synthetic family 分组 |
| R020 | M2 | official relation Gate | VitaminC official test（ContractNLI 已移出，见 M0 §3.0） | MUST | TODO | — | ensemble | — | — | precision / macro-F1 / coverage | test 只运行一次 |
| R021 | M2 | SAME_SOURCE 与 metamorphic tests | synthetic unit fixtures | MUST | TODO | — | deterministic | — | — | exact pass rate | 目标 100% |
| R022 | M2 | 生成 train Graph | NIAH train | MUST | TODO | — | OOF ensemble | — | — | cache / hash audit | 禁止 gold graph features |
| R023 | M2 | 生成 dev Graph | NIAH dev | MUST | TODO | — | frozen ensemble | — | — | cache / hash audit | 不读取 utility/harm/gold answer |
| R030 | M3 | 同协议复现 q2d/fixed/v1/oracle | NIAH dev | MUST | TODO | — | frozen set | — | — | v1 parity / metrics | v1/v2 同数据与预算 |
| R031 | M3 | NLI 无图对照 | NIAH dev | MUST | TODO | — | 13/42/73 | — | — | harmful / recall / NDCG | 排除“仅新增 NLI”解释 |
| R032 | M3 | Graph-only 规则对照 | NIAH dev | MUST | TODO | — | deterministic | — | — | harmful / recall / NDCG | 隔离图规则价值 |
| R033 | M3 | Graph+ML | NIAH dev | MUST | TODO | — | 13 | — | — | harmful / recall / NDCG | — |
| R034 | M3 | Graph+ML | NIAH dev | MUST | TODO | — | 42 | — | — | harmful / recall / NDCG | — |
| R035 | M3 | Graph+ML | NIAH dev | MUST | TODO | — | 73 | — | — | harmful / recall / NDCG | — |
| R036 | M3 | CLAIM_REFUTES 消融 | NIAH dev | MUST | TODO | — | 13/42/73 | — | — | delta vs full | — |
| R037 | M3 | SAME_SOURCE 消融 | NIAH dev | MUST | TODO | — | 13/42/73 | — | — | delta vs full | — |
| R038 | M3 | shuffled-Graph 负控 | NIAH dev | MUST | TODO | — | frozen set | — | — | delta vs v1 | 不应复现 Graph 收益 |
| R039 | M3 | 冻结 primary ensemble | NIAH dev | MUST | TODO | — | 13/42/73 | — | — | config / hash | fresh test 不选 seed |
| R040 | M4 | 冻结 code/model/config/hash | fresh NIAH 600 manifest | MUST | TODO | — | ensemble | — | — | audit | 冻结后不调参 |
| R050 | M5 | 一次性主测试：全部冻结系统 | fresh NIAH sealed 600 | MUST | TODO | — | frozen | — | — | harmful / recall / NDCG | 一次运行全部 baseline |
| R051 | M5 | Graph 2.0 vs v1 配对确认统计 | fresh NIAH per-query | MUST | TODO | — | statistic seed | — | — | CI / p / non-inferiority | grouped bootstrap + randomization |
| R052 | M5 | 次级比较和消融统计 | fresh NIAH per-query | MUST | TODO | — | statistic seed | — | — | Holm-adjusted p | 不改变 primary Gate |
| R060 | M6 | RAMDocs official 外部方向验证 | official candidate pool→Top-3 | MUST | TODO | — | frozen | — | — | recall / misinfo / NDCG / coverage | 不调参 |
| R061 | M6 | RAMDocs adapted secondary | unjudged mined passages→Top-20 | OPTIONAL | TODO | — | frozen | — | — | judged-only diagnostics | 不进入主结论 |
| R070 | M7 | selector downstream transmission check | fresh NIAH frozen subset | CONDITIONAL | TODO | — | frozen | — | — | F1 / cover-EM | 仅 C1 通过后 |
| R071 | M7 | citation proxy | 同一 frozen subset/generator | OPTIONAL | TODO | — | frozen | — | — | citation proxy | 自动 judge 明确标 proxy |
| R080 | M7 | 汇总主表、图和审计 | all certified outputs | MUST | TODO | — | — | — | — | report completeness | 不包含 FinanceBench 结果 |

## 状态更新示例

完成一次运行后，把对应行更新为类似：

```text
DONE | 4f23abc | 13 | BluePebble / 12345678 |
docs/data/graph_selector_validation/R033/ |
Harmful@10=..., Recall@10=..., NDCG@10=... |
Graph 2.0 improved recall but did/did not pass the harmful-rate Gate.
```

不要只写“运行成功”；interpretation 必须说明该 Run 对研究判断产生了什么影响。
