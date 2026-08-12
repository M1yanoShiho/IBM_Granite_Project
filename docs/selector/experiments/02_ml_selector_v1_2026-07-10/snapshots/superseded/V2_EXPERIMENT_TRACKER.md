# Graph-Assisted Evidence Selector 2.0 — Experiment Tracker

**Protocol:** zero-new-human-annotation；三类核心关系；FinanceBench=`EXPOSED_DIAGNOSTIC_ONLY` 且完全排除

**Plan:** [V2_EXPERIMENT_PLAN.md](V2_EXPERIMENT_PLAN.md)

**Current state:** V2 尚未开始；所有 Run 均为 `TODO`

## 使用规则

- 正式运行前先填写 Git commit、seed、host/job 和 output package；运行完成后填写 Gate/metric 与 interpretation。
- `outputs` 必须指向团队可访问的精简结果包，不能只写私有服务器路径。
- 状态只使用 `TODO`、`RUNNING`、`DONE`、`FAIL`、`STOPPED`。
- 每次运行的精简结果包至少包含 `run_manifest.json`、aggregate metrics、per-query metrics、统计/Gate 结果和必要 hash。
- FinanceBench 不得出现在 V2 的 command、input、调参、失败分析或结果包中。

## Run table

| Run | M | 工作与系统 | 数据 / split | 优先级 | 状态 | Git commit | Seed | Host / Job | Outputs | Gate / metric | Interpretation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R000 | M0 | 冻结 protocol/schema/splits | 全部 | MUST | TODO | — | — | — | — | hash / leakage audit | — |
| R001 | M0 | 自动 label provenance Gate | NIAH train/dev/fresh manifest | MUST | TODO | — | deterministic | — | — | violations / overlap | — |
| R002 | M0 | 冻结 NIAH train 500 或 2000 | NIAH train | MUST | TODO | — | — | — | — | query count / hash | dev 前决定 |
| R003 | M0 | paired Monte Carlo 样本量 sensitivity | NIAH dev | MUST | TODO | — | frozen in run | — | — | power / MDE | query 为统计单位 |
| R010 | M1 | ContractNLI adapter sanity | official train/dev | MUST | TODO | — | — | — | — | counts / leakage | 同合同不跨 split |
| R011 | M1 | VitaminC adapter 与 revision-family decontamination | decontaminated train/dev + official test | MUST | TODO | — | — | — | — | counts / removed families | official test 不移动 |
| R012 | M1 | pretrained NLI Relation Builder baseline | ContractNLI/VitaminC dev | MUST | TODO | — | model default | — | — | macro-F1 / precision / coverage | — |
| R013 | M1 | fine-tuned Relation Builder | ContractNLI/VitaminC train/dev | MUST | TODO | — | 13 | — | — | per-class F1 | — |
| R014 | M1 | fine-tuned Relation Builder | ContractNLI/VitaminC train/dev | MUST | TODO | — | 42 | — | — | per-class F1 | — |
| R015 | M1 | fine-tuned Relation Builder | ContractNLI/VitaminC train/dev | MUST | TODO | — | 73 | — | — | per-class F1 | — |
| R016 | M1 | claim→cluster→relation→graph 全链路 OOF | NIAH train | MUST | TODO | — | 13/42/73 | — | — | OOF coverage / hash | 按父页面和 synthetic family 分组 |
| R020 | M2 | official relation Gate | ContractNLI/VitaminC official test | MUST | TODO | — | ensemble | — | — | precision / macro-F1 / coverage | test 只运行一次 |
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
