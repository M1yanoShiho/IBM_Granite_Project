# Graph-Assisted Evidence Selector 2.0 — Experiment Tracker

**Protocol:** `g2-proto-2`（[M0_PROTOCOL_FREEZE.md](M0_PROTOCOL_FREEZE.md)，状态 DRAFT — 待 R001 的 δ 后转 FROZEN）

**版本沿革：** `g2-proto-1` → `g2-proto-2`（2026-08-03，修订案 A1：关系判定改二分类支持判断）。A1 是**在 R012 判 FAIL 之后**提出的事后修订，其时间线、利益冲突声明与阈值防护条款见 M0 §9，**批准不消除该披露**。R012 的三类 FAIL 仍是预注册主结果，永久保留。

**Plan:** [TRAINING_PLAN.md](TRAINING_PLAN.md) + [Graph 2.0 设计](../superpowers/specs/2026-07-30-graph-2.0-relation-layer-design.md) + [Plan 1](../superpowers/plans/2026-07-30-graph-2.0-plan-1-foundation.md)

**Current state:** Plan 1 代码全部落地。已完成：**R011**（VitaminC 去污染）、**R001b(a)**（parent 碰撞率 0.931）、**R001**（G-FC 基线 0.4355，job 18225682）、**R012**（Gate 0B，job 18235972，**判定 FAIL**）。

**测量缺陷侧已穷尽（2026-08-04）。** 九项机制全部实测排除或量化：截断、标签序、答案规范化、大小写、premise 质量、premise 长度、hypothesis 形式、argmax 记账、canonical-vs-surface。**DeBERTa 的 `.7942` 自此可作能力读数**，§3.8 在测量这一侧的启动理由已经干净。

**Gate 0B 的分叉点已走完（2026-08-05）。** MiniCheck-FT5 臂（R012c）跑完三个 rung，**九格无一过门**，且它并不优于另两臂 ⇒ **族级断言成立**（"任何零训练模型都不够"现由三臂支撑，含唯一为该任务专训者）、**A1 的出样检验通过**、**§3.8 训练路径解锁**。

**仍未清的一项：0B-1 已挂起**（M0 §9.11），故 Gate 0B 的**外部效度层本轮无证据**，验收只有任务层。解除须先有修订案 **A2**。

**R011b 状态存疑，须补台账。** 探针实际已物化并被 18235972 消费，但**它是用一条与文档不符、且从未入台账的命令建的**：所有文档都写 `runs/niah-train/`，而该路径不可能产出任何一对（探针需要反事实文档与 mutation log，两者只在注入后存在）。2026-08-03 用真实路径 `runs/niah-train-injected/` 重导 rung 2 得 `n_pairs 5888 / n_records 1472 / n_skipped_records 0`，与 sweep 的 `n_task_pairs` 吻合，故 `n_skipped_records = 0` 对 rung 1 同样成立（跳过与 hypothesis 形式无关）。状态保持 TODO 而非 DONE——按本文件规则，没有台账条目的运行不算已记录。**并须确认 `runs/niah-train/` 是否存在**：在此之前，"探针必须建在 train split 上"这条冻结要求无法仅凭文档核验。

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
| R001 | M0 | **G-FC 固定分母基线**：带 `--dump` 跑一轮 3B+single E1，再离线 `cluster_rescore_cli` | `runs/niah-injected` | MUST | **DONE** | 10289bc | greedy（无随机源） | BluePebble / 18225682 | `results/r001-gfc-baseline.json` | **fixed_false_conflict 0.4355**（429/985，CI [.405,.467]） | 基线已得；δ 暂定 .05 待 R003 配对 MDE。附带拿到 rerun-stability：exact 三项与历史 E1 复现到 4 位小数 |
| R001b | M0 | parent 碰撞率 **(a) DONE** + Graph 1.0-lenient 的 `support_unit=parent` 基线臂 **(b) TODO** | `runs/niah-injected` | MUST | **PARTIAL** | d6607b3 / 2b946a0 | 13 | bp1 login (a) | (a) stdout，见 hpc-run-log | (a) **collision_rate 0.931**（1862/2000）；20.0 doc → 16.47 parent；unresolved 0；**needle_parent_inflation 0.328**（415/1264，上界） | (a) 缺陷普遍存在，非边角；(b) **不作方向性主张** —— 原预注册（门 fire 更少 ⇒ harm↑recall↑）只考虑了条件 3，漏了条件 4 使 `own≤cap` 更易满足，已于运行前修正 |
| R002 | M0 | ~~冻结 NIAH train 500 或 2000~~ | — | — | **N/A** | — | — | — | — | — | 已消解：D1=A + 零训练 ⇒ 无 Selector 训练 |
| R003 | M0 | paired Monte Carlo 样本量 sensitivity + G-FC 的 MDE | NIAH dev | MUST | TODO | — | frozen in run | — | — | power / MDE | 依赖 R001；δ = max(0.05, MDE) |
| R010 | M1 | ~~ContractNLI adapter sanity~~ | — | — | **DROPPED** | — | — | — | — | — | 17 假设×607 NDA，与任务无结构相似性；v1 迁移 −0.134。理由见 M0 §3.0 |
| R011 | M1 | VitaminC adapter 与 revision-family decontamination | official test（+去污染 train/dev） | MUST | **DONE** | b5dbb5d | deterministic | bp1 login | `data/gate0b/vitaminc_decontamination.json` | test 55197 **未动**；train −810 行/38 page；dev −70 行/2 page | official split 确有跨 split family 重叠（量小但非零），去污染非形式主义；test-preserving 满足 |
| R011b | M1 | Gate 0B-2 探针构造（mutation log → 四类确定性对） | NIAH **train** split | MUST | TODO | ddb5342 | deterministic | — | — | n_pairs / n_skipped | 必须用 train，不得用 dev |
| R012 | M1 | **零训练 Relation Builder sweep（三臂同场）** | VitaminC official test + 0B-2 探针 | MUST | **DONE** | be9dd2c | model default | bp1 / 18235972 | `results/gate0b/sweep-full.json` | **Gate FAIL**：albert 0B-1 五项全过、0B-2 twin .674 / gold-supports **.192**；DeBERTa 0B-1 挂三项、0B-2 .638 / .794 | **实跑只有两臂**——预注册的 MiniCheck-FT5 需二分类双向打分通路，`load_score_fn` 无此路径故从未上场，**族级断言（"任何零训练模型都不够"）不成立**。~~twin 失败跨臂一致可信~~ **← 已由 R012b rung 2 证伪（albert .7602 越过阈值）：跨臂一致不蕴含能力上限，两臂共用了同一个有缺陷的 hypothesis 形式**。SUPPORTS 塌陷同为形式伪影（干净对照组 `cf_replacement` 弃权实测 64.0%）。详见 [hpc-run-log R012](../hpc-run-log.md) |
| R012b | M1 | hypothesis 形式消融（`template` / `question_answer` / `qa2d` 三级阶梯） | 0B-2 探针（NIAH **train**） | MUST | **DONE** | — | greedy（QA2D 于登录节点预生成到缓存） | bp1 / 18246568、18246569、18257977 | `results/gate0b/sweep-{template,qa,qa2d}.json` + dump | **形式是主因**：albert gold_supports **.1916 → .3635 → .5360**（2.80×，单调）；但**两个指标反向** —— albert twin 峰值在 rung 2（.7602 过阈）、rung 3 崩到 .5312，DeBERTa 两项单调下降。**六格无一两项同时过。** | R012 的归因实验，执行设计 §2.4 的预注册 QA2D 消融并补一级确定性中间形式。**三级都不动 ⇒ 塌陷按能力不足读，启动 §3.8**。R012 的 FAIL 是主结果，不因本轮改写；若 twin 被顶过 .70 只能记作第二次测量 |
| R012c | M1 | **MiniCheck-FT5 臂（§3.1 预注册三臂中从未上场的一臂）** | 0B-2 探针（NIAH **train**），三个 hypothesis rung | MUST | **DONE** | 69a0c04 | deterministic（argmax，无阈值） | bp1 / 18269630-32 | `results/gate0b/sweep-minicheck-{template,qa,qa2d}.json` + dump | 联合门 `gold_supports_recall ≥ .85` ∧ `twin_not_supported_accuracy ≥ .70` | **两条预注册预测均被证伪**：它并未高于另两臂（.5727/.5883/.5360，三 rung 全低于 DeBERTa），rung 排序实测 qa > template > qa2d。**九格无一过门** ⇒ 触发第三条判读规则：**族级断言成立、A1 出样检验通过、§3.8 解锁**。**最有价值的观察**：MiniCheck 对 hypothesis 形式近乎免疫（极差 5.2pp vs albert 34.4 / DeBERTa 20.7）——唯一为该任务专训的臂最不受措辞扰动，独立印证 R012b 的形式伪影结论。**跨任务倒挂**：G 模块 triage 中 MiniCheck 胜 deberta-large，此处相反。（原预注册理由：**本臂同时解锁两个至今做不出的结论**：(1) §3.1 的**族级断言**——R012 只跑两臂，"任何零训练模型都不够"不成立，而它是 §3.8 是否启动的唯一依据；(2) **§9.8 的 A1 出样检验**——A1 由 albert/DeBERTa 促成故二者不能验证它，§9.8 明文"若出样结果与本修订预期相悖，以出样结果为准"。真实阻塞不是"二分类双向探测"（A1 落地后已消失）而是**架构**：`T5ForConditionalGeneration` 且无 `id2label`，故走独立的二分类注册表而非 `LABEL_ORDER`。独立臂报告，不并入 R012b 阶梯（§9.10a 例外）。）详见 [hpc-run-log R012c](../hpc-run-log.md) |
| R012d | M1 | 答案串 canonical vs surface 对照（原题「大小写对照」不准确） | 0B-2 探针（NIAH **train**） | MUST | **DONE** | — | deterministic | bp1 / 18259086 | `results/gate0b/sweep-surface.json` + dump | **预注册方向被证伪**：DeBERTa gold_supports **.7942 → .7792**（McNemar p≈.0007，*下降*）。albert 阴性对照按 ±1pp 通过（+0.68pp）。⇒ 大小写/规范化**不是**那 5.6pp 的来源 | 实测发现 `gold_value` 经 canonicalize 小写化、`replacement_value` 是原始串 ⇒ 两个门指标建在不同大小写分布上，而 DeBERTa 是 cased 且距 .85 仅 5.6pp。**albert `do_lower_case=True`，大小写对它可证不可见，故它移动即实验无效。**本轮结果不改变 Gate 0B 判定（M0 §9.6）；R012c 已由 MiniCheck 臂占用 |
| R013 | M1 | fine-tuned Relation Builder **（配方待裁决，见 M0 §9.12）** | 去污染 VitaminC train/dev + NIAH-train 域适配 | CONDITIONAL | TODO | — | 13 | — | — | per-class F1 | **触发条件已满足**（Gate 0B 九格无一过门，族级断言成立）**但配方未就绪**：§3.8 仍写「三类范式」，而 A1 已改二分类且 §9.7 未提及此处。**训练开始前必须裁决训三类头还是二类头**（M0 §9.12(a)） |
| R014 | M1 | fine-tuned Relation Builder **（配方待裁决，见 M0 §9.12）** | 同上 | CONDITIONAL | TODO | — | 42 | — | — | per-class F1 | 同上；训练路径启动时三 seed 条款恢复生效 |
| R015 | M1 | fine-tuned Relation Builder **（配方待裁决，见 M0 §9.12）** | 同上 | CONDITIONAL | TODO | — | 73 | — | — | per-class F1 | 同上 |
| R016 | M1 | claim→cluster→relation→graph 全链路 OOF | NIAH train | MUST | TODO | — | 13/42/73 | — | — | OOF coverage / hash | 按父页面和 synthetic family 分组 |
| R020 | M2 | official relation Gate | VitaminC official test（ContractNLI 已移出，见 M0 §3.0） | MUST | **BLOCKED** | — | ensemble | — | — | precision / macro-F1 / coverage | **0B-1 已于 2026-08-04 挂起（M0 §9.11）**：A1 使关系模型二分类，而本层五项阈值建在三类分割上——完全正确的模型也会得 refutes_precision 0.0 / macro_f1 0.5 三项自动 FAIL 加一个空洞 PASS。解除阻塞须先有修订案 A2。test 仍只运行一次 |
| R021 | M2 | SAME_SOURCE 与 metamorphic tests | synthetic unit fixtures | MUST | TODO | — | deterministic | — | — | exact pass rate | 目标 100% |
| R022 | M2 | 生成 train Graph | NIAH train | MUST | TODO | — | OOF ensemble | — | — | cache / hash audit | 禁止 gold graph features |
| R023 | M2 | 生成 dev Graph | NIAH dev | MUST | TODO | — | frozen ensemble | — | — | cache / hash audit | 不读取 utility/harm/gold answer |
| R030 | M3 | 同协议复现 q2d/fixed/v1/oracle | NIAH dev | MUST | TODO | — | frozen set | — | — | v1 parity / metrics | v1/v2 同数据与预算 |
| R031 | M3 | NLI 无图对照 | NIAH dev | MUST | TODO | — | 13/42/73 | — | — | harmful / recall / NDCG | 排除“仅新增 NLI”解释 |
| R032 | M3 | Graph-only 规则对照 | NIAH dev | MUST | TODO | — | deterministic | — | — | harmful / recall / NDCG | 隔离图规则价值 |
| R033 | M3 | Graph+ML | NIAH dev | MUST | TODO | — | 13 | — | — | harmful / recall / NDCG | — |
| R034 | M3 | Graph+ML | NIAH dev | MUST | TODO | — | 42 | — | — | harmful / recall / NDCG | — |
| R035 | M3 | Graph+ML | NIAH dev | MUST | TODO | — | 73 | — | — | harmful / recall / NDCG | — |
| R036 | M3 | ~~CLAIM_REFUTES 消融~~ | — | — | **N/A** | — | — | — | — | — | 经 g2-proto-2（M0 §9.7）消解：关系模型不再产出 REFUTES，`conflict_mode=refutes_edge` 臂失去可执行性，本消融无实质含义。如 A1 日后被推翻则本行恢复 |
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
