# Graph-Assisted Evidence Selector 2.0 — 实验执行计划

**项目：** IBM Granite Needle-in-a-Haystack RAG

**日期：** 2026-07-11

**修订：** zero-new-human-annotation protocol

**状态：** 待执行

**适用范围：** 模块二 q2d_granite Top-20 → Top-10；不修改模块一 Retriever，不替代模块三 grounded generation

**数据边界：** FinanceBench 状态为 `EXPOSED_DIAGNOSTIC_ONLY`；不参与训练、调参、停止决策、失败分析或最终盲测

> **V1/V2 边界：** V1 已经使用 FinanceBench 全部 150 个问题，因此它不能再承担任何“未见最终 benchmark”角色。若未来为了历史可比性报告 FinanceBench，必须单列为已暴露诊断结果，不能进入 V2 主张或模型选择。

## 0. Design review status

本计划经过三轮独立只读设计审核，最终 verdict 为 **PASS**。PASS 只表示协议内部一致、可审计且能在 zero-new-human-annotation 条件下执行，不保证实现正确或结果一定支持主张。

最终审核冻结了以下修正：

- 核心关系缩减为 `CLAIM_SUPPORTS`、`CLAIM_REFUTES`、`SAME_SOURCE`；
- primary label 只来自 official/deterministic provenance，不新增人工标注；
- claim extraction、clustering、relation prediction 和 Graph features 使用完整 OOF 流程；
- 必须运行 NLI-without-Graph、Graph-only 和 shuffled-Graph 控制；
- RAMDocs mined passages 标为 unjudged，不当作已标注 noise；
- FinanceBench 不进入 V2 主张或结果；
- downstream generation 只称为 Selector transmission check。

## 1. 研究问题与结论边界

### 1.1 核心研究问题

在读取完全相同的 q2d_granite Top-20、使用相同基础特征和相同训练预算时，显式建模候选证据与候选主张之间的支持、反驳和来源关系，能否比 ML Evidence Selector v1 选出更可靠的 Top-10 evidence set？

### 1.2 主要主张 C1

Graph + ML Selector 2.0 相比重新训练的 ML Selector v1：

- 记 ΔH = Harmful Rate@10(Graph 2.0) − Harmful Rate@10(v1)；成功要求点估计 ΔH ≤ −0.02，且双侧 95% CI 上界 < 0；
- Required Evidence Recall@10 的单侧非劣置信下界不低于 −0.01；
- NDCG@10 的单侧非劣置信下界不低于 −0.01；
- 结论在 fresh NIAH sealed 600 上成立；
- RAMDocs official 只承担外部方向验证，不承担相同 Top-20→Top-10 接口的确认性证明。

三个条件必须同时满足，不能用某个次级指标的改善替代失败的主指标。

### 1.3 支撑主张 C2

只有 C1 通过后，才检查更可靠的 Top-10 是否能传导到现有 Granite generator 的答案质量。

这是 Selector downstream transmission check，不是模块三的 generator 研究：

- 不修改或优化 generator、prompt、解码和 token budget；
- 只比较 v1 Top-10 与 Graph 2.0 Top-10；
- 自动 judge 得到的 citation/unsupported-claim 结果只能称为 proxy；
- 不研究模型先验知识抑制、generator 训练或 prompt intervention。

### 1.4 明确不作的主张

- 不声称已解决一般化的时间错配或条件错配；
- 不声称在 FinanceBench 上验证了 Selector 2.0，也不把它称为未见 benchmark；
- 不把官方数据集已有人工标签称为本项目新增人工标注；
- 不把 LLM teacher 输出当作 primary ground truth；
- 不把 RAMDocs adapted mined passages 当作已标注 noise；
- 不把旧 sealed NIAH 300 或昨天已查看的结果当作新的确认性证据。

## 2. 冻结的三模块接口

~~~text
模块一：q2d_granite Retriever
→ 冻结的 Top-20 candidate IDs 与抽取缓存

模块二：Graph + ML Selector 2.0
  ├─ Claim extraction / clustering
  ├─ Relation Builder
  ├─ Query-local Evidence Graph
  ├─ Graph features
  └─ LightGBM LambdaRank
→ Top-10 evidence set

模块三：现有 Granite generator
→ 仅在 C1 通过后做固定条件的下游传导检查
~~~

Relation Builder 和 query-local graph 都是 Selector 2.0 内部，不是 Retriever 前的新模块。本计划不训练 GNN、不建立全语料知识图谱、不引入 Neo4j。

## 3. Graph 2.0 方法定义

### 3.1 节点

- Query：当前问题；
- Passage：Top-20 候选证据；
- Claim/Answer：从 passage 自动抽取并聚类的候选主张；
- Source：source_parent_id 对应的父页面、合同或来源。

### 3.2 三类核心关系

主实验只冻结三类有官方或确定性监督来源的关系：

1. CLAIM_SUPPORTS：passage 支持某个候选 claim；
2. CLAIM_REFUTES：passage 反驳某个候选 claim；
3. SAME_SOURCE：passage 来自同一父来源。

无法确定时必须输出 UNKNOWN，不能强制建边。每条预测边保存 confidence、模型版本、source ID 与输入文本 hash。

### 3.3 时间和条件关系的处理

TIME_MISMATCH 与 CONDITION_MISMATCH 不进入本轮主要关系集合，因为 ContractNLI 和 VitaminC 没有直接提供“是否匹配当前 query 时间/条件”的官方标签。

只有存在明确、机器可验证 metadata 时，才允许生成辅助规则值；否则固定为 UNKNOWN。这些辅助值：

- 不进入 C1 的主要 Graph 特征；
- 不用于主张时间或条件推理能力；
- 不进入主消融结论。

若以后加入带可逆 mutation log 的自动 time/condition challenge set，必须作为新的协议版本重新冻结，不能在 fresh test 结果出来后追加。

### 3.4 Relation Builder

- 基础模型固定为 cross-encoder/nli-deberta-v3-base；
- ContractNLI official train/dev/test 用于 claim–passage entailment、contradiction、unknown。只有 singleton evidence span 可直接形成 passage→claim gold edge；multi-span annotation 必须按原文顺序拼成一个 evidence-set premise，不能把每个 span 独立标成 direct relation；NotMentioned 只为采样的非证据 premise 提供 UNKNOWN；
- VitaminC 使用 official test，并在保持 official test 完全不动的前提下对 train/dev 做 revision-family 去污染，用于近似文本变化下的 support/refute/unknown；
- VitaminC 的 revision 来源不等于 TIME_MISMATCH 标签，只用于 contrastive fact verification；
- NIAH train 中由官方 DPR qrels、确定性 counterfactual provenance、answer cluster 和 source_parent_id 推导的关系仅用于领域适配；
- SAME_SOURCE 只由 source_parent_id 确定性生成；
- 正式 dev/test 建图不得读取 utility_grade、harm_type、gold answer_cluster_id、gold answers、official evidence flag 或 RAMDocs type。

NIAH train 按父页面和 synthetic family 做 5-fold OOF。每折的 claim extraction、claim clustering、relation prediction 和 Graph feature generation 都必须只用其余四折拟合；不能只对最后一层 relation score 做 OOF。NIAH dev/test 由冻结后的完整链路直接预测。

### 3.5 核心 Graph 特征

在 v1 基础特征上只增加：

- independent_support_count；
- contradiction_count；
- same_source_duplicate_count；
- answer_cluster_size；
- alternative_answer_entropy。

时间和条件特征不进入本轮主要模型。

### 3.6 Selector 与随机种子

- v1 和 v2 都使用 LightGBM LambdaRank；
- 两者读取相同训练 query、Top-20、utility labels、基础特征与 label_gain；
- v2 唯一新增的是冻结的 Graph 特征；
- Top-20、Top-10、tie-break 和种子 13/42/73 完全一致；
- dev 上冻结配置后，三个 seed 的标准化预测分数取平均形成唯一的 primary ensemble；
- fresh test 不允许挑选表现最好的 seed；单 seed 结果只作为敏感性分析。

## 4. 数据与划分

| 数据 | 唯一用途 | 是否训练 Selector | 标签来源 | 最终角色 |
|---|---|---:|---|---|
| ContractNLI train/dev/test | Relation Builder 的 claim–passage NLI | 否 | official choice；singleton span 或完整 evidence set | 关系训练、调参、一次测试 |
| VitaminC decontaminated train/dev + official test | Relation Builder 的 contrastive support/refute | 否 | official claim/evidence/label | 关系训练、调参、一次测试 |
| NIAH train 2000 | v1/v2 Selector 训练与 OOF Graph | 是 | DPR qrels + deterministic provenance | 主要训练集 |
| NIAH dev 300 | 特征、阈值、消融与配置选择 | 否 | 同上 | 唯一开发集 |
| 旧 sealed NIAH 300 | 代码回归与昨日结果复现 | 否 | 已有冻结标签 | 不作最终证明 |
| fresh NIAH sealed 600 | 一次性确认性测试 | 否 | unused DPR/NQ + deterministic provenance | 主要结论 |
| RAMDocs official 500 | 冲突、misinfo、noise 外部压力测试 | 否 | official correct/misinfo/noise | 外部方向验证 |
| RAMDocs adapted Top-20 | 可选 secondary stress test | 否 | official rows judged；mined rows unjudged | 不进主结论 |

当前已物化 NIAH artifact 只有 500 个 train query。开始训练前必须按已有 split manifest 完整物化 2000 个 train query，或在 M0 预注册把训练规模改为 500；不能在看到 dev 结果后改变训练规模。

FinanceBench 在 Graph 2.0 中不加载、不训练、不调参、不测试、不分析失败案例。V1 的 FinanceBench 输出只保留为 `EXPOSED_DIAGNOSTIC_ONLY` 历史探索结果，不进入本轮模型选择、停止决策或最终结论。

### 4.1 零新增人工标注约束

本实验不安排任何新的人工标注、双标或人工裁决。

Primary ground truth 只允许：

- 数据集 official labels/qrels；
- 可复现的 deterministic rule；
- 带 source record、mutation log、seed 和 hash 的确定性 synthetic provenance。

Independent teacher 或 LLM judge 只能用于特征、故障诊断或 secondary proxy，不能作为 primary label、确认性评分或通过 Gate 的唯一依据。

### 4.2 fresh NIAH 600 的自动构建

- 从现有 split 未使用的 DPR/NQ query 中抽取；
- 与 train/dev/旧 test 在 query、父页面、answer entity、passage hash 和 synthetic family 上零重叠；
- 每题保留固定 q2d_granite Top-20 和至多一个按以下冻结算法生成的确定性 counterfactual；
- 只接受具有单一 normalized gold value、且某个 gold surface alias 在 needle 中恰好出现一次的 query；
- replacement 只从 seed=42 冻结的 NIAH-train answer bank 中选择，并与当前 gold value 不同；
- replacement 必须与 gold 属于同一机械字符串类别：integer、decimal、year/date、proper-name token-length bucket 或 common-noun token-length bucket；无法归类或没有同类 replacement 时丢弃；
- 替换后必须满足：所有 gold alias 均不残留、replacement 不属于 gold aliases、只有目标字符 span 发生变化、文本可由 mutation log 反向恢复；
- mutation log 必须记录 answer-bank hash、原答案、替代值、字符串类别、seed、变换位置和变换前后文本 hash；
- 多个互异 gold value、替代值可能仍被官方 aliases 接受、实体类别不匹配或任何规则无法机械确认时，样本自动丢弃；
- 在训练和 dev 实验前只冻结 manifest/hash，不查看模型结果。

### 4.3 样本量依据

统计单位是 query，不是 Top-10 中的 6000 个 passage，避免伪重复。

在当前“一题最多一个 harmful counterfactual”的 NIAH 结构下，Harmful Rate@10 下降 0.02 等价于 harmful query exposure 下降约 20 个百分点。规划计算使用旧 test 的 baseline exposure 0.68、目标 0.48、双侧 alpha=0.05、power=0.90，并按保守的两个独立比例正态近似计算为每组约 126 个 query；实际主比较是在同一 600 个 query 上配对完成。M0 必须保存公式、输入参数和可复现脚本。

在 M0 必须再使用 NIAH dev 上的 v1/v2 paired discordance 做 Monte Carlo sensitivity analysis。如果数据结构或最小效应改变，必须在冻结 fresh test 前重算，而不是事后报告 observed power。

### 4.4 防泄漏规则

- NIAH 按 Wikipedia 父页面、answer entity 和 synthetic family 分组；
- ContractNLI 使用 official split，同一合同不得跨 split；
- VitaminC 先审计 official split 的 revision family/page；official test 永不移动。若 train/dev 与 dev/test 共享 family，依次删除 train 中与 dev/test 冲突的样本及 dev 中与 test 冲突的样本，形成 test-preserving decontaminated train/dev，并保存删除清单；不得重构或移动 official test；
- fresh 600 与现有 split 做 query、父页面、entity、文本 hash 和模板族审计；
- dev/test 的 official labels 只能在预测完成后评分；
- dataset name、gold flag、official evidence flag、utility_grade、harm_type 和 RAMDocs type 不得进入模型特征；
- 所有系统读取完全相同的 Top-20 candidate manifest 和 cache hash。

## 5. 自动质量 Gate

### Gate 0A：Label provenance 与数据完整性

自动检查：

- primary label 的 provenance 只能是 official 或 deterministic_rule；
- split/group/hash 零重叠；
- 每题 candidate count、Top-20 ID、utility range 与 derived flags 一致；
- counterfactual mutation 可逆且 gold alias 不残留；
- fresh 600 manifest 在运行前冻结；
- 不存在人工标注待办。

通过条件：所有硬性 invariant 零违规。任何违规都停止训练并重新构建数据。

### Gate 0B：Relation Builder 自动验证

调参只使用 ContractNLI/VitaminC dev；模型冻结后各自 official test 只运行一次。

报告：

- per-class precision/recall/F1、macro-F1；
- UNKNOWN/abstention rate 与 edge coverage；
- CLAIM_REFUTES precision；
- overall edge coverage 与 SUPPORT/REFUTES 各自 coverage；
- SAME_SOURCE exact-rule unit tests；
- metamorphic tests：重复同一 source 不增加 independent support；打乱 edge 后 Graph 收益应消失。

通过条件：

- ContractNLI 与 VitaminC 必须分别通过，不能 pooled 后掩盖某一数据集失败；
- CLAIM_REFUTES precision ≥ 0.85；
- support/refute macro-F1 ≥ 0.80；
- overall non-UNKNOWN coverage ≥ 0.80，SUPPORT 与 REFUTES 各自 coverage ≥ 0.70；
- SAME_SOURCE 规则测试 100% 通过；
- parse failure 与 UNKNOWN rate 完整报告。

未通过时只修 Relation Builder，不得进入 Selector 确认性测试。官方 test 失败后不得反复调参再重测并仍称其为一次性确认。

## 6. 实验对照

| 系统 | 目的 | 主表 |
|---|---|---:|
| q2d | 原始 Top-20 截取 Top-10 | 是 |
| fixed_0.6 | 已有固定 corroboration | 是 |
| ml_selector_v1_full | 使用相同数据重新训练的 v1 | 是 |
| nli_node_ml_no_graph | 加入本地 NLI 概率但不做 Graph aggregation | 是 |
| graph_rule_only | 检查图规则本身 | 是 |
| graph_ml_selector_v2 | 主方法 | 是 |
| shuffled_graph_ml | 随机图负控 | 诊断 |
| oracle_at_20 | 候选池理论上限 | 诊断 |

nli_node_ml_no_graph 用于排除“收益只是来自额外 NLI 模型或更多参数”的替代解释。

## 7. 必须运行的实验块

### Block 1：自动数据与关系 Gate

- 数据：全部 manifest、ContractNLI dev/test、VitaminC dev/test、deterministic rule tests；
- 产物：provenance audit、split/hash audit、relation confusion matrix、coverage/UNKNOWN table；
- 通过：Gate 0A 与 Gate 0B 同时通过；
- 优先级：MUST。

### Block 2：Graph 2.0 主结果

- 数据：fresh NIAH sealed 600；
- 系统：q2d、fixed、重新训练 v1、NLI-no-graph、graph-only、Graph 2.0、oracle；
- 唯一确认性比较：Graph 2.0 primary ensemble vs v1 primary ensemble；
- 主指标：Harmful Rate@10、Required Evidence Recall@10；
- guardrail：NDCG@10；
- 次指标：direct support precision、conflict exposure、MRR；
- 通过：满足 C1 全部条件；
- 失败：不得进入 Graph 可靠性主张；
- 优先级：MUST。

### Block 3：Graph 贡献隔离

只在 NIAH dev 上选配置，fresh test 上只执行已冻结版本：

- 完整 Graph 2.0；
- 去掉 CLAIM_REFUTES；
- 去掉 SAME_SOURCE；
- NLI node features、无 graph aggregation；
- shuffled graph；
- graph-only、无 ML。

完整 Graph 必须优于 v1，至少一个有意义关系消融应降低效果，shuffled graph 不得复现收益。

### Block 4：RAMDocs official 外部方向验证

- 使用官方 candidate pool 与 official correct/misinfo/noise 标签；
- 报告官方 Top-3 protocol 下的 required recall、misinfo exposure、NDCG 和 answer coverage；
- 不训练、不调参；
- 因输入池不是固定 Top-20，本 Block 只支持外部方向一致性，不支持相同接口的确认性结论；
- adapted Top-20 如运行，跨 query mined passages 一律标为 unjudged，结果单列为 secondary stress test。

### Block 5：Selector downstream transmission check

只有 Block 2 通过后运行：

- 数据：fresh NIAH 中预注册的固定答案子集；
- 系统：v1 Top-10 → 同一 Granite 与 Graph 2.0 Top-10 → 同一 Granite；
- 固定 generator、prompt、解码、passage order、token budget；
- 主要报告 answer F1/cover-EM；
- citation support 与 unsupported claim 若由自动 judge 评分，标记为 proxy；
- 本 Block 不构成模块三的 prior-knowledge mitigation 实验。

## 8. 统计协议

- 独立统计单位：query；
- bootstrap group：NIAH 父页面/synthetic family；
- 主比较：Graph 2.0 vs v1，query-level paired randomization test + grouped bootstrap CI；
- Harmful Rate 报告双侧 95% CI，并严格使用 ΔH ≤ −0.02 且 CI 上界 < 0 的联合判定；
- Required Recall 与 NDCG 使用预注册的单侧非劣 CI；
- 次级指标、消融和额外系统比较使用 Holm correction；
- 同一次 fresh run 必须产生全部冻结系统的结果，不能先看 v2 再选择是否运行 baseline；
- 报告 effect size、CI 和 raw per-query outcomes，不只报告 p-value；
- 三个 seed 的 primary ensemble 预先固定，不能在 fresh test 选 seed。

## 9. 运行顺序与停止门

| 里程碑 | 目标 | 决策门 |
|---|---|---|
| M0 | 冻结协议、三关系 schema、2000-train 决策、fresh 600 manifest、power sensitivity | 自动 provenance/hash audit 零违规 |
| M1 | ContractNLI/VitaminC adapter 与 Relation Builder | dev 配置冻结 |
| M2 | official relation test、OOF claim/relation/graph cache | Gate 0B 通过 |
| M3 | 同数据复现 v1，训练 NLI-no-graph、graph-only、Graph 2.0，dev 消融 | dev 有可归因 headroom |
| M4 | 冻结代码、模型、三-seed ensemble、threshold、hash | 不再调参 |
| M5 | fresh NIAH 600 一次性运行全部系统 | C1 Gate |
| M6 | RAMDocs official 外部方向验证 | 不调参 |
| M7 | 条件性 downstream transmission check 与汇总 | 仅 C1 通过后 |

立即停止 Graph 2.0 主张的条件：

1. primary labels 存在非 official/deterministic provenance；
2. relation Gate 未通过；
3. Graph 特征与 v1 基础特征重复且 dev 增益不足 0.01；
4. fresh test 未满足 harmful、required recall、NDCG 三项联合 Gate；
5. shuffled graph 或 NLI-no-graph 复现全部收益，无法归因 Graph；
6. 必须读取 dev/test gold label 才能建图；
7. RAMDocs official 出现明显负迁移时，停止外部通用性表述。

## 10. 计算预算

- ContractNLI/VitaminC Relation Builder，3 seeds：约 8–16 GPUh；
- OOF claim/relation/Graph cache：约 2–8 GPUh；
- LightGBM v1/v2 与消融：CPU，通常少于 2 小时；
- fresh NIAH + RAMDocs 推理：约 4–10 GPUh；
- 条件性 generation：约 8–16 GPUh；
- 不包含任何新增人工标注工时。

首轮 VitaminC 使用 relation-balanced 100k train subset；只有 dev 欠拟合且在协议冻结前，才允许扩大。

## 11. 结果与复现要求

正式运行必须保存：

- protocol、三关系 schema、split manifest 和 provenance audit；
- candidate pool、claim extraction、relation model、Graph cache 与 feature cache hash；
- 每题 Top-20、预测 claims/edges、UNKNOWN、Graph features 和 Top-10；
- 三个 seed 与 primary ensemble；
- per-query metrics、aggregate metrics、CI、raw/adjusted p-value；
- GPU、模型版本、运行时间、调用数和缓存命中率。

建议目录：

~~~text
data/graph_selector/
results/graph_selector/
docs/data/graph_selector_validation/
~~~

主表报告 q2d、fixed、v1、NLI-no-graph、graph-only、Graph 2.0 与 oracle 在 fresh NIAH 上的：

- Harmful Rate@10；
- Required Evidence Recall@10；
- NDCG@10；
- Conflict Exposure@10；
- 成本与延迟。

RAMDocs official 单独报告，不能与 NIAH Top-20→Top-10 主表混成同一 protocol。

## 12. 执行检查清单

- [ ] FinanceBench 未被 Graph 2.0 加载或用于失败分析
- [ ] 三类 core relation schema 与 UNKNOWN 策略冻结
- [ ] 没有新增人工标注、双标或裁决步骤
- [ ] primary labels 全部为 official/deterministic
- [ ] ContractNLI/VitaminC official split 无 family 泄漏
- [ ] NIAH train 明确使用 500 或 2000，并在 dev 前冻结
- [ ] OOF 覆盖 claim extraction、clustering、relation 与 Graph feature 全链路
- [ ] v1/v2 共享完全相同训练 query、Top-20、labels、基础特征和预算
- [ ] NLI-no-graph 与 shuffled graph 负控完成
- [ ] fresh 600 split/hash/provenance audit 通过
- [ ] fresh test 一次运行全部冻结系统
- [ ] query-level paired statistics 与 Holm correction 已实现
- [ ] RAMDocs mined passages 标为 unjudged
- [ ] downstream check 未被表述成模块三研究
- [ ] 所有结果保存 per-query 数据与 hash
