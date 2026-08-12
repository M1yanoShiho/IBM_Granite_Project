# ML Evidence Selector 实验计划

**分支：** `week5_MLSelector`
**版本：** 2.0（专家审查修订版）
**日期：** 2026-07-10

## 1. 实验要回答什么

现有 CorroborationReranker 使用固定公式对前 20 条候选证据重排序：

```text
最终分数 = 0.6 × 相关性 + 0.4 × 候选答案印证
```

本实验训练一个 ML Evidence Selector，让模型学习相关性、答案印证、直接支持、
冲突、条件、时间和来源等信号怎样共同决定证据的排序位置。

所有主要方法保持同一运行流程：

```text
Query2Doc + Granite dense
→ 固定候选池 top-20
→ 对前 20 条重新打分和排序
→ 第 20 名之后保持原顺序
→ 固定 top-10 进入 Granite 生成器
```

本轮方法范围限定为 candidate reranking。动态阈值、拒答、证据集合组合选择和
Granite 微调在本轮不执行。

## 2. 核心研究结论

### 主要结论

在相同候选池、上下文预算和生成器下，ML selector 能够比固定 `0.6/0.4`
公式选入更多可支撑答案的证据，并减少有害证据进入 LLM context。

### 支撑结论

模型在合同、财报、歧义和冲突环境中保持有效，提升不会只集中在人工构造的
NIAH 反事实证据上。

### 需要排除的替代解释

- 模型记住了旧 NIAH 的实体替换规律。
- 模型通过数据集缺失字段识别数据来源。
- 训练和测试共享同一父文档、合同或公司。
- Full 模型的提升全部来自新增的 LLM judge。
- 不同方法使用了不同候选、不同 token 预算或不同抽取结果。
- 结果只是在旧 NQ 300 上重复调参。

## 3. 比较系统

### 3.1 主要比较

| 系统 | 作用 | 推理成本 |
|---|---|---|
| `q2d_granite` | 原始相关性排序 | 一阶段基线 |
| `q2d_corroborate_fixed` | 固定 `alpha=0.6` 的现有方法 | 现有印证成本 |
| `q2d_corroborate_alpha_star` | 只在开发集选择最佳固定 alpha | 检查 ML 是否只修正了固定权重 |
| `ml_selector_core` | 学习 relevance、rank、corroboration 的组合 | 与现有方法成本匹配 |
| `ml_selector_full` | Core 加入支持、冲突、条件、时间和来源特征 | 单独报告新增成本 |

### 3.2 诊断对照

| 系统 | 回答的问题 |
|---|---|
| Granite relevance reranker | 普通相关性重排是否仍然无效 |
| support-score-only | Full 的提升是否主要来自语义 judge |
| source-deduplicated fixed corroboration | 去掉同一来源的重复投票后是否更稳 |
| Oracle@20 | top-20 内理论上还能提升多少 |
| shuffled-label model | 模型是否能在随机标签上产生虚假提升 |

端到端 RAG 只运行 `q2d`、fixed、Core 和 Full，控制 GPU 成本。

## 4. 数据集角色

| 数据 | 角色 | 训练使用 | 最终结论 |
|---|---|---|---|
| 旧 NQ NIAH 300 | 复现已有结果 | 否 | legacy replication，不承担盲测结论 |
| 新 sealed NIAH | 可控误导环境下的确认测试 | 训练集与盲测集严格隔离 | 主要受控结论 |
| ContractNLI | 合同内部的证据定位、条件和例外 | 官方 train/dev | 法律领域结论 |
| FinanceBench open-source | 财报数字、引用和报告周期 | company-level 交叉验证 | 企业财报结论 |
| RAMDocs official | 未见冲突和歧义压力测试 | 否 | 次要 top-3 诊断 |
| Adapted RAMDocs-20 | 固定 20→10 的外部选择测试 | 否 | 外部冲突选择结论 |

数据来源：

- NQ / DPR：<https://github.com/facebookresearch/DPR>
- ContractNLI：<https://stanfordnlp.github.io/contract-nli/>
- FinanceBench：<https://github.com/patronus-ai/financebench>
- RAMDocs：<https://github.com/HanNight/RAMDocs>

## 5. 数据划分

### 5.1 旧 NQ NIAH 300

这批 query 已经用于 alpha、组合规则、模型大小和失败分析。它只用于检查新代码能否
复现已有结果，不再称为 final holdout。

### 5.2 新 sealed NIAH

从未参与现有实验的 NQ/DPR query 中建立三部分数据：

- 训练：2,000 个 query；先以 500 个完成小规模验证。
- 开发：300 个 query，用于特征、alpha 和 LightGBM 调参。
- sealed test：300 个 query，在 schema、特征、prompt 和超参数冻结后才运行。

划分规则：

- 按规范化 Wikipedia 父页面分组，不能只按 passage doc_id。
- 同一父页面产生的所有 query、gold passage 和 distractor 必须进入同一 split。
- 同一 gold 生成的所有 counterfactual 属于同一个 synthetic family，不能跨 split。
- sealed test 使用与训练集不同的反事实生成模板，并抽样检查文本流畅性。
- split 后对 query、父页面、文本 hash 和 synthetic family 做自动交叉检查。

### 5.3 ContractNLI

- 使用官方 train/dev/test 划分，并验证同一 contract 不跨 split。
- 每个样本的 query 是 hypothesis，候选只来自对应合同内部。
- 在合同中切分条款或 span，检索 top-20，再选择 top-10。
- Entailment 和 Contradiction 进入主排序训练。
- NotMentioned 单独作为“无充分证据”诊断，不进入主 LambdaRank loss。
- 多个不连续 evidence span 作为一个 required evidence set 保存。

### 5.4 FinanceBench

公开版本只有 150 个问题，按 company 做 nested 5-fold GroupCV：

- 同一 company 的所有报告和问题全部进入同一个 outer fold。
- outer fold 产生公司级 held-out 测试预测。
- inner fold 只在训练公司中选择参数和 early stopping。
- 最终汇总全部 150 个问题的 out-of-fold 预测。
- 表格证据和纯文本证据分层报告。

FinanceBench 不再使用随机 60/20/20 划分，也不允许同公司不同年份跨训练和测试。

### 5.5 RAMDocs

官方 RAMDocs 每题文档数通常少于 10，固定 top-10 几乎不会过滤候选。因此分为：

- Official RAMDocs：保留官方候选，使用 top-3，测试排序和冲突暴露。
- Adapted RAMDocs-20：保留全部官方文档，再从外部语料中用 q2d mining 补足到 20，
  最终取 top-10。

Adapted RAMDocs-20 必须保存补充规则和 candidate IDs，并明确标注为改编测试集。
RAMDocs 的任何标签和 type 字段都不能参与训练或调参。

## 6. 候选池和公平条件

### 6.1 候选生成

- NIAH 和 FinanceBench：在对应语料库运行 `q2d_granite`，按父文档去重后取 top-20。
- ContractNLI：只在当前合同内部运行检索并取 top-20 span。
- Official RAMDocs：使用官方候选集合。
- Adapted RAMDocs-20：官方候选加 q2d mined distractors，固定为 20 条。

### 6.2 所有方法共同锁定

- query text 和 candidate IDs。
- chunking 方式、chunk 长度和 overlap。
- top-20 候选顺序及原始分数。
- 每条候选允许进入 context 的最大 token 数。
- top-10 context 的总 token budget。
- 文档去重和 tie-break 规则。
- Granite 抽取模型、prompt、版本和输出缓存。
- RAG 生成器、prompt、解码参数和候选呈现顺序。

Core 和 fixed 使用完全相同的 Granite 抽取缓存。Full 新增的计算单独记录模型调用数、
GPU 时间和每 query 延迟。

### 6.3 候选池诊断

每个数据集都先报告：

- `Required Evidence Recall@20`：需要的证据是否进入候选池。
- `Oracle NDCG@10`：top-20 完美排序的理论上限。
- `Retrieval Failure Rate`：top-20 内没有任何可用证据的比例。

训练排序器时，全为同一标签的 query group 不产生有效排序梯度。这些 group 从
LambdaRank loss 中排除，但必须保留在检索失败、风险和端到端评估中。

## 7. 标签设计

### 7.1 保留多维原始标签

每条候选至少保存：

| 字段 | 可选值或含义 |
|---|---|
| `support_level` | direct / partial / none |
| `factual_status` | correct / incorrect / disputed / unknown |
| `entity_match` | match / mismatch / unknown / not_applicable |
| `time_match` | match / outdated / unknown / not_applicable |
| `condition_match` | match / mismatch / unknown / not_applicable |
| `answerability` | answer / non_answer / extraction_failure |
| `source_quality` | strong / weak / unknown |
| `conflict_to_gold` | yes / no / unknown |
| `answer_cluster_id` | 候选支持的答案或 stance 簇 |
| `required_fact_set_id` | 多条证据共同构成答案时的集合标识 |
| `source_parent_id` | Wikipedia 页面、合同、公司或财报 |
| `harm_type` | outdated、wrong_condition、entity_error、misinfo 等 |
| `label_provenance` | official / human / deterministic_rule / independent_teacher |
| `label_confidence` | 0–1 |

候选之间的冲突另外保存为 pairwise relation：

```text
query_id, candidate_a, candidate_b, supports | contradicts | unrelated
```

两条候选互相冲突时，不能自动把双方都标成有害。需要结合 gold、时间、条件和来源判断。

### 7.2 五级排序目标

| `utility_grade` | 定义 |
|---:|---|
| 4 | 单条即可正确、完整、条件适用地支持答案或决策 |
| 3 | 正确且必要的部分或互补证据，单条不足 |
| 2 | 主题相关且真实，但无法形成答案证据 |
| 1 | 无关材料或一般噪声 |
| 0 | 错误、冲突、过时、错误实体或条件不适用，可能诱导错误答案 |

同时派生：

- `is_harmful=1`：utility 0。
- `is_direct_support=1`：utility 4。
- `is_required_support=1`：utility 3 或 4。

旧的 0/1/2 标签只作为消融，不作为唯一 canonical label。

### 7.3 各数据集映射

**NIAH**

- 直接回答问题且通过独立 answerability 检查：4。
- 正确但需要和其他证据合用：3。
- generative non-answer 若主题相关且内容真实：2。
- 一般无关 passage：1。
- counterfactual、错误实体、错误条件和事实冲突：0。
- designated needle 只作 legacy 指标；其他可正确回答的 gold passage 同样是正例。

**ContractNLI**

- 单条 span 足以判断 entailment 或 contradiction：4。
- required evidence set 中必要但单独不足的 span：3。
- 相关但不能决定 stance 的条款：2。
- 无关条款：1。
- 会导致错误 stance、条件不适用或被例外条款推翻的证据：0。
- NotMentioned 不进入主排序训练。

**FinanceBench**

- 官方 evidence 单条即可支持答案：4。
- 官方 evidence set 中必要的部分：3。
- 同节相关背景但不构成答案：2。
- 一般无关财报内容：1。
- 错误公司、报告周期、指标、单位或数值：0。

**RAMDocs**

- `correct`：4，并保留 answer cluster。
- `noise`：1。
- `misinfo`：0。
- 多个合法 disambiguated answer 分别保留，不能按多数票删掉少数合法答案。

### 7.4 标签来源隔离

- 官方 evidence 标注优先作为标签。
- 时间、公司和报告周期等可验证字段使用确定性规则。
- partial、weak-source 和复杂冲突由人工审查或独立 teacher 判断。
- 提供 Full 特征的 Granite/NLI judge 不能同时生成该候选的训练标签。
- gold answer、official evidence flag 和 RAMDocs type 不得进入模型特征。

### 7.5 标签质量检查

- 从每个数据集抽取完整 query group，总计至少 80 组。
- 两名成员独立标注，分歧由第三人裁决。
- 报告 weighted kappa、每级召回、harmful/direct 二分类一致率和混淆矩阵。
- raw agreement 只作补充，不能单独作为通过标准。
- weighted kappa 低于 0.70 时，先修订标注手册再继续训练。

## 8. 特征设计

### 8.1 Core 特征

- normalized first-stage relevance score。
- first-stage rank 和 reciprocal rank。
- exact-normalized candidate answer vote count。
- vote count / 有效答案数量。
- parametric answer agreement。
- 抽取是否失败或返回 `NONE`。

Core 与 fixed baseline 使用相同抽取缓存，形成严格 cost-matched 比较。

### 8.2 Full 特征

- query-candidate direct support score。
- condition coverage score。
- time、entity、jurisdiction 和数值条件匹配。
- candidate 与池内其他候选的最大 support 和 contradiction score。
- source-deduplicated support count。
- 与高排名候选的重复度。
- 文档版本、报告周期和来源等级。
- 每类 metadata 的 missing indicator。

### 8.3 QA 与 NLI 分开抽取

当前 corroboration prompt 只适合短答案 QA。本实验冻结两套结构化输出：

**QA schema**

```text
answer, answerable, answer_type, unit, condition_coverage
```

**ContractNLI schema**

```text
stance = entail | contradict | unknown
evidence_sufficiency, exception_detected, condition_coverage
```

固定 baseline、Core 和 Full 在同一任务中共享同一份抽取结果。ContractNLI 不使用
短答案 exact vote。

### 8.4 防止域指纹

- 不输入 dataset name、failure type、gold flag 或构造模板。
- 训练一个 diagnostic classifier，尝试只用 Full 特征预测 dataset。
- dataset prediction 明显高于多数类基线时，执行 metadata-free 和
  missingness-masked 消融。
- 同时报告 natural weighting 与 domain-balanced weighting。

### 8.5 防止构造规律过拟合

- 轻量模型不直接读取 raw candidate text。
- sealed NIAH 使用不同反事实生成模板。
- 训练时加入 fluent counterfactual 和真实 mined negatives。
- 运行 shuffled-label negative control，结果应接近随机排序。

## 9. 模型训练

### 9.1 主模型

使用 LightGBM `LGBMRanker`：

```text
objective = lambdarank
metric = ndcg
eval_at = 10
label_gain = [0, 1, 3, 7, 15]
group = 每个 query 的候选数量
```

每个 query 总权重相同，候选行权重除以该 query 的候选数。再通过 dataset-level
weight 避免 NQ 数量压过 FinanceBench 和 ContractNLI。

### 9.2 两个正式版本

- `ml_selector_core`：只使用 Core 特征。
- `ml_selector_full`：使用 Core + Full 特征。

模型输出直接作为前 20 条候选的最终排序分数。本轮不设置阈值，固定取 top-10。

### 9.3 调参

- 只在开发数据或 inner GroupCV 中调参。
- 预先固定唯一 early-stopping 主指标：`NDCG@10`。
- 小范围搜索 `num_leaves`、`max_depth`、`learning_rate`、
  `min_data_in_leaf` 和 `feature_fraction`。
- 运行 3 个随机种子。
- FinanceBench 重点报告 company fold 方差，不能用 seed 方差代替 split 方差。

### 9.4 混合训练

主 mixed model 使用：

```text
NQ sealed-NIAH train
+ ContractNLI train
+ FinanceBench 当前 outer-fold 的训练公司
```

每个数据集保持相近 query 总权重。NIAH 内继续平衡 counterfactual、non-answer、
mined negative 和 natural noise。

### 9.5 跨数据集训练

至少运行：

```text
NQ + FinanceBench → ContractNLI
NQ + ContractNLI → FinanceBench
NQ + ContractNLI + FinanceBench → RAMDocs / RAMDocs-20
```

这些结果用于判断模型是否学到可迁移的证据判断，还是依赖单一数据域。

## 10. 实验矩阵

| Block | 数据与切分 | 候选设置 | 主要作用 |
|---|---|---|---|
| B1 Legacy | 旧 NQ NIAH 300 | 原 top-20→10 | 复现已有结果 |
| B2 Blind controlled | 新 sealed NIAH test | 同一 top-20→10 | 主要确认实验 |
| B3 Contract | ContractNLI official test | 合同内 top-20 span→10 | 条件、例外和 stance |
| B4 Finance | FinanceBench company nested 5-fold OOF | 全财报库 top-20→10 | 企业财报证据 |
| B5 Conflict official | RAMDocs official | official pool→top-3 | 排序与冲突诊断 |
| B6 Conflict adapted | Adapted RAMDocs-20 | 20→10 | 外部证据筛选 |
| B7 LODO | 留一数据集测试 | 与对应数据一致 | 跨域泛化 |
| B8 Ablation | 所有 test | 固定候选 | 特征和标签贡献 |
| B9 RAG | B2、B4、B5/B6 | 固定 token budget | 答案与引用闭环 |

## 11. 必做消融和负控

- relevance only。
- relevance + rank。
- relevance + corroboration。
- 去掉 parametric vote。
- support-score-only。
- Core。
- Full。
- Full 去掉 relevance。
- Full 去掉 metadata。
- missingness-masked Full。
- 三级标签 vs 五级标签。
- source-deduplicated vote vs passage vote。
- shuffled labels。
- feature-to-dataset prediction probe。
- 无训练的 source-cap / MMR 诊断 baseline，用于检查重复来源控制是否已经足够。

## 12. 评价指标

### 12.1 通用证据指标

- `NDCG@10`：五级 utility 的主要排序指标。
- `Required Evidence Recall@10`：答案所需证据是否进入 context。
- `Direct Support Precision@10`：utility 4 的比例。
- `Harmful Evidence Rate@10`：utility 0 的比例。
- `Noise Rate@10`：utility 1 的比例。
- `Conflict Exposure@10`：context 是否暴露互相冲突的证据。
- `MRR`：第一条 direct support 的位置。

### 12.2 数据集专用指标

**NIAH**

- needle-found@10 作为 legacy 指标。
- NDCG@10、Harmful@10 和 Required Evidence Recall@10 作为新 selector 指标。

**ContractNLI**

- evidence span precision、recall 和 F1。
- required evidence-set coverage。
- 最终 NLI accuracy。

**FinanceBench**

- evidence-set coverage。
- citation support。
- 数值、百分比和单位感知的 answer accuracy。
- 表格题和纯文本题分层表现。

**RAMDocs**

- answer-cluster coverage。
- misinfo exposure。
- strict EM：包含所需 gold answer 且不包含 wrong answer。

### 12.3 RAG 指标

- Blind NIAH：Answer F1 作为预注册主指标。
- FinanceBench：数值 answer accuracy 作为主指标，citation support 为共同主结果。
- RAMDocs：strict EM 作为主指标。
- cover-EM、context precision 和 faithfulness 作为次要指标。

Citation Support 使用独立 entailment judge，并抽样进行双人复核。提供 Full 特征的 judge
不能同时承担最终 citation judge。

## 13. 统计检验

### 13.1 统计单位

- NIAH 和 RAMDocs：query。
- ContractNLI：contract cluster。
- FinanceBench：company cluster。

### 13.2 主要比较

- 主要方法比较：`ml_selector_full` vs `q2d_corroborate_fixed`。
- 成本匹配比较：`ml_selector_core` vs `q2d_corroborate_fixed`。
- 其他系统为诊断或消融。

### 13.3 方法

- 使用 cluster-level paired bootstrap 或 paired randomization。
- 报告 95% confidence interval。
- 多数据集、多指标比较使用 Holm correction。
- macro average 只作描述，各数据集必须单独报告。
- “p≥0.05”不能作为无退化证据；无退化使用单侧 non-inferiority interval。

## 14. 通过标准

### Gate 0：数据和标签可信

- query、父页面、contract、company 和 synthetic family 无跨 split 泄漏。
- 标签审查 weighted kappa ≥ 0.70。
- gold、dataset type 和生成规则字段未进入模型特征。
- candidate cache 对每个方法完全一致。

### Gate 1：受控盲测有效

在新 sealed NIAH 上，Full 相对 fixed 同时满足：

- `NDCG@10` 至少提高 0.02。
- `Harmful Evidence Rate@10` 至少降低 0.02。
- 主差值 95% CI 下界大于 0，并报告 Holm-adjusted `p < 0.05`。
- Required Evidence Recall@10 的单侧 95% CI 下界高于 -0.01。

Core 同时报告相同指标，用来判断在相同成本下是否已经有效。

### Gate 2：真实数据和跨域泛化

- ContractNLI、FinanceBench、Adapted RAMDocs-20 中至少两个取得正向结果。
- 任一数据集 Required Evidence Recall 的非劣下界不得低于 -0.01。
- LODO 至少两个目标域为正。
- RAMDocs official 和 adapted 结果分开报告。

### Gate 3：答案端受益

- Blind NIAH、FinanceBench、RAMDocs 中至少两个主答案指标改善。
- 任何数据集均通过预注册的非劣检验。
- Citation Support 的改善通过独立 judge 和人工抽样检查。
- Full 的质量增益与额外延迟一起报告。

### 停止条件

- Gate 0 未通过：停止训练，修复 split 或标签。
- Gate 1 未通过：停止完整 RAG，检查特征、标签和候选 headroom。
- Gate 1 通过、Gate 2 失败：结论限定为受控 NIAH 环境。
- Gate 1 和 Gate 2 通过：运行端到端 RAG。
- Gate 3 通过：考虑 set-aware selection 或 Granite 微调。

## 15. 执行顺序

| 阶段 | 工作 | 产出 | 继续条件 |
|---|---|---|---|
| M0 协议冻结 | 冻结 schema、数据角色、主指标和 Gate | protocol manifest | 团队确认 |
| M1 数据与 split | 建立父页面、contract、company 级划分 | split manifest | 泄漏检查通过 |
| M2 标签 | 生成多维标签和五级 utility | labels + audit | Gate 0 标签部分通过 |
| M3 候选与特征 | 生成并缓存固定 top-20 和抽取结果 | candidate/feature cache | pool 和 hash 一致 |
| M4 Baseline | 复现 q2d、fixed、alpha*、source-dedup | per-query metrics | 与 legacy 数字一致 |
| M5 Sanity | Logistic/linear、Core、小规模 Full、负控 | sanity report | 无明显域指纹 |
| M6 主训练 | mixed LightGBM Core 和 Full | frozen models | 只在 dev 选择模型 |
| M7 确认测试 | sealed NIAH、Contract、Finance、RAMDocs | main results | Gate 1–2 判断 |
| M8 RAG | 固定 top-10 和 token budget 生成答案 | RAG results | Gate 3 判断 |
| M9 系统接入 | 注册 `q2d_ml_selector` 并补测试 | retriever arm | 离线和在线排名一致 |

## 16. 产出文件

```text
data/selector/
  dataset_manifest.json
  split_manifest.json
  label_schema.json
  protocol_manifest.json

results/ml_selector/
  candidates/
  features/
  labels/
    label_audit.csv
    pairwise_relations.parquet
  models/
    core/
    full/
  metrics/
    legacy_replication.csv
    blind_niah.csv
    contractnli.csv
    financebench_oof.csv
    ramdocs_official.csv
    ramdocs20.csv
    leave_one_domain_out.csv
    ablations.csv
    rag_results.csv
  predictions/
  figures/
```

每次运行记录 git commit、随机种子、数据版本、父文档 split、模型版本、prompt hash、
候选池 hash、特征 schema、超参数、硬件、运行时间和模型调用次数。

## 17. 最终呈现

最终报告只保留五项核心结果：

1. 主表：fixed、Core、Full 在各数据集上的 NDCG、Required Recall 和 Harmful Rate。
2. 成本图：证据质量提升与每 query 延迟、模型调用数的关系。
3. 泛化图：LODO 和 RAMDocs 上相对 fixed 的差值与置信区间。
4. 消融图：relevance、corroboration、support/conflict、metadata 的贡献。
5. RAG 图：证据选择提升是否传递到答案正确性和 citation support。

结论按 Gate 决定：

- Gate 1–3 全部通过：ML selector 是固定 corroboration 的有效可学习扩展。
- Gate 1 通过、Gate 2 失败：方法只在受控证据环境中成立。
- Gate 1 未通过：当前特征不足以证明 ML 增加了价值。

## 18. 本轮不执行

- Granite selector 微调。
- 枚举所有 top-10 证据组合。
- 学习式 set-aware selection。
- 动态阈值、校准和证据不足拒答。
- Granite embedding fine-tuning。
- 私有企业数据。

这些扩展只在 Core 或 Full 通过 Gate 1 和 Gate 2 后讨论。
