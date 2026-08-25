# Experiment 05 — 普通开放域 RAG 的可靠性与可追溯性完整系统评估（v4 快速执行版）

**日期：** 2026-08-22  
**当前状态：** `GOAL 1–3 PASS / GOAL 4 ACTIVE / USER-AUTHORIZED RAPID EXECUTION AMENDMENT`  
**前序实验：** [Experiment 04](../04_frozen_three_module_system_evaluation_2026-08-21/FINAL_REPORT.md)  
**当前允许的工作：** 严格按本 v4 的 Goal 边界连续执行；当前唯一 active goal 为 Goal 4。  
**当前禁止的工作：** Goal 4 只运行并冻结三个正式消融臂；不得读取 scorer-only gold、运行正式评分或提前运行 Goal 5。

> v3 已完成独立复审并获得 `PASS`。用户随后在执行对话中明确启动实验，并于
> 2026-08-22 在任何 formal 输出或 formal 评分产生前授权切换到时间受限的快速方案。
> v4 保留完整语料 BM25 检索、400/400/400 正式样本、五系统、三消融、六主指标和
> gold 隔离，只把不可在时限内完成的“全语料独立 dense index”替换为冻结的两阶段
> 检索：`full-corpus BM25 Top-1000 -> Granite dense candidate scoring -> RRF/rerank`。

## 0. v4 快速执行修订（当前权威）

本节取代 v3 中所有要求“构建或使用全语料独立 Granite dense index”的条款；其余未明确
修改的抽样、模型、预算、评分、gold 隔离、Goal 接力和失败分母规则继续有效。

1. 每个 query 先在冻结的完整 KILT/DPR BM25 index 上检索 Top-1000；不得注入 gold、
   oracle passage 或人工候选。
2. Granite bi-encoder 只对该 query 的 Top-1000 候选与 query 编码并按 cosine score 排名；
   BM25 rank 与 candidate-dense rank 用 RRF `k=60` 融合。
3. 同一 query 的全部系统共享相同 BM25 Top-1000 候选、文本、IDs 与 hashes。
4. 本实验不再声称评估“全语料独立 dense retrieval”；结论范围固定为
   “完整语料 BM25 初筛后的两阶段 neural/hybrid RAG”。
5. 正式规模仍为三个数据集各 400 题、十臂共 12,000 outputs；主表、消融和 scorer
   协议不缩减。
6. 被取消的全量 dense jobs `18684111/18684112` 只作为时间约束审计记录；partial
   artifacts 不进入任何正式系统。

---

## 1. 为什么需要 Experiment 05

Experiment 04 的执行和审计是有效的，但它不支持 `Ours` 在旧主指标 RAR 上优于 baseline。该实验仍作为完整的负结果保留，不覆盖、不删除，也不把它重新解释成正结果。

Experiment 05 只修正上一轮已经确认的两类设计不匹配：

1. **数据任务不匹配。** HotpotQA、MuSiQue 和 RGB 主要体现多跳、复杂组合或人为噪声场景；它们可以作为压力证据，但不应独自代表本项目所声称的普通开放域 RAG。
2. **评分目标不匹配。** 对完整解释性回答使用短答案 exact match/token F1，会因回答较长而产生额外 precision 惩罚；旧 RAR 又要求答案和引用同时全对，过于全有或全无，不能解释系统到底在哪个环节有价值。

因此，本实验保留上一轮合理的系统矩阵、模型、种子、公平预算和模块级消融，只更换主数据任务、Selector 的已决规则以及主评分体系。

---

## 2. 研究范围与两条主张

### 2.1 适用范围

本实验只支持以下范围内的结论：

> 面向 Wikipedia 类知识源的普通开放域事实问答，以及需要内联引用的事实型长答案 RAG。

它不声称覆盖所有领域、所有文档形态、所有多跳推理，也不把人工噪声、反事实或 unanswerable 压力场景当作新的主任务。

### 2.2 最多两条主张

**Claim A — 可靠回答。** 与使用同一 Hybrid Retriever 的普通 `Hybrid RAG` 相比，`Ours` 能减少有回答时的无证据事实，同时不以明显降低参考事实覆盖率或回答率为代价。

**Claim B — 可追溯回答。** 与 `Hybrid RAG` 相比，`Ours` 能提高“回答中出现且被有效引用支持”的参考事实比例，并保持或改善引用 precision/recall。

四个 baseline 的完整比较用于判断方法相对于常见替代方案的位置；三项模块级消融用于定位 Retriever、Selector 和 Generator 的贡献。内部 TRUE verifier 不单独做系统消融，因此论文也不得单独声称 TRUE 的独立因果贡献。

---

## 3. 三个且只有三个主数据集

| Dataset | Formal n | 普通 RAG 角色 | 选择理由 |
|---|---:|---|---|
| KILT–Natural Questions | 400 | 真实用户式开放域事实问答 | 问题源自真实搜索问题；KILT 版本同时提供统一知识源和 provenance，适合同时观察答案与来源 |
| KILT–TriviaQA | 400 | 普通 factoid 开放域问答的第二种问题分布 | 问题由 trivia 作者自然创作、证据后收集，主题广且不是为本项目人为制造的干扰测试 |
| ALCE–ASQA | 400 | 需要覆盖多个参考事实并给出内联引用的事实型长答案 | correctness 与 citation quality 可以分开评价，直接对应本系统的事实覆盖和引用可追溯性 |

官方来源：

- KILT 数据格式、NQ/TQA split 与统一 provenance：[facebookresearch/KILT](https://github.com/facebookresearch/KILT)
- Natural Questions 的真实用户问题来源：[Natural Questions 官方仓库](https://github.com/google-research-datasets/natural-questions)
- TriviaQA 的自然创作问题与证据设置：[TriviaQA 论文](https://aclanthology.org/P17-1147/)
- ASQA 的 correctness/citation 评估与官方检索资源：[Princeton NLP ALCE](https://github.com/princeton-nlp/ALCE)

### 3.1 明确不做的数据筛选

正式样本不得按以下条件筛选：难度、多跳数量、干扰文档数量、Selector 是否会删除、某系统是否答对、答案长度、引用数量或预估收益。不得注入人工 distractor、反事实证据或专门挑选 unanswerable 样本。

三个数据集逐个报告，不计算一个掩盖差异的 pooled 总平均。

### 3.2 历史暴露与确定性取样

这些公开数据在仓库历史开发中可能已有部分样本被使用，尤其是 ASQA。因此 Goal 1 必须先建立**历史暴露 ID 注册表**，只通过 ID、已有 artifact manifest 和 hash 工作，不查看候选正式样本的内容。

正式样本与 development fixture 的选择顺序固定为：

1. 建立每个数据集的 canonical ID 映射和历史暴露集合 `E`；
2. KILT–NQ/TQA 的 development pool 固定为各自 train split，按  
   `SHA256("experiment05-dev|<dataset>|<canonical_id>|20260822")` 升序取前 120；formal pool 固定为各自 dev split，并排除 `E`；
3. ASQA 先从历史暴露集合 `E` 按同一 dev hash 取前 120；若 `|E|<120`，再从 `all_ids - E` 按 dev hash 补足，补入的 ID 立即记为 development-exposed；
4. ASQA formal pool 为 `all_ids - E - development_ids`；三个 formal pool 均按  
   `SHA256("experiment05-formal|<dataset>|<canonical_id>|20260822")` 升序取前 400；
5. 保存 development/formal ordered IDs、两套 ordered-ID hash、原始文件 hash、历史暴露 registry hash 和选择脚本 hash；
6. development 与 formal 必须互斥；正式 ID 一旦冻结，不因内容、运行错误或结果变化而替换。

这套顺序保证每个数据集恰好有 120 条可重复的 revealed development fixtures，同时不会把补充的 ASQA development 样本误放入 formal set。

**硬门槛：** 每个数据集必须有至少 400 个未用于本项目方法开发的合格 ID。若 ASQA 或其他数据集不足 400，Goal 1 为 `FAIL`，不得静默换数据、缩小样本后继续，必须交回用户重新决定。

### 3.3 语料、索引与冻结候选检索

- KILT–NQ 与 KILT–TriviaQA 共用官方 KILT Wikipedia knowledge source；页面和 paragraph provenance 必须保留。
- ALCE–ASQA 使用官方 ALCE Wikipedia 100-word passage collection；不得使用 oracle Top-5 作为系统输入。
- 每个 corpus 构建/核验完整 BM25 index；Granite dense 只在每题 BM25 Top-1000
  候选内评分，不构建全语料 dense index。同一数据集的全部系统读取完全相同的 corpus
  snapshot、chunking、BM25 index、Top-1000 candidates 和 document IDs。
- KILT 的 page provenance 与 passage citation 建立冻结映射：`passage_id -> wikipedia_id + paragraph span`。
- 正式第一阶段是 full-corpus BM25 retrieval。不得把每题官方 gold passage、oracle
  passage 或人为候选池当作检索范围。
- 若完整 BM25 index、Top-1000 生成规则或 candidate-dense/RRF fingerprints 无法复现和
  hash 审计，Goal 1/2 `FAIL`；不得把两阶段结果写成全语料独立 dense retrieval。

---

## 4. 冻结系统矩阵：沿用 Experiment 04

### 4.1 五个主系统

| System | Retriever | Selector / Pruner | Generator |
|---|---|---|---|
| BM25 RAG | full-corpus BM25 Top10 | keep-all | Direct Granite |
| Hybrid RAG | full-corpus BM25 Top-1000 -> Granite dense candidate rank，与 BM25 RRF `k=60`，Top10 | keep-all | Direct Granite |
| Granite Rerank RAG | Hybrid RRF Top40 -> Granite reranker Top10 | keep-all | Direct Granite |
| Provence RAG | Hybrid RRF Top10 | official Provence | Direct Granite |
| **Ours** | Hybrid RRF Top10 | threshold-only NLI Selector | Granite GR-C + TRUE |

这五个系统分别回答：最基础语义 RAG、较强混合检索、相关性重排、外部上下文剪枝，以及完整三模块方法。它们不是因为 Experiment 04 的结果而临时更换。

### 4.2 三个且只有三个模块级消融

| Configuration | 相对 Full 的唯一变化 | 可解释的模块 |
|---|---|---|
| Full | 无 | 完整系统 |
| w/ BM25 Retriever | 两阶段 Hybrid RRF -> full-corpus BM25 Top10 | Retriever |
| w/o Selector / Keep-all Top10 | threshold-only Selector -> keep-all | Selector |
| w/ Direct Generator | GR-C + TRUE -> Direct Granite | 整个 Generator 模块 |

本轮不增加 Selector 内部阈值/门槛消融，也不增加 `GR-C without TRUE`。因此 `w/ Direct Generator` 只能说明完整 Generator 模块的贡献，不能拆分 GR-C 与 TRUE。

### 4.3 沿用的冻结模型

除 Selector 决策规则和统一回答 prompt 外，默认沿用 Experiment 04 已审计身份：

- dense embedder：`ibm-granite/granite-embedding-english-r2@47ea694...`；
- Granite reranker：`ibm-granite/granite-embedding-reranker-english-r2@d09d3d6...`；
- Provence：`naver/provence-reranker-debertav3-v1@ef49e23...`，threshold `0.1`；
- Generator base：`ibm-granite/granite-4.1-3b@c065040...`；
- Selector backbone/checkpoint：沿用 Experiment 04 seed-13 checkpoint 与 hashes；
- Ours Generator adapters：seeds `13/42/73`，全部报告，不挑最好 seed；
- runtime verifier：冻结 TRUE；
- scorer-only entailment judge：`lytang/MiniCheck-Flan-T5-Large@96eafd0...`，不得由 TRUE 替代。

完整 hash 从 Experiment 04 的 [model/config manifest](../04_frozen_three_module_system_evaluation_2026-08-21/artifacts/goal2_model_config_manifest.json) 继承；Goal 2 再记录本实验的实际本地文件 hash、代码 hash 和新 prompt hash。

---

## 5. Selector 的唯一新规则

冻结阈值：

```text
t = 0.9212157130241394
delete(candidate) iff
    harm_score >= t
    AND protect_score <= 1 - t
```

即只有同时满足“风险足够高”和“支持答案的可能性足够低”才删除。

必须遵守：

1. 删除 `max_delete=2`，不再设置任何最多删除条数；
2. 删除 `minimum_retained=7`，不再为凑数量强制保留；
3. Top10 中每条 evidence 独立应用同一双门槛；满足条件的全部删除，不满足的全部保留；
4. 任一 query 出现缺失、NaN、越界或无法解析的 Selector score 时，对该 query 执行 `keep-all` fail-safe，并记录错误；
5. 如果合法判断后十条全部删除，Generator 不接收伪造证据，也不强留最低分证据；它输出统一的 evidence-insufficient abstention；
6. 阈值不在三个 formal set 上调整，也不搜索新阈值网格。

Goal 2 的 development 检查只验证实现是否忠实于上述规则、空上下文是否可靠 abstain，并记录 Supporting Evidence Loss 与 Context Reduction；这些数值只作附录诊断，不用于偷偷恢复硬上限、挑选新策略或判断主张是否成立。

---

## 6. 公平运行协议

### 6.1 共同输入与预算

同一数据集内所有系统和消融必须使用：

- 相同 query IDs、corpus snapshot、chunk IDs 与 index；
- 相同最终 Top10 上限；
- 相同 `2,304` Generator input-token cap；
- 相同按排名取“最大完整 evidence 前缀”的 overflow policy；
- 相同最多 `256` new tokens；
- greedy decoding：`temperature=0`、`do_sample=false`、`top_p=1`；
- 相同 evidence 编号、标题/正文格式、内联引用格式 `[n]`；
- 相同数据集级回答说明；
- 相同失败分母、输出 schema 和 scorer。

NQ/TQA 的共同说明要求“直接回答，可有必要解释，每个事实句给出引用”，不强迫只输出一个词。ASQA 的共同说明要求“覆盖能够由证据支持的主要参考事实，形成简洁长答案，每个事实句给出引用”。同一数据集内五个系统和全部消融使用完全相同的说明。

### 6.2 统一输出契约

每行至少包含：

```text
query_id
answer_text
cited_evidence_ids
abstained
runtime_error
retrieved_evidence_ids
selected_evidence_ids
selected_evidence_records
presented_evidence_ids
presented_evidence_records
model/config/prompt fingerprints
```

三个 evidence 集合含义固定：

- `retrieved_evidence_ids`：Retriever 的最终 Top10；
- `selected_evidence_ids`：Selector/Pruner 后、token 截断前的集合；
- `presented_evidence_ids`：应用 2,304-token budget 后实际进入 Generator prompt 的完整 evidence 前缀。

ID 列表只用于集合与顺序审计，不能代替 Generator 实际看见的文本。`selected_evidence_records` 必须为 Selector/Pruner 输出的每条证据保存 stage ordinal、canonical `evidence_id`、精确的 post-pruning 文本 artifact URI、`SHA-256` 和 token 数；`presented_evidence_records` 必须进一步保存 prompt ordinal、canonical `evidence_id`、实际进入 prompt 的精确 post-pruning/post-budget 文本 artifact URI、`SHA-256` 和 token 数。URI 指向只读、内容寻址的 sealed artifact；正文不得写入控制台日志。`[n]` 固定解析为 prompt ordinal `n`，再映射到该 presented record 的 canonical ID 和精确文本。

UCR、VRFC、CP 和 CR 只能读取 `presented_evidence_records` 指向的精确文本；不得回读同 ID 的原始完整 passage，也不得用虽然被 Selector 保留、但实际未进入 prompt 的 passage 给答案补依据。附录诊断中，ER@10 使用 retrieved 原文；SELR 比较 retrieved 原文与 post-pruning `selected_evidence_records` 的精确文本；CRR 同时记录 selection reduction 与最终 presented-token reduction。Provence 即使保留原 passage ID，只要裁掉了支持句，该支持单元仍必须在 SELR 中记为丢失。

Ours 的内部 claim/TRUE trace 单独保存，不能成为主评分器的输入。主 scorer 只看每个系统都具备的 raw answer、引用、evidence ID 审计字段、`selected/presented_evidence_records` 的 sealed 精确文本和 scorer-only gold sidecar。

### 6.3 数据与 gold 隔离

- runtime bundle：只含 `query_id`、question 和 corpus/index 访问所需字段；不含 gold answer、required facts、gold provenance 或 support labels。
- scorer-only sidecar：含 required fact groups、answer aliases、provenance 及 scorer 所需映射。
- generation 入口没有读取 sidecar 的路径或权限。
- Goal 3 只生成并冻结七个主实验臂，Goal 4 只生成并冻结三个消融臂；在三个 dataset 的全部十个 formal arms、合计 12,000 条输出均完成 ID/hash 审计并冻结前，任何 formal scorer 都不得读取 sidecar。
- scorer-only sidecar 只能在 Goal 5 一次性解锁；Goal 3/4 不得计算、展示或据此推断任何正式主指标、诊断指标或科学结论。
- 正式执行前只允许程序打印 count、schema、ordered IDs hash 与文件 hash；日志不得打印问题、答案、evidence 正文或 labels。

### 6.4 错误、abstention 与零分母决策表

`substantive answer` 的机器定义为：无 runtime error、非空、不是冻结的 evidence-insufficient canonical response，且 scorer-only claim extractor 至少得到一个忠实 factual claim。只含“无法回答/证据不足”等元叙述的自然语言输出没有 factual claim，因此 `RR=0`。Goal 1 必须验证正常的单词/短语事实答案在拼接 question 后可以形成一个 factual claim，不能被误判成拒答。

| Case | RFC | VRFC | UCR | CP | CR | RR | 处理 |
|---|---:|---:|---:|---:|---:|---:|---|
| canonical abstention / empty / runtime failure | 0 | 0 | NA | 0 | 0 | 0 | 保留 query，不静默删除 |
| 非空但无忠实 factual claim 的空洞/元叙述 | 0 | 0 | NA | 0 | 0 | 0 | 按 non-substantive 处理 |
| substantive factual answer、无任何 citation marker | 正常算 | 0 | 正常算 | 0 | 0 | 1 | 不能靠不引用逃避 CP/CR |
| 非法或越界 citation ID | 正常算 | 不提供有效支持 | 正常算 | 计入 CP 分母且判 0 | 对应 claim 未覆盖 | 1 | 保留非法 link 与错误原因 |
| scorer infrastructure failure | 不记系统 0 | 不记系统 0 | 不记系统 0 | 不记系统 0 | 不记系统 0 | 不改 | 修复后重跑 scorer；无法修复则整个 bundle 技术 FAIL |

每个表必须同时报告 `n_total`、`n_answered`、`n_runtime_error`、`n_scorer_error`、`n_claims` 和 `n_undefined_ucr`。某 arm 在某 dataset 的 runtime error rate `>1%`，整个 dataset-arm bundle 技术无效；先修复同一冻结配置并整包重跑，不能用剩余样本凑数。

---

## 7. 最终且唯一的评分体系

### 7.1 评分单位与唯一判定过程

#### Reference fact group

- KILT–NQ/TQA：每题一个事实组 `{question, acceptable_aliases}`；官方多个答案字符串只是同一事实的 aliases，不因回答包含解释文字而扣 precision。
- ALCE–ASQA：每个官方 `qa_pair` 形成一个事实组 `{subquestion, short_answer_aliases}`。
- 每个 alias 使用冻结模板形成 canonical fact：  
  `For the question "<fact_question>", the answer is "<alias>".`
- 模板、Unicode/大小写 normalization 和 alias 去重代码在 Goal 1 hash 冻结；不能按 formal 内容人工改写。

#### Atomic factual claim

scorer-only claim extractor 的输入固定为 `question + raw answer`，而不是只看 answer。这样 `Paris [1]` 之类正常短答也能被重写为可验证的完整事实。引用 marker 的位置先保存，语义提取时再移除。

所有系统均从 raw answer 重新提取，不复用 Ours 的 runtime claim/TRUE trace。默认 extractor 为 adapters-disabled 的冻结 Granite base 加 scorer-only prompt；它只拆分和忠实改写，不作支持裁决。结构化提取失败时使用同一冻结 sentence-level fallback，不能只让某个系统丢 query。

抽取后按首次出现顺序去重：两个 claims 只有在 scorer-only MiniCheck 双向都判 entailment 时才归为同一语义事实。UCR 与 CR 使用去重后的 claims，防止重复书写同一事实稀释错误率。

#### Fact、claim 与 citation 的绑定

1. 对每个 reference fact group 与 extracted claim，FactMatch 的输入严格为  
   `premise = "Question: <fact_question>\nAnswer statement: <claim_text>"`、  
   `hypothesis = <canonical fact>`。至少一个具体 `claim_id` 对任一 alias hypothesis 通过才记 `matched=1`；不是只检查整段回答是否出现某个字符串。
2. RFC 是 coverage 指标，不另外处罚回答中的额外 claim；额外且无依据的事实由 UCR 处罚，错误引用由 CP/CR 处罚。
3. citation marker `[n]` 绑定到其所在原始句子的所有 extracted factual claims；句首且本句没有 factual claim 的 marker 是 invalid link。
4. CP 的最小计数单位是唯一 `(claim_id, evidence_id)`。同一 claim 重复引用同一 evidence 只计一次，并单独报告 duplicate count；重复引用不能稀释错误引用。
5. 越界、无法解析或不能按 prompt ordinal 映射到 `presented_evidence_records` 的 citation 保留为一个 invalid `(claim_id, raw_citation)` link，进入 CP 分母且判为 unsupported。
6. `validly_cited(claim)` 当且仅当至少一个绑定到该 claim 的合法 citation 映射到一条 `presented_evidence_record`，且 MiniCheck 判该 record 的精确 post-pruning/post-budget 文本支持该 claim；禁止改用同 canonical ID 的原始完整 passage。
7. VRFC 的某 reference fact 只有在**实现该 fact 的同一个 claim**满足 `validly_cited(claim)` 时记 1；“答案别处出现正确事实、另一处有无关引用”不能通过。
8. UCR 的 `supported(claim)` 当且仅当至少一条 `presented_evidence_record` 的精确文本支持该 claim；Selector 保留但因 token cap 未呈现的 evidence 不能使用。

MiniCheck 是唯一的 support/entailment verdict 模型；系统内部 TRUE 完全排除。Goal 1 必须验证：简短正确与扩展正确能同样匹配、错误实体/关系反转/否定不能通过、claim extractor 不把引用文本或元叙述当事实，并且同一 fact/claim/citation 的绑定符合上述规则。

### 7.2 六个主系统指标

| Code | 指标 | 定义 | 方向 | 回答的问题 |
|---|---|---|---:|---|
| RFC | Reference Fact Coverage | 回答中语义表达正确的 reference fact groups / 全部 reference fact groups | ↑ | 答案内容覆盖得对不对、全不全 |
| VRFC | Verified Reference Fact Coverage | 回答中表达正确，且至少有一个有效引用真正支持的 reference facts / 全部 reference facts | ↑ | 正确事实是否同时可追溯 |
| UCR | Unsupported Claim Rate | substantive answers 中，不能被任何 presented evidence 支持的去重 atomic factual claims / 全部去重 atomic factual claims | ↓ | 回答中有多少无依据事实 |
| CP | Citation Precision | 真正支持其对应 claim 的合法引用链接 / 全部引用链接 | ↑ | 给出的引用是否准确 |
| CR | Citation Recall | 至少有一个有效支持引用的 factual claims / 所有需要引用的 factual claims | ↑ | 应该引用的事实是否都被引用 |
| RR | Response Rate | substantive、非 abstention 回答数 / 全部 queries | ↑ | 系统是否通过大量拒答换取表面安全 |

逐题公式：

```text
RFC_q  = matched_reference_facts / all_reference_facts
VRFC_q = matched_and_validly_cited_reference_facts / all_reference_facts
UCR_q  = unsupported_unique_atomic_claims / all_unique_atomic_claims  # answered only
CP_q   = supporting_unique_claim_evidence_links / all_unique_or_invalid_links
CR_q   = validly_cited_factual_claims / citation_required_claims
RR_q   = 1 if substantive answer else 0
```

RFC、VRFC、CP、CR、RR 的主表值为 all-query macro average；abstention/runtime failure 按 §6.4 记分。**主 UCR 固定为 dataset-level claim micro：**

```text
UCR_arm = sum_q unsupported_unique_claims_q
          / sum_q all_unique_claims_q
```

abstention 对 UCR 分子和分母均贡献 0，但它仍使 RR/RFC/VRFC 为 0；因此任何 UCR 改善只有在 RFC 与 RR 同时通过 non-inferiority gate 时才可支持 Claim A。附录另报 answered-only query-macro UCR、双方都回答的 common-answer sensitivity 和总 claim 数；不得把附录版本换成主版本。

### 7.3 附录诊断信息（不是主评价指标）

| Code | 诊断信息 | 唯一解释 | 方向/角色 | 对应模块 |
|---|---|---|---|---|
| ER@10 | Evidence Recall@10 | reference facts 中有多少能在 Retriever Top10 找到支持 passage | ↑ 越高表示检索覆盖越好 | Retriever |
| SELR | Supporting Evidence Loss Rate | Retriever 已找到、但其支持内容在 post-pruning selected 精确文本中不再存在的 `(reference fact, supporting evidence)` 单元 / Retriever 已找出的同类单元 | ↓ 越低表示误删越少 | Selector/Pruner |
| CRR | Context Reduction Rate | Selector 删除的 tokens / Selector 输入 tokens；另记录 token cap 后未呈现比例 | 仅描述删了多少；不定义越高或越低更好 | Selector |

`SELR` 的分母只包含 Retriever 已经找到的支持单元，因此不会把 Retriever 漏证据错误算到 Selector 头上。其保留/丢失判断基于 `selected_evidence_records` 的精确 post-pruning 文本，不基于 passage ID 是否仍在列表中；因此 Provence 保留 ID 但裁掉支持句时会正确计为丢失。没有检索到支持单元的 query 对 SELR 为 `NA`，但该 query 仍保留在全部六个主指标的共同分母中。

三项诊断来自同一次完整系统运行，只放入 appendix diagnostic table，用来解释主结果，不参与 Claim A/Claim B、Goal PASS 或系统优越性判断。Selector 是否有实际价值，只由 `Full vs w/o Selector / Keep-all` 在六个主指标上的配对结果决定；SELR 解释是否误删，CRR 只确认实际改变了多少上下文。

Retriever 的详细 MRR/nDCG、Selector 的 harm/protect AUROC/calibration、Generator splitter/TRUE 内部准确率继续属于既有模块开发实验，本轮不重跑也不塞进主表。

### 7.4 外部裁判、locked validation 与公平性

- 系统内部 TRUE 只参与 `Ours` 的运行，不得评分任何系统。
- fact match、fact/evidence support、citation support 和 unsupported claim 的 verdict 使用 scorer-only MiniCheck；其 revision/hash 在 Goal 1 冻结，且不得用于任何 evaluated arm 的训练或 formal output 选择。
- scorer-only atomic claim extractor 不读取 Ours runtime trace，所有系统从 `question + raw answer` 重新处理。
- 任何 judge threshold、fact template、citation mapping、dedup 和 invalid-output rule 必须在 revealed development/synthetic calibration fixtures 上确定，然后在独立 locked validation fixtures 上一次验收。

Goal 1 的 locked validation 最少包含：

| Cell | Locked items | PASS gate |
|---|---:|---|
| FactMatch：简短正确、扩展正确、部分正确、错误实体、否定/关系反转，类别平衡 | 160 fact-claim pairs | macro-F1 ≥0.90；positive recall ≥0.90；negative specificity ≥0.90 |
| Evidence support：支持/不支持平衡，覆盖三数据集文体 | 160 claim-passage pairs | macro-F1 ≥0.85；recall 与 specificity 均 ≥0.85 |
| Claim extraction：短答、Direct-like、GR-C-like、ASQA long-form | 60 answers / 至少 180 gold claims | span coverage recall ≥0.90；faithful-claim precision ≥0.95 |
| Citation/abstention/error contract | 80 deterministic cases | 100% exact；任一失败即 FAIL |

人工/规则 gold 标签不得显示 system identity。calibration 与 locked validation IDs/fixtures 分离；locked 集只用于验收，不用于挑 judge checkpoint。

公平 gate：

1. 语义等价短答与扩展回答在 RFC、VRFC、UCR、CP、CR 上的差异分别不超过 `5pp`；
2. Direct-like 与 GR-C-like 文体的 FactMatch/support false-negative-rate 差分别不超过 `5pp`；
3. 重复事实与重复引用去重测试 100% 通过；
4. 只增加有证据的正确解释不能使 UCR 变差；加入一个无证据事实必须使 UCR 严格变差；
5. 简短事实片段必须被识别为 substantive answer；自然语言证据不足元叙述必须为 RR=0。

任一 gate 失败，Goal 1 留在 `FAIL`；只能用 calibration fixtures 修复规则后重新建立新的 locked fixture version，不能在 formal 上调 scorer。

### 7.5 仅作补充的官方指标

为和已有文献对齐，附录可以报告：

- KILT–NQ/TQA：official EM/token F1、page-level R-Precision 和可计算的 KILT 联合指标；
- ASQA：official STR-EM/EM Recall 及 ALCE citation metrics。

它们不是 Experiment 05 的主 claim gate。尤其不能再用“完整长回答是否等于短 gold answer”作为核心正确性结论，也不再使用旧 RAR 作为主指标。

---

## 8. 结果判断和统计

### 8.1 技术 PASS 与科学结论必须分开

- **Goal PASS：** 数据、运行、评分、覆盖率和审计符合协议；它不要求 `Ours` 获胜。
- **Scientific supported/not supported：** 只由预先写明的指标和置信区间决定。

即使结果为负，只要执行有效，Experiment 05 仍可 `FINAL PASS`，并诚实报告主张不成立。

### 8.2 主比较

首要比较是 `Ours - Hybrid RAG`，因为二者使用同一 Hybrid Retriever；其他三种 baseline 是重要的完整背景比较。

固定判断规则：

**Claim A supported** 当且仅当：

1. 至少两个数据集上 `ΔUCR = UCR(Ours) - UCR(Hybrid) < 0`，Bonferroni-adjusted two-sided `98.33%` paired percentile-bootstrap CI 的上界小于 0；
2. 未达到 superiority 的剩余数据集，`ΔUCR` 的 ordinary 95% CI 上界不得高于 `+5pp`；
3. 三个数据集上 RFC 的 `Ours - Hybrid` one-sided 95% bootstrap 下界均不低于 `-5pp`；
4. 三个数据集上 RR 的 `Ours - Hybrid` one-sided 95% bootstrap 下界均不低于 `-5pp`。

**Claim B supported** 当且仅当：

1. 至少两个数据集上 `ΔVRFC = VRFC(Ours) - VRFC(Hybrid) > 0`，Bonferroni-adjusted two-sided `98.33%` paired percentile-bootstrap CI 的下界大于 0；
2. 未达到 superiority 的剩余数据集，`ΔVRFC` 的 ordinary 95% CI 下界不得低于 `-5pp`；
3. 三个数据集上 CP 与 CR 的 `Ours - Hybrid` one-sided 95% bootstrap 下界均不低于 `-5pp`。

Claim A 与 Claim B 分别使用下面的唯一三值规则，不得在结果出来后自由选择标签：

- `SUPPORTED`：相关 claim 上至少 `2/3` 个数据集通过其 superiority CI，且该 claim 的全部 non-inferiority/harm gates 均通过；
- `MIXED`：恰好 `1/3` 个数据集通过相关 superiority CI，另外两个数据集的相应 non-inferiority gate 通过，且该 claim 的全部其他 harm gates 均通过；
- `NOT SUPPORTED`：`0/3` 个数据集通过相关 superiority CI，或任一相应 non-inferiority/harm gate 失败、`UNDEFINED` 或 `NOT ESTIMABLE`。

对 Claim A，superiority 指上文 `ΔUCR` 的 adjusted CI gate，non-inferiority/harm gates 指剩余数据集的 UCR `+5pp` 上界以及三个数据集的 RFC、RR `-5pp` 下界。对 Claim B，superiority 指上文 `ΔVRFC` 的 adjusted CI gate，non-inferiority/harm gates 指剩余数据集的 VRFC `-5pp` 下界以及三个数据集的 CP、CR `-5pp` 下界。三种情况已穷尽；不得修改指标、margin、标签定义或主 baseline。

`5pp` 是执行前固定的 maximum practically acceptable loss：本项目不接受为了减少无依据事实或改善引用而牺牲超过五个百分点的事实覆盖、回答覆盖或引用覆盖；它也延续此前 Generator citation regression guard 的量级。它不是根据 Experiment 05 结果或显著性临时选择的 margin。

### 8.3 统计方法

- 每个 dataset 独立报告，不 pooled。
- Ours 点估计是 seeds `13/42/73` 各自 dataset-level metric 的算术均值；sample SD 是三个 seed-level metric 之间的样本标准差。四个确定性 baseline 各运行一次。
- RFC/VRFC/CP/CR/RR：每次 bootstrap 对完整 query ID 列表有放回抽样；同一抽样同时用于 baseline 和三个 Ours seeds；先算每个 seed 的 `Ours_seed - baseline` 差，再对三 seed 差值取算术均值。
- UCR：每次 bootstrap 同样抽完整 query IDs；对 baseline 和每个 Ours seed 分别在本次样本内重算  
  `sum(unsupported claims) / sum(all claims)`；再将三个 `Ours_seed_UCR - baseline_UCR` 差值平均。abstention 的 UCR 不填 0，也不在逐题层面直接做 NA-ignoring seed average。若某次 resample 的任一 arm 没有 factual claim，该 replicate 记为无效并保存原因。若原始 arm 的 claim 总数为 0，UCR/CI 记 `UNDEFINED`、Claim A 自动 `NOT SUPPORTED`；若无效 bootstrap replicates 超过 1%，CI 记 `NOT ESTIMABLE`、Claim A 自动 `NOT SUPPORTED`。这两种都是可报告的科学结果，不导致技术 FAIL。只有零 claim 来自 claim extractor/scorer 基础设施错误时，才按 §6.4 技术 FAIL 并修复。最终审计保存无效 replicate 数量和比例。
- 附录另外报告双方都回答的 common-answer subset UCR sensitivity，但它不替换上述主 UCR。
- paired percentile bootstrap `10,000` 次，seed `13`；ASQA 以 query 为重采样单位，不能把同题多个 facts 当成独立样本。
- Table 2 使用 seed-13 Full 与三个单模块消融逐题配对；Full 直接复用 Goal 3，不重新生成。
- 同时报 effect size、CI、n、claim count 和 error count；Table 1 附录列出三个 seed 各自的 effect，禁止用 seed mean 掩盖明显方向相反的 seed。
- 所有 CI 明确标注为“以冻结的三个 Generator seeds 为条件”；三个 seeds 的 SD 描述已训练 checkpoints 的波动，不声称完整估计训练随机性。

### 8.4 结果表

1. **Table 1 — Main system comparison：** 每个数据集一个 panel；五个系统，列为 RFC、VRFC、UCR、CP、CR、RR。
2. **Table 2 — Module ablation：** 每个数据集一个 panel；Full + 三个模块消融，列为六个端到端指标。
3. **Appendix diagnostic table：** ER@10、SELR、CRR；与 Table 1/2 使用同一逐题结果，不另起实验，不参与主张 gate。
4. **Appendix：** 官方 EM/F1/KILT/STR-EM、错误率、answer/claim counts 和 scorer audit。

不定义一个新的加权总分，也不以单一“高质量答案数”替代这些维度。若需要给非技术读者一句总结，只能根据上述预注册 gate 写 supported/mixed/not supported。

---

## 9. 接力式执行：五个 Goal

### Goal 0 — 计划审查与交接（已 PASS）

**只做：** 阅读、修改、独立复审和冻结本计划，并交接到新对话。  
**状态：** `PASS / HANDOFF READY`。  
**PASS 证据：** 用户授权写入已讨论修改并要求无阻断后交接；独立 agent 首审发现的三项阻断已修复，完整复审结论为 `PASS`。  
**边界：** 该 PASS 只代表计划可交接，不代表实验已经启动。Goal 1–5 仍全部 `NOT STARTED`；只有用户在交接后的新对话中明确要求开始执行，Goal 1 才能成为唯一 active goal。

### Goal 1 — 数据、语料、隔离与 scorer readiness

**唯一目的：** 证明三套数据可以无泄漏地运行，并证明新指标能够公平评分短答和解释性长答。

**只做：**

1. 固定官方数据/语料版本、license、原始 hash；
2. 建立历史暴露 ID 注册表，按第 3.2 节确定性选出 `400/400/400` formal IDs；
3. 物理分离 gold-free runtime manifest 与 scorer-only sidecar；
4. 构建/核验 KILT 与 ALCE corpus 的完整 BM25 indices、provenance mapping，并冻结
   `BM25 Top-1000 -> Granite candidate scoring -> RRF` 的版本、参数和审计契约；
5. 使用分离的 calibration/locked fixtures，按 §7.4 的固定样本数与数值 gate 验证六个主指标；同时用 deterministic contract tests 验证 ER@10、SELR、CRR 的计算定义，但不为三项附录诊断设置系统成败 gate；
6. 验证 presented-evidence 边界、短答/长答与 Direct/GR-C 文体公平、空答、重复/非法引用、runtime/scorer 失败、UCR-NA 和共同分母规则；
7. 只用 count/schema/ordered-ID hash/file hash 对 formal bundles 做机器核对。

**不得做：** 不运行任何 formal Retriever、Selector、Generator、TRUE 或 MiniCheck；不打印或人工查看 formal question、answer、evidence、required facts 或 provenance labels。

**交付：** data/corpus/index manifest、historical-exposure report、runtime/sidecar isolation report、scorer validation、自动测试证据、`PASS/FAIL`。

**PASS：** 三套 development IDs 均为 120、三套未暴露 formal IDs 均为 400，且两者互斥；runtime 无 gold；完整 BM25 indices/provenance 与两阶段检索契约可复现；§7.4 的全部 locked-validation 与公平 gate 通过；formal 内容零暴露、零模型调用。

**预计：** 快速修订后剩余 1–3 小时；完整 BM25 indices 已构建完成。

### Goal 2 — 新 Selector、五系统接线与 development smoke

**依赖：** Goal 1 `PASS`。

**唯一目的：** 在不使用 formal 数据的前提下，证明全部冻结 arm 可以公平运行。

**只做：**

1. 实现 threshold-only 删除、移除 `max_delete`/`minimum_retained`、加入 all-delete abstention；
2. 冻结 10 个运行臂：4 baselines + Ours seeds 13/42/73 + 3 ablations；
3. 冻结模型、代码、index、prompt、token budget、output schema 和 scorer hashes；
4. 在每个数据集按 dev hash 排名的前 5 条 revealed fixtures 上做 10-arm 共同 smoke，共 `10 × 3 × 5 = 150` 个 smoke outputs；
5. 在全部 `3 × 120` development fixtures 上只运行 Retriever/Selector 诊断，不运行 10-arm Generator 网格；
6. 做规则 property tests、gold-leak test、presented-evidence/output/parser test 和失败分母 test；其中必须包含“canonical evidence ID 仍保留、但 Provence 裁掉唯一支持句”的确定性案例，并验证 SELR 记为丢失、主 scorer 不能回读原始完整 passage；
7. 报告 development ER@10/SELR/CRR 与 all-delete rate，但不据此在 formal 前挑新阈值、恢复数量上限或判断主张成立。

**不得做：** 不读取或运行任何 formal query；不根据 development 的系统胜负换 baseline、指标或数据。

**交付：** 10 frozen configs、model/code/prompt/index manifest、development smoke report、自动测试、`PASS (= READY)/FAIL`。

**PASS：** 10/10 arms 在三个数据集的固定 5-query slice 完成共同 smoke；Selector 在 360 条 dev queries 上的删除集合与双门槛谓词逐条相等；空 context 一律 abstain；`presented_evidence_records` 的 ordinal、artifact text hash、token 数与实际 prompt byte/token audit 一致；Provence 的“保留 ID、裁掉支持句”测试通过；所有输出能被同一 scorer 读取；formal output 目录仍为空。

**预计：** 6–12 小时。

### Goal 3 — 五系统正式主运行（Table 1 输入）

**依赖：** Goal 2 `PASS (= READY)`。

**唯一目的：** 生成并冻结五系统在三个普通 RAG 数据集上的全部正式主运行输出，不接触 formal gold。

**运行量：**

- 4 baselines × 1,200 queries = `4,800` outputs；
- Ours 3 seeds × 1,200 queries = `3,600` outputs；
- 合计 `8,400` 正式 outputs。

**顺序：** 每个 dataset 的七臂全部生成 -> 核对 count/schema/ordered IDs/hash/runtime errors -> 冻结 generation outputs。此阶段 scorer-only sidecar 始终锁定，不计算或展示任何正式分数。

**交付：** 七个主实验臂的完整逐题 generation artifacts、presented-text sealed artifacts、ID/hash/runtime-error 审计、供 Goal 5 生成 Table 1 的冻结输入、`PASS/FAIL`。

**PASS：** 7 arms × 3 datasets 覆盖全部 frozen IDs；每个 bundle 错误率 ≤1%；无 seed selection；每条输出及其 presented 精确文本可由 manifest/hash 追溯；sidecar 未解锁且没有正式分数泄露。科学结果尚不在本 Goal 判断。

**预计：** 20–36 小时，不含 Goal 1 的首次索引构建。

### Goal 4 — 三个模块级消融正式运行（Table 2 输入）

**依赖：** Goal 3 `PASS`。

**唯一目的：** 生成并冻结三个模块级消融的全部正式输出，为之后判断 Retriever、Selector、Generator 模块贡献提供输入；本 Goal 不接触 formal gold。

**运行量：** 三个新消融 × 1,200 queries = `3,600` 新 outputs。`Full` 复用 Goal 3 的 Ours seed-13，禁止重新生成。

**交付：** 三个消融的完整逐题 generation artifacts、presented-text sealed artifacts、ID/hash/runtime-error 审计、供 Goal 5 生成 Table 2 的冻结输入、`PASS/FAIL`。

**PASS：** 每个配置只替换一个模块；其余 fingerprints 与 Full 相同；全部 IDs 完整；Full 复用一致性测试通过；Goal 3/4 共十臂、12,000 条输出全部冻结；sidecar 仍未解锁且没有正式分数泄露。

**预计：** 10–20 小时。

### Goal 5 — 统计、审计与最终报告

**依赖：** Goal 4 `PASS`。

**唯一目的：** 将冻结结果变成可直接用于论文、且能被逐项复核的证据。

**只做：**

1. 核对 Goal 3/4 的十臂、12,000 条 generation outputs 全部冻结后，一次性解锁 scorer-only sidecar；
2. 统一评分，并从逐题文件独立重算 mean、sample SD、macro/micro metrics 和 10,000 次 paired bootstrap；
3. 核对 IDs、hash、共同分母、abstention、missing/error 和 UCR denominator；
4. 生成 CSV、JSON、Markdown、LaTeX 的 Table 1/Table 2、appendix diagnostic table 及审计 manifest；
5. 严格按第 8.2 节的唯一三值规则分别写 Claim A/B 为 `SUPPORTED / MIXED / NOT SUPPORTED`；
6. 明确 Experiment 04 压力结果与 Experiment 05 普通 RAG 结果各自的适用范围。

**不得做：** 不生成新答案，不调方法，不补内部消融，不换指标，不挑数据/seed。

**交付：** final report、只含六主指标的 Table 1/2、appendix diagnostic table、machine-readable results、bootstrap、final audit、`FINAL PASS/FAIL`。

**预计：** 3–6 小时。

---

## 10. Goal 接力和失败处理

Goal 0 已 `PASS`，但这只代表交接就绪。只有用户在交接后的新对话中明确发出“开始执行”指令后，以下执行接力才生效：

```text
Goal 1 active
  ├─ FAIL -> 留在 Goal 1 修复和复测；不得进入 Goal 2
  └─ PASS -> 冻结 Goal 1，立即把 Goal 2 设为唯一 active goal

Goal 2 PASS -> 自动接 Goal 3
Goal 3 PASS -> 自动接 Goal 4
Goal 4 PASS -> 自动接 Goal 5
Goal 5 FINAL PASS/FAIL -> 整个实验停止
```

规则：

1. 同时只能有一个 active execution goal；
2. Goal 内普通可修复错误自动修复、复测，不因中间步骤停下等用户；
3. 每个 Goal 产出 PASS/FAIL 和审计材料，但 PASS 后不等待新的启动指令，自动接力下一 Goal；
4. 只有需要用户改变科学范围的失败才停止，例如未暴露 ASQA 不足 400、必须更换数据集、必须改变主指标或需要新增方法；
5. Goal 5 是唯一终点，不自动创建 Goal 6；
6. 用户已在交接后的新对话中明确发出“开始执行”，并随后授权 v4 快速方案；自动接力已生效，当前按 Goal 3 边界执行。

---

## 11. 正式运行规模与保存位置

| Formal group | New outputs |
|---|---:|
| Goal 3：四个 baseline | 4,800 |
| Goal 3：Ours 三 seeds | 3,600 |
| Goal 4：三个单模块消融 | 3,600 |
| **Total** | **12,000** |

建议路径：

```text
docs/full-flow/experiments/05_general_rag_reliability_traceability_evaluation_2026-08-22/
  PLAN.md
  TRACKER.md
  PLAN_SELF_REVIEW.md
  artifacts/
  reports/
  results/

runs/experiment05/<dataset>/<configuration>/<seed>/
```

Git 只保存可审计的小型 manifest、逐题指标、汇总表和报告；大模型缓存、corpus index、raw generations 保存在服务器，并以 hashes/URI 指向。

---

## 12. 本实验明确不做

- 不覆盖、删除或重写 Experiment 04 的负结果；
- 不把 HotpotQA、MuSiQue、RGB 放入新的主表；
- 不添加刻意噪声、反事实、多跳困难或 unanswerable stress slice；
- 不在 formal 结果出现后更换 v4 冻结的四个 baseline；v4 唯一预结果变更是以
  `BM25 RAG` 替代无法按时完成的全语料 `Dense RAG`；
- 不增加 TRUE 内部消融或 Selector 内部消融；
- 不重训 Retriever、Selector 或 Generator；
- 不搜索 selector threshold，不恢复任何 delete/keep 数量上限；
- 不用系统内部 TRUE 给自己评分；
- 不以 exact-match-only、长答案 token precision 或旧 RAR 作为核心成败标准；
- 不把三个数据集合成一个总平均；
- 不在用户于交接后的新对话中明确发出“开始执行”指令前启动 Goal 1。
