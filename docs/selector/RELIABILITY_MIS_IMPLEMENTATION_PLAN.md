# ReliabilityRAG-inspired MIS Selector 代码改造执行计划

**日期：** 2026-08-08  
**适用代码基线：** refactor/three-module-baseline @ 0d39d260bac84df7ad33ff6241a117e8d1e76801  
**计划性质：** 代码实施计划，不是实验方案  
**方法定位：** ReliabilityRAG-inspired 项目适配，不是论文逐行复现  
**目标 Selector 名称：** reliability-mis

## 1. 最终决定

本次不继续扩展现有 Graph1/Graph2，也不复现一个完整的新 RAG 系统。只新增一个独立的 passage-level、ReliabilityRAG-inspired Selector：

    Hybrid RRF 返回候选 passage
              ↓
    每个 passage 单独回答当前问题
              ↓
    DeBERTa 判断两份回答是否明确矛盾
              ↓
    构造“矛盾图”，求最大无冲突集合 MIS
              ↓
    返回集合中原 passage 的 evidence_id
              ↓
    现有 Generator 原样工作

这里有两个容易混淆的概念：

- DeBERTa 是模型，负责判断两份回答是否矛盾。
- MIS（Maximum Independent Set，最大独立集）不是模型，而是一个确定性的图算法。项目版借用它的“内部不能存在矛盾边”原则，再增加现有 max_selected 容量和独立来源目标。

这版 Selector 的唯一职责是：

> 只有发现明确语义矛盾时才过滤候选；没有明确矛盾时完全退化为当前 TopK。

它不判断“证据是否足够”，不控制 Generator 拒答，不使用参数知识回答，不改写证据文本。

## 2. 为什么这次方案可以直接接入现有项目

现有接口已经足够：

- 输入：Query + CandidateSet。
- 输出：SelectionResult，只含原候选的 evidence_id、selection_score、selection_rank。
- Pipeline 会用 evidence_id 找回 Retriever 返回的原始 passage，再交给 Generator。

因此 MIS 只需决定“保留哪些已有 evidence_id”，不需要给任何公共契约加字段。

当前已有能力也可以复用：

- selector/extraction.py 已支持每个 passage 单独回答问题。
- selector/answer_norm.py 已支持识别 NONE 等无效短答案。
- selector/top_k.py 已提供安全回退结果。
- granite optional dependencies 已包含 torch 和 transformers。
- composition.py 已是 Selector 的统一注册入口。

## 3. 严格范围

### 3.1 允许新增

- src/evidence_rag/selector/reliability_mis.py
- src/evidence_rag/selector/deberta_contradiction.py
- tests/selector/test_reliability_mis.py
- tests/selector/test_deberta_contradiction.py
- tests/pipeline/test_reliability_mis_pipeline.py

### 3.2 允许最小修改

- src/evidence_rag/composition.py
  - 加一个新的 Selector 注册分支。
  - 加 fake contradiction scorer 的依赖注入口，供离线测试使用。
- tests/pipeline/test_selector_registration.py
  - 增加新 Selector 的构建与参数校验测试。
- tests/typecheck.py
  - 增加新类满足现有 Selector Protocol 的静态检查。

### 3.3 明确禁止修改

- src/evidence_rag/retriever/**
- src/evidence_rag/generator/**
- src/evidence_rag/pipeline/service.py
- src/evidence_rag/contracts/models.py
- src/evidence_rag/contracts/protocols.py
- src/evidence_rag/contracts/validation.py
- 现有 Graph1 文件：gated.py、clusters.py、corroboration.py、coverage.py
- selector/top_k.py
- pyproject.toml
- 当前默认配置和已有实验配置

如果实现需要修改上述冻结文件，先停止，不要继续写代码；这说明设计越过了本计划的模块边界。

## 4. 我们采用论文的什么，不采用什么

| 内容 | ReliabilityRAG 官方方法 | 本项目第一版 |
|---|---|---|
| 判断单位 | 每篇检索文档单独回答 | 每个 EvidenceCandidate passage 单独回答 |
| 冲突判断 | answer-to-answer DeBERTa NLI | 相同 |
| 建边条件 | P(contradiction) ≥ 0.5 | 相同，固定 0.5 |
| 图算法 | 对全部有效文档求 exact MIS | 精确求“容量不超过 max_selected、独立来源最多、内部无矛盾”的集合 |
| 并列处理 | 优先原检索排名更高的完整 MIS | 使用当前 TopK 的规范顺序决胜 |
| NLI 错误模拟 | 源码有随机 err 翻边 | 删除；生产代码不得有随机翻边 |
| 无答案节点 | 删除 I don't know | 保留但不建边，防止误删多跳桥接证据 |
| 最终回答 | 官方类内继续调用 LLM | 不移植；只返回 SelectionResult |
| 候选数量 | 论文主要设置 k=10 | 项目默认 top_n=20，硬上限 20 |
| source reliability | 论文把检索序当 reliability signal | 当前 TopK 顺序只作 tie-break；不宣称 RRF score 等于来源可信度 |
| 独立来源 | 文档节点直接计数 | passage 节点不合并；目标先最大化不同 source_parent 数 |

### 4.1 特别说明：这不是句子 claim 图

第一版不做以下工作：

- 不把 passage 切成句子 claims。
- 不建立 support / neutral 多关系图。
- 不做网页域名信誉。
- 不做外部反证检索。
- 不训练新模型。
- 不把 source_parent_id 预合并成图节点，也不做简单多数票；只用它计算独立来源覆盖目标。
- 不采用 Graph2 relation training 结果。

ReliabilityRAG 的核心是“逐 passage 的独立答案之间是否矛盾”，不是“句子 claims 的复杂关系图”。

### 4.2 特别说明：使用 source_parent_id，但绝不预合并节点

项目的 gold needle 与 counterfactual twin 经常共享同一个 parent。若先按 parent 合并，可能在做 NLI 之前把真、假 passage 塞进同一个节点，使真正需要发现的矛盾消失。

因此第一版：

- 一个 EvidenceCandidate 仍对应一个图节点，真/假 passage 可以在同一 parent 内建立矛盾边。
- 通过现有 SOURCE_PARENT_INDEX 取得 document_id 到 source_parent_id 的映射。
- source_parent_id 不决定是否建边，也不直接删除任何 passage。
- 选择目标首先最大化所代表的不同 source_parent 数，防止同一网页切出的多个 chunks 被当成多份独立支持。
- 若某个 document_id 在 sidecar 中没有记录，使用 document_id 自身作为保守 parent key，并在内部事件中记录 unresolved_parent_count。
- 不做“一个 parent 只留一个 passage”的预合并；同一 parent 的多个非矛盾 chunks 仍可在容量允许时保留，用于完整证据覆盖。

这是项目适配，不是 ReliabilityRAG 原论文的一部分。

## 5. 冻结的行为规格

这一节是实现时的唯一真相。不要边写代码边改规则。

### 5.1 固定常量

| 项目 | 固定值 |
|---|---|
| Selector config name | reliability-mis |
| 默认 top_n | 20 |
| top_n 硬上限 | 20 |
| contradiction threshold | 0.5 |
| NLI model | MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli |
| NLI revision | b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7 |
| NLI max length | 512 |
| NLI batch size | 32 |
| answer passage_chars | 600，沿用现有抽取器 |
| answer generation | temperature=0，do_sample=False |
| parametric answer | 禁用，use_parametric=False |
| prompt version | mis-answer-v1 + mis-statement-v1 |
| independent-source mapping | SOURCE_PARENT_INDEX；缺少 sidecar 时配置失败 |
| random seed / err | 不存在；算法路径完全确定 |

配置层只允许 top_n。阈值、模型与 revision 不开放为实验调参面，避免同事在代码实现阶段继续搜索方法。

### 5.2 输入检查

ReliabilityMISSelector.select 必须沿用现有 Selector 的检查：

1. query.query_id 与 candidates.query_id 不一致：抛 ValueError。
2. max_selected <= 0：抛 ValueError。
3. 空 CandidateSet：直接返回 TopKSelector 的空结果，不调用 LLM 或 NLI。
4. top_n 不在 1 到 20：构造时抛 ValueError。
5. reliability-mis 未配置 SOURCE_PARENT_INDEX：构建时抛 ValueError；这是缺少必需数据，不是推理后端故障。

这些是调用或配置错误，不能伪装成 TopK fallback。

### 5.3 候选顺序

1. 唯一规范顺序必须与当前 TopKSelector 完全相同：按 (-retrieval_score, evidence_id)。
2. 不依赖输入 tuple、dict 或 set 的迭代顺序。
3. selection_score 始终沿用原 candidate.retrieval_score。
4. 最终 SelectionItem 的 selection_rank 从 1 连续编号。
5. retrieval_rank 保留在原 EvidenceCandidate 中用于审计，但第一版不再用它建立第二套排序。

无矛盾与后端失败分支必须直接返回：

    TopKSelector().select(query, candidates, max_selected)

不要重新实现一份“看起来相同”的 TopK。

### 5.4 独立答案

对前 min(top_n, 候选数) 个 passage 分别调用现有 AnswerExtractionEngine：

- 每次 prompt 中只能放一个 passage。
- 使用现有 EXTRACT_PROMPT。
- 在新文件中固定记录 prompt version；若以后改 EXTRACT_PROMPT 或 statement 模板，必须升级 version，不能静默改变。
- use_parametric=False，绝不额外让模型凭自身知识回答。
- 原始 passage、answer 或其他候选不能互相泄漏。
- 对输出 strip；仅移除可选的开头 “Answer:”。

新增一个 Selector 私有函数 is_unanswerable_answer，至少识别：

- 空字符串
- NONE
- unknown
- I don't know
- I do not know
- not in the passage
- not in the evidence

不要直接扩大 answer_norm.py 的全局 STOPWORDS，因为那会改变现有 Graph1 行为。

### 5.5 NLI 输入

只对两端都有有效答案的 i < j 组合打分。每个答案先变成完整陈述：

    The answer to the question: {question}
    is {answer}.

然后：

- statement i 作为 premise。
- statement j 作为 hypothesis。
- 为保持与论文 answer-pair 构造接近，第一版只计算这个固定方向，不增加双向平均或 max；整体方法仍属于项目适配，不声称论文等价。
- 三分类模型只读取 contradiction probability；entailment 和 neutral 都不建边。
- 当且仅当 probability >= 0.5 时，在 i、j 之间加一条无向边。
- 不按字符串不等直接建边。
- 不建立 support 边。

若窗口为 20：

- 最多 20 个独立答案调用。
- 最多 190 对 NLI。
- NLI 按固定顺序分 batch，输出数量必须与输入 pair 数完全相等。

### 5.6 NONE 与“没有明确矛盾”

NONE、neutral、entailment 的共同含义是“没有足够依据建立矛盾边”，不是“证据错误”。

固定处理：

- NONE 节点不参与 NLI pair，但作为无边节点保留。
- 图中一条 contradiction 边都没有：直接返回当前 TopK，结果对象必须完全相等。
- 全部答案都是 NONE：同样返回 TopK。
- 不允许仅凭“缺少第二来源支持”删除候选。
- Selector 不返回空集合来强迫 Generator 拒答。

这条规则是为了防止 2Wiki 一类多跳任务中的中间证据，因为一条必要 passage 可能无法单独回答最终问题。

### 5.7 容量受限、parent-aware 的精确独立集与 tie-break

图中至少有一条边时才执行独立集搜索。

目标顺序固定为：

1. 集合内部不能包含任何 contradiction 边。
2. 集合大小不能超过 max_selected。
3. 首先最大化集合覆盖的不同 parent key 数量。
4. 然后最大化集合中的 passage 数量。
5. 若前两项仍相同，选择规范 TopK 索引向量字典序最小的集合。
6. 最终按同一规范 TopK 顺序输出。

这样直接搜索最终要交给 Generator 的集合，不允许“先在 20 个节点上求完整 MIS，再事后截断成 10 条”。后者既不一定得到排名最优的 10 条无冲突证据，也不等同论文算法。

实现约束：

- 只用 Python 标准库，不引入 networkx。
- top_n 硬限制为 20，因此 exact search 可控。
- 推荐使用 bit mask 表示邻接关系。
- 穷举大小不超过 min(max_selected, n) 的组合，计算三元目标：
  (不同 parent 数，passage 数，规范索引向量)。
- 前两项越大越好；前两项相同时，规范索引向量字典序越小越好。
- 可以用 branch-and-bound 或标准库 combinations；top_n 的硬上限为 20。
- 不调用 random。

伪代码：

    ranked = candidates sorted exactly like TopK:
             (-retrieval_score, evidence_id)
    window = ranked[:top_n]
    baseline = TopKSelector.select(...)
    parent_key[i] = source_parent_id for window[i],
                    otherwise document_id

    answers = extract each window passage independently
    valid = indices whose answer is not unanswerable
    pairs = all (i, j), i < j, with i and j in valid
    ordered_pairs = [(statement[i], statement[j]) for every valid i < j]
    probabilities = contradiction_scorer.score_pairs(ordered_pairs)

    graph = empty undirected graph over every window node
    for pair, probability in zip(pairs, probabilities, strict=True):
        if probability >= 0.5:
            add undirected edge

    if graph has no edges:
        return baseline

    chosen_indices = exact_parent_aware_independent_set(
        graph, parent_key, capacity=max_selected
    )
    chosen = window nodes in chosen_indices, kept in canonical TopK order
    return SelectionResult made from original evidence_id and retrieval_score

候选数大于 top_n 时，冲突分支不得用未经 NLI 检查的尾部候选补位。正式集成配置应让 Retriever 输出数量与 top_n 对齐；这一点之后由实验计划冻结。

## 6. 后端失败与回退

### 6.1 新增异常类型

在 reliability_mis.py 定义 SelectorBackendError。它只表示外部推理后端失败，例如：

- Granite 模型加载、tokenization 或 generation 失败。
- DeBERTa 权重加载、tokenization、OOM 或 inference 失败。
- NLI 返回数量不匹配、NaN、inf 或超出 [0,1] 的概率。

答案生成和 NLI adapter 可以在自己的边界捕获底层异常并转换成 SelectorBackendError。

由于现有 GraniteLLMClient 会在构造时立即加载模型，不能直接在 composition 阶段实例化它，否则“模型加载失败时回退 TopK”无法发生。reliability_mis.py 还应提供一个 Selector 私有的 LazyAnswerGenerator：

- 构造参数是 TextGenerator factory，不直接依赖 Generator 包。
- 第一次 generate 时才调用 factory，随后缓存同一个实例。
- factory 创建失败或 generate 失败时，在这个外部后端边界统一转成 SelectorBackendError。
- composition.py 只负责传入创建 GraniteLLMClient 的 factory。

这样既不修改 generator/granite.py，也能让 Selector 在模型加载或推理失败时原子回退。

### 6.2 Selector 只捕获这一类错误

ReliabilityMISSelector.select 只捕获 SelectorBackendError：

- 任意一个 passage 抽取失败：整条 query 原子回退 TopK。
- 任意一个 NLI batch 失败：整条 query 原子回退 TopK。
- 不允许使用部分答案或半张图继续过滤。

以下错误必须显式抛出，不得吞掉：

- query ID 错误。
- max_selected、top_n 或必需 source-parent sidecar 非法/缺失。
- MIS 内部不变量被破坏。
- 返回未知 evidence_id。
- 编程错误。

这样既保证运行时后端故障不会牵连 Generator，又不会用“静默 fallback”掩盖实现 bug。

## 7. 可观测性，但不改公共契约

在 reliability_mis.py 增加一个内部 MISSelectionEvent dataclass，并提供可选 on_event 回调，沿用现有 gated.py 的做法。

事件只记录：

- query_id
- mode：mis / topk_no_conflict / topk_backend_fallback
- candidate_count、window_count、valid_answer_count、unanswerable_count
- independent_parent_count、unresolved_parent_count
- pair_count、contradiction_edge_count
- selected_ids、dropped_ids
- model_id、model_revision、threshold、prompt_version
- fallback_stage 和错误类型（若发生）

默认不要记录 passage 原文或完整答案。必要时记录 statement 的 SHA-256 短 hash，减少数据泄露。

on_event 自身失败不能改变选择结果：捕获 sink 异常并写一次 logger warning。

SelectionResult 不增加 status、reason、graph 或 answer 字段。

## 8. 文件级实现说明

### 8.1 新增：selector/reliability_mis.py

建议包含以下最小元素：

    class SelectorBackendError(RuntimeError)

    class ContradictionScorer(Protocol):
        score_pairs(
            pairs: Sequence[tuple[str, str]]
        ) -> tuple[float, ...]

    class LazyAnswerGenerator:
        __init__(factory)
        generate(prompt) -> str

    @dataclass(frozen=True)
    class MISSelectionEvent

    def is_unanswerable_answer(answer: str) -> bool
    def answer_statement(question: str, answer: str) -> str
    def build_pair_batch(...)
    def build_conflict_graph(...)
    def is_independent(...)
    def exact_parent_aware_independent_set(
        graph, parent_keys, capacity
    )

    class ReliabilityMISSelector:
        __init__(
            answer_extractor,
            contradiction_scorer,
            parent_by_document,
            top_n=20,
            threshold=0.5,
            passage_chars=600,
            on_event=None,
        )
        select(query, candidates, max_selected) -> SelectionResult

实现注意：

- 构造 AnswerExtractionEngine 时强制 use_parametric=False。
- threshold 构造参数可以保留用于 fake 单测，但 composition 的生产注册必须固定传 0.5，不从用户配置读取。
- 用 TopKSelector 对象产生 baseline 和 backend fallback。
- 只从输入 CandidateSet 中映射输出 evidence_id。
- parent_by_document 只参与选择目标，不参与建边或预合并；缺失单条映射时回退 document_id。

### 8.2 新增：selector/deberta_contradiction.py

建议类名 DebertaContradictionScorer，职责只限“pair → contradiction probability”：

- 构造时不加载权重。
- 第一次 score_pairs 时 lazy load，此后进程内复用。
- AutoTokenizer.from_pretrained 与 AutoModelForSequenceClassification.from_pretrained 都传固定 revision。
- 读取 HUGGINGFACE_API_KEY、MODEL_CACHE_DIR、LLM_DEVICE，与现有运行环境一致。
- model.eval()。
- torch.inference_mode()。
- padding=True、truncation=True、max_length=512。
- batch_size=32。
- 从 model.config.id2label 按名字找到 contradiction index。
- 必须按名称恰好解析到一个 contradiction label；零个或多个都抛 SelectorBackendError。
- 返回 float probability tuple，严格校验长度与数值范围。
- 输入是有序的 (premise, hypothesis) tuple 序列；输出必须保持同序且长度完全一致。

不要：

- import evidence_rag.generator.nli。
- import evidence_rag.relations。
- 复用只有 argmax label 的 NLI API。
- 硬编码 logits[:, 2]。
- 写作者机器的 /scratch 路径。

### 8.3 修改：composition.py

只做加法式注册：

1. import ReliabilityMISSelector、ContradictionScorer、DebertaContradictionScorer。
2. 将 build_selector 签名扩为：

       build_selector(config, *, llm=None, contradiction_scorer=None)

   新参数有默认值，因此现有调用完全不变。
3. 新增 config.name == "reliability-mis" 分支。
4. 参数白名单只允许 top_n。
5. top_n 默认 20，并额外拒绝 top_n > 20。
6. reliability-mis 固定调用现有 _load_parent_index；缺少 SOURCE_PARENT_INDEX 时明确拒绝构建。
7. source_parent_provenance 对 reliability-mis 也记录 sidecar 路径与 SHA-256，不能只为 gated selector 记录。
8. answer extractor：
   - 测试传 llm 时，让 LazyAnswerGenerator 的 factory 返回该 fake。
   - 生产未传时，让 LazyAnswerGenerator 的 factory 创建现有 GraniteLLMClient。
   - 不得在 build_selector 阶段直接实例化 GraniteLLMClient。
   - Granite generation config 固定 temperature=0、max_new_tokens=32。
9. contradiction scorer：
   - 测试传 fake 时直接使用。
   - 生产未传时创建固定 model ID + revision 的 DebertaContradictionScorer。
10. 构造 ReliabilityMISSelector，传入 parent_by_document，并固定 threshold=0.5、use_parametric=False。

不要修改 build_pipeline_from_config、EvidenceRAGPipeline 或 Generator 构造逻辑。

### 8.4 新增与修改测试

#### tests/selector/test_reliability_mis.py

使用 fake answer extractor + fake contradiction scorer，覆盖：

1. 空候选：不调用任一后端，返回空 TopK。
2. 单候选：无 NLI pair，结果等于 TopK。
3. 全 NONE：结果等于 TopK。
4. 全 neutral / entailment：无边，结果等于 TopK。
5. threshold 0.499：不建边。
6. threshold 正好 0.5：建边。
7. 一条冲突边的一对候选：选规范 TopK 顺序更高者。
8. 三节点路径 A-B-C：MIS 为 A+C。
9. 3-v-1 冲突图：选三个内部一致节点。
10. 多个同目标独立集：按规范 TopK 索引向量固定决胜。
11. 输入 tuple 任意打乱，结果不变。
12. 同一输入重复运行，结果不变。
13. NONE 节点与有效答案节点同时存在：NONE 保留但不参与 pair。
14. 同一 parent 复制多个同义 chunks，不会增加独立来源目标或改变胜出的事实侧。
15. 同一 parent 内的真/假 twin 不预合并，仍能建立 contradiction 边。
16. 缺失单条 parent 映射时使用 document_id，并记录 unresolved count。
17. 输出只含原 evidence_id，分数沿用 retrieval_score，rank 连续。
18. 输出数量不超过 max_selected，且直接搜索最终集合，不做事后截断。
19. CandidateSet 超过 top_n：冲突路径不从未检查 tail 补位。
20. answer backend 失败：整条 query 精确回退 TopK，只发一次 fallback event。
21. NLI 第二个 batch 失败：不使用第一批结果，整条 query 精确回退 TopK。
22. query ID / max_selected / top_n 非法：显式报错，不 fallback。
23. event sink 报错：SelectionResult 不变。

#### tests/selector/test_deberta_contradiction.py

全部使用 fake tokenizer/model，不联网：

1. 构造对象时模型加载次数为 0。
2. 第一次 score 时加载一次，后续不重复加载。
3. 用伪造 id2label 顺序证明没有硬编码 contradiction index。
4. 返回 contradiction probability，不是 argmax confidence。
5. pair 与输出顺序一一对应。
6. 33 个 pair 被分成 32+1，但输出仍完整有序。
7. 输出长度不匹配、NaN、inf、越界值转成 SelectorBackendError。
8. checkpoint 无 contradiction label 时明确失败。

LazyAnswerGenerator 另需在 test_reliability_mis.py 验证：build_selector 后加载次数为 0；首次答案抽取时为 1；后续 passage 和后续 query 都复用同一个实例。

#### tests/pipeline/test_selector_registration.py

增加：

- reliability-mis 可由 fake llm + fake scorer 构造。
- 默认 top_n=20。
- top_n=0、top_n=21、未知参数全部被拒绝。
- 缺少 SOURCE_PARENT_INDEX 时构建被拒绝；有效 sidecar 可被加载并记录 hash。
- 构建测试不得加载真实模型。
- 原有 top-k、corroboration、gated tests 全部不变。

#### tests/pipeline/test_reliability_mis_pipeline.py

用 fake Retriever、fake Selector 后端和现有简单 Generator 跑一次真实 Pipeline：

- Generator 收到的是筛选后的原 EvidenceCandidate passage。
- 内部 isolated answer statement 没有进入 Generator。
- 被 MIS 删除的 evidence_id 不能被引用。
- Pipeline 的 top_k / max_selected 调用方式不变。
- 无冲突路径与 TopK pipeline 输出一致。
- NLI 后端失败时 pipeline 仍完成，且使用 TopK 证据。

#### tests/typecheck.py

增加一条静态赋值，证明 ReliabilityMISSelector 满足现有 Selector Protocol；不修改 Protocol。

## 9. 推荐实施顺序

每一步通过本步测试后再进入下一步。

### Step 0：保存基线

执行：

    git status --short
    git rev-parse HEAD
    python -m pytest tests/selector tests/pipeline/test_selector_registration.py

记录现有失败；不得把已有失败算到新实现头上。

### Step 1：先实现纯图算法

文件：

- 新增 selector/reliability_mis.py 中的纯函数。
- 新增 test_reliability_mis.py 中不需要模型的 MIS 测试。

先只完成：

- statement 生成。
- pair 索引生成。
- 冲突图。
- 容量受限、parent-aware 的精确独立集。
- 规范 TopK tie-break。

验收：

    python -m pytest tests/selector/test_reliability_mis.py -q

### Step 2：实现 DeBERTa 概率适配器

文件：

- 新增 selector/deberta_contradiction.py。
- 新增 test_deberta_contradiction.py。

验收：

    python -m pytest tests/selector/test_deberta_contradiction.py -q

所有默认测试必须只使用 fake，不下载模型。

### Step 3：接通 ReliabilityMISSelector

在 reliability_mis.py 中加入：

- AnswerExtractionEngine。
- NONE 保守处理。
- TopK no-conflict 与 backend fallback。
- MISSelectionEvent。
- SelectionResult 映射。

重新跑：

    python -m pytest tests/selector -q

### Step 4：只在 composition 注册

修改 composition.py 和 test_selector_registration.py。

验收：

    python -m pytest tests/pipeline/test_selector_registration.py -q
    python -m pytest tests/architecture/test_boundaries.py -q

确认 build_selector 的旧调用全部继续通过。

### Step 5：最小全链路工程冒烟

新增 test_reliability_mis_pipeline.py，不改 pipeline/service.py。

验收：

    python -m pytest tests/pipeline/test_reliability_mis_pipeline.py -q

这一步只证明接口接通，不评价研究效果。

### Step 6：全仓回归

执行：

    ruff check src tests
    mypy src tests/typecheck.py
    python -m pytest

再执行范围检查：

    git diff -- src/evidence_rag/retriever
    git diff -- src/evidence_rag/generator
    git diff -- src/evidence_rag/pipeline
    git diff -- src/evidence_rag/contracts
    git diff -- pyproject.toml

以上五项必须为空。

## 10. 代码完成标准

只有全部满足，才能说“Selector 代码已经完成，可以开始另列实验计划”：

- 新 selector 名为 reliability-mis，旧 Selector 全部保留。
- 输入输出签名未变。
- 没有新增公共字段或 schema version。
- 没有改 Retriever、Generator、Pipeline、Contracts。
- 没有参数知识提取。
- 没有 claim slicing、source reputation、外部搜索、训练或 Graph2 依赖。
- 无 contradiction 时结果与 TopKSelector 完全相等。
- 全 NONE 时结果与 TopKSelector 完全相等。
- 任一外部后端失败时整条 query 原子回退 TopK，并留下事件。
- 只有 P(contradiction) >= 0.5 才会建边。
- 容量受限、parent-aware 的精确独立集与规范 TopK tie-break 有独立单测。
- 所有输出 ID 来自输入 CandidateSet。
- selection_score 沿用 retrieval_score。
- 相同输入重复运行结果一致。
- 默认测试不联网、不下载模型、不要求 GPU。
- 全仓 lint、mypy、pytest 通过。
- 冻结目录的 git diff 为空。

以下不属于“代码完成”：

- harmful-in-context 是否下降。
- required-evidence recall 是否守住。
- 2Wiki supporting-fact recall。
- 最终答案准确率。
- 与 TopK / Graph1 的实验对比。

这些问题必须在下一份独立实验计划里回答，不能用单元测试代替。

## 11. 已知风险，先记录但不在本次加功能

1. Hybrid RRF rank 是相关性，不是来源真实性。它只能作为 tie-break，不能直接继承论文的可靠性理论保证。
2. 多个一致的错误来源仍可能成为最大集合；MIS 保证内部不矛盾，不保证集合一定真实。
3. source_parent 目标能抑制重复 chunks 冒充独立多数，但 parent sidecar 本身可能存在错误或缺失；事件必须报告 unresolved 数，后续实验还要验证这一适配的实际影响。
4. 单段答案抽取若失真，NLI 图也会失真。后端失败可 fallback，但“模型给出合法却错误的答案”只能靠后续组件评估发现。
5. Top-20 会产生最多 190 个 NLI pair，成本高于论文常用 Top-10。先保证代码正确，再由实验计划测延迟并决定最终窗口。

这些风险不能成为继续堆方法的理由。第一版先按本计划完成最小闭环，实验失败时回退 TopK，而不是继续无限扩展 Selector。

## 12. 论文与固定版本参考

### 本地论文

- [ReliabilityRAG_NeurIPS_2025.pdf](references/ReliabilityRAG_NeurIPS_2025.pdf)

### 正式论文

- 本地 PDF SHA-256：3ff139a521ea4149d66e90eaa1b4523baff698cf74e853dd1d5f625c5136b092
- [NeurIPS 2025 论文页](https://proceedings.neurips.cc/paper_files/paper/2025/hash/41457d56a2fdd20b4c072190290a549b-Abstract-Conference.html)
- [NeurIPS 2025 官方 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/41457d56a2fdd20b4c072190290a549b-Paper-Conference.pdf)

### 官方代码（固定 commit）

- [ReliabilityRAG 仓库固定版本 e354aa9](https://github.com/zeyushen-yo/ReliabilityRAG/tree/e354aa91dd9bc44d082d706e2b5205fc85ce950c)
- [MIS 主流程：defense.py L174–L265](https://github.com/zeyushen-yo/ReliabilityRAG/blob/e354aa91dd9bc44d082d706e2b5205fc85ce950c/src/defense.py#L174-L265)
- [exact MIS 与 tie-break：defense.py L267–L294](https://github.com/zeyushen-yo/ReliabilityRAG/blob/e354aa91dd9bc44d082d706e2b5205fc85ce950c/src/defense.py#L267-L294)
- [官方 CLI 默认值](https://github.com/zeyushen-yo/ReliabilityRAG/blob/e354aa91dd9bc44d082d706e2b5205fc85ce950c/main.py#L20-L54)

注意：截至 2026-08-08，固定 commit 的仓库中没有 LICENSE、COPYING 或 NOTICE。可以阅读论文与代码理解方法，但不要把上游源码复制进本项目。本计划要求依据论文算法独立实现，并在报告中引用论文和仓库。

### NLI 模型（固定 revision）

- [模型卡](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli)
- [固定 revision b3546ea](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli/tree/b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7)

模型卡标注 MIT License。固定 revision 的 config.id2label 为 entailment / neutral / contradiction，但实现仍必须按标签名称解析，禁止依赖具体下标。

## 13. 交接时同事只需要回答的五个问题

1. 是否只改了本计划允许的文件？
2. 无矛盾、全 NONE、后端失败是否都精确回到当前 TopK？
3. 是否只有 DeBERTa contradiction probability ≥ 0.5 才删证据？
4. 输出是否仍然只是原始 evidence_id，并能被现有 Pipeline/Generator 正常消费？
5. 全仓测试是否通过，且没有真实模型被默认测试下载？

五项全部为“是”，代码改造阶段才算完成。随后再新建实验计划，决定它是否优于 TopK 并值得成为最终 Selector。
