# 三模块 RAG 并行开发说明

这份文档是当前项目的统一说明。

当前 RAG 系统分成三段：

```text
用户问题
  -> QueryChecklist 按当前问题整理任务重点
  -> Retriever 找全候选证据
  -> Selector 从候选证据里选准证据
  -> Generator 使用被选证据生成答案和引用
```

中间由 Pipeline 负责连接三段，但 Pipeline 不做检索、不做选择、不做生成算法。

`QueryChecklist` 不是写死的固定清单，而是每个问题运行时临时生成的一份轻量任务提示。它现在只保留三个必要信息：

- `focus`：这个问题主要问什么
- `required_facts`：回答时至少要覆盖哪些关键事实
- `constraints`：明显的年份、实体等限制

它的作用是把问题先整理清楚，后续 Retriever、Selector、Generator 都可以按需要使用它；如果某个模块暂时不用，也不影响三模块接口。

当前有几个组合入口：

- 轻量 baseline：`BM25Retriever -> TopKSelector -> ExtractiveGenerator`，用于快速测试结构。
- Granite dense baseline：`GraniteDenseRetriever -> TopKSelector -> GraniteGenerator`。
- 推荐主线 baseline：`Query2Doc + GraniteDenseRetriever -> TopKSelector -> GraniteGenerator`。
- Selector 实验入口：`Query2Doc + GraniteDenseRetriever -> CorroborationSelector -> GraniteGenerator`。

旧实验结果显示 Query2Doc + Granite 在 NIAH/needle 检索任务上比纯 Granite dense 更适合“把证据找全”。Selector 暂时仍可以是简单 baseline，因为后续重点就是让 Selector 组替换和提升它。

## 1. 三个模块分别负责什么

### Retriever：把证据找全

目标：让后面的 Selector 有足够完整的候选证据。

输入：

- 用户问题 `Query`
- 检索系统自己的语料或索引
- 需要返回多少候选证据 `top_k`

输出：

- `CandidateSet`
- 每条候选证据包含：证据 ID、文档 ID、chunk ID、文本、来源、检索分数、检索排名

主要优化指标：

- 相关证据有没有被找出来
- 候选池覆盖率
- 候选证据排序质量

Retriever 不负责决定最终采用哪几条证据，也不负责生成答案。

### Selector：把证据选准

目标：从 Retriever 给出的候选证据里排除干扰，选出真正应该交给 Generator 的证据集合。

输入：

- 用户问题 `Query`
- Retriever 给出的 `CandidateSet`
- 最多选择多少条证据 `max_selected`

输出：

- `SelectionResult`
- 只包含：被选中的 evidence IDs、selection scores、selection ranks

主要优化指标：

- Retriever 已经找到了相关证据时，Selector 有没有保住它们
- 被选证据的 precision
- 面对干扰、冲突、重复来源、相似但错误的证据时是否稳定

Selector 可以在内部使用图特征、support/refute 信号、reranker、分类器等方法。
但这些都是内部方法，跨模块输出仍然只给 evidence IDs、分数和排序。

Selector 不直接告诉 Generator “证据是否充分”“是否冲突”“缺少什么”。如果以后需要这些字段，三组要一起重新确认接口。

### Generator：把证据用好

目标：基于 Selector 给出的证据，生成正确、可引用、尽量忠实于证据的答案。

输入：

- 用户问题 `Query`
- Pipeline 根据 Selector 选出的 IDs 找回来的原始证据 `SelectedEvidenceSet`

输出：

- `GenerationResult`
- 包含答案文本和实际引用过的 evidence IDs

主要优化指标：

- 相关证据已经被选中时，答案是否正确
- 引用是否来自被选证据
- 答案是否忠实于证据
- 证据不足时是否能拒答、表达不确定，或通过 checklist 做自检

Generator 不应该自己重新检索新证据。如果证据不够好，应该通过评测反馈给 Retriever 或 Selector。

## 2. Pipeline 怎么把三组接起来

Pipeline 的连接方式是：

```text
Retriever 输出原始候选证据
Selector 只输出选中的 evidence IDs
Pipeline 根据 IDs 找回 Retriever 的原始证据
Generator 只能使用这些被选证据
```

这样可以保证：

- Selector 不能悄悄改写证据文本
- Generator 不能引用没被选中的证据
- 三个模块可以独立替换

## 3. 三组怎么并行开发

Retriever 组：

- 主要改 `src/evidence_rag/retriever`
- 测试放 `tests/retriever`
- 只要输出仍然是 `CandidateSet`，内部可以换 Granite dense retrieval、hybrid retrieval、query expansion 等方法

Selector 组：

- 主要改 `src/evidence_rag/selector`
- 测试放 `tests/selector`
- 可以用固定的 `CandidateSet` 独立开发和评测，不需要等 Retriever 完成
- 目前已迁入一个 `CorroborationSelector`，可作为 Selector 组继续实验的起点

Generator 组：

- 主要改 `src/evidence_rag/generator`
- 测试放 `tests/generator`
- 可以用固定的 `SelectedEvidenceSet` 独立开发 prompt、checklist、citation check、answer verification 等方法

集成负责人：

- 维护 `src/evidence_rag/contracts`
- 维护 `src/evidence_rag/pipeline`
- 维护 `src/evidence_rag/composition.py`
- 负责确认三个模块可以互换，并且完整系统能跑通

## 4. 仓库文件夹怎么用

当前只需要记住三类位置：代码、测试、文档/计划。

```text
IBM_Granite_Project/
├── src/evidence_rag/
│   ├── retriever/       Retriever 代码
│   ├── selector/        Selector 代码
│   ├── generator/       Generator 代码
│   ├── contracts/       三组共同接口
│   ├── pipeline/        三组连接逻辑
│   ├── evaluation/      整体评测和过程指标
│   ├── query_analysis.py 每个问题的轻量 checklist
│   ├── cli/             简单运行入口，平时不用重点关注
│   └── composition.py   选择本次使用哪三个模块组合
├── tests/
│   ├── retriever/       Retriever 测试
│   ├── selector/        Selector 测试
│   ├── generator/       Generator 测试
│   ├── pipeline/        三组接起来的测试
│   ├── contracts/       共同接口测试
│   ├── architecture/    防止模块互相乱依赖
│   └── evaluation/      评测和过程报告测试
└── docs/
    ├── README.md        当前这份统一说明
    └── selector/        Selector 组已有计划和实验记录
```

每组只需要按同样方式管理自己的东西：

- 代码放 `src/evidence_rag/<模块名>/`
- 测试放 `tests/<模块名>/`
- 计划、实验记录、讨论结论放 `docs/<模块名>/`

目前只有 Selector 已经有自己的文档文件夹。Retriever 和 Generator 如果需要记录计划或实验，再创建自己的 `docs/retriever/` 或 `docs/generator/`，不需要提前空建。

文档形式各组自己决定。可以是 plan、tracker、实验记录、失败分析、结果总结，只要别人能看懂当前做到哪一步、下一步要做什么就行。

大数据、模型权重、完整日志、缓存不要提交进仓库，放共享存储，并在文档里写清楚位置和版本。

## 5. 合起来怎么测试

整体测试不能只看最终答案分数，还要保留过程信息。

每个 case 至少要能看到：

- Retriever 找到了哪些候选证据
- Selector 选中了哪些证据
- Generator 引用了哪些证据
- 最终答案是什么

这样失败时可以判断问题出在哪里：

- Retriever 没找到关键证据
- Retriever 找到了，但 Selector 过滤掉了
- Selector 给对了，但 Generator 没用好
- 某个模块指标变好，但完整系统指标退步

模块改进要接入主线时，至少要报告两类结果：

- 自己模块的指标有没有提升
- 完整 RAG 系统接起来后有没有提升或退步

不能只因为单个模块指标变好，就直接认为整体系统变好。

## 6. 合并新版本的基本规则

一个模块的新版本要接入主线，需要满足：

1. 输入输出接口不变
2. 自己模块的测试通过
3. 三模块接起来能跑通
4. 过程评测能说明变化来自哪个阶段
5. 同时报告模块指标和整体系统指标

如果要改接口，比如给 Selector 增加新的跨模块字段，必须三组先讨论同意。

## 7. 当前注意事项

- 旧项目只作为参考，不要直接把旧代码混回当前主线
- Selector V2 的旧计划不是已经批准的训练任务，需要重新确认任务、数据和指标
- FinanceBench 目前保留给最终完整系统对比，不用于各模块训练、调参或开发测试
- 本地缓存、模型权重、大数据、大日志不要提交进仓库

当前基线代码位置：

- 正式包：`src/evidence_rag`
- baseline 组合入口：`src/evidence_rag/composition.py`
- 旧版本备份标签：`legacy-before-three-module-reset-2026-07-13`
