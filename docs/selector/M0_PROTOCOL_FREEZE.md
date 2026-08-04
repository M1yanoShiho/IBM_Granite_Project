# Graph 2.0 — M0 协议冻结与预注册

**日期:** 2026-07-30 | **状态:** **DRAFT — 决定已全部填,等 R001 的 G-FC 基线数值后转 FROZEN** | **协议版本:** **g2-proto-2**(2026-08-03 经修订案 A1 升版,见 §9;g2-proto-1 为其前身)

**转 FROZEN 的唯一剩余前置:** §3 的 G-FC 需要 R001 实测的基线值与 δ。§1 的决定已全部拍板(2026-07-30),§6 审计待数据就位后执行。
**数值缺位时不得标 FROZEN。**

**上游:** [TRAINING_PLAN.md](TRAINING_PLAN.md)(2026-07-11 迁移草稿)、
[../superpowers/specs/2026-07-30-graph-2.0-relation-layer-design.md](../superpowers/specs/2026-07-30-graph-2.0-relation-layer-design.md)(设计与论证)、
[selector.md](selector.md) §5(S1–S6 证据链)、[EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md) R000–R003。

**与 TRAINING_PLAN 的关系:** 凡本文件冻结的条目,**以本文件为准**;未提及的条目仍沿用 TRAINING_PLAN。
本轮相对 TRAINING_PLAN 有四处**实质修改**,各自附理由:ContractNLI 移出(§3.0)、Gate 0B 拆两层(§3)、
SAME_SOURCE 无条件修复(§3.4)、废除 3-seed ensemble(§5.4)。

---

## 0. M0 是什么 / 不是什么

- **是** 预注册里程碑:冻结协议、关系 schema、数据决定、指标与守卫、样本量依据。
- **不是** 实现里程碑:M0 **不看任何 dev 或 test 结果**。(Plan 1 的地基代码可以先写,因为它不产生任何 Graph 2.0 的效果量。)
- **退出判据:** Gate 0A(§6)零违规,且 §1 全部决定已填、§3 全部守卫已带具体数值。
- **冻结后的修改规则:** 冻结之后任何改动都必须新开协议版本号、写明原因与时间,**不得在看到结果之后追加或调整**。
  这条是 S4/S5 两次自我推翻换来的纪律 —— 那两次之所以能诚实更正,是因为预期写在前面。

---

## 1. 已拍板的决定

### D1 — Graph 2.0 的收法 = **A(只升级"边与票的构造")** [2026-07-30 拍板]

| | A — 已采纳 | B — 未采纳 |
|---|---|---|
| 改什么 | NLI 关系边 + UNKNOWN + `source_parent_id` 替换"怎么数票";门的四条件与整数票判据**不动** | 回到 LightGBM LambdaRank + graph 特征做排序/选择 |
| 与承诺 1 | 一致 —— NLI 只进"边怎么建",不进"踢留的刻度" | 冲突 —— 需重新论证 V1 的 harm 失败为何不会重演 |
| 主对照 | 同池配对 Graph 1.0 vs 2.0(gate-on),外加 gate-off | 需重训学习型 v1 作对照臂 |

**选 A 的三条理由:** (1) B 的主张要求先复现一个在 harm 轴上 FAIL 过、且特征重要性 72% 塌回 relevance/rank 的
学习型 v1 作对照;(2) S1–S6 全部建立在"同池配对、唯一变量=门"的对照结构上,E1/E2 harness 现成;
(3) B 的路径长度里没有一个可交付的中间态。

**代价(如实记录):** 放弃 TRAINING_PLAN §3.5 里"把 graph 特征喂给学习器"那部分。
`alternative_answer_entropy` 一类连续特征无处可去 —— 按承诺 5,它们只能进重排段,不能进门。这是信号毕业制的必然结果,不是遗漏。

**可执行保证:** 簇的**构造**已抽成 `SupportProvider` 协议,Graph 1.0 与 2.0 共用同一份 `_decide` 与四条件代码;
单测断言"给定相同 clusters,两臂 GateDecision 完全相同"。D1=A 因此是机器可验证的,不是口头承诺。

### D2 — NIAH train 规模:**问题已消解** [2026-07-30]

D1=A 下选择器**无参数**,本轮没有 Selector 训练。"训练集"只对关系模型有意义,而关系模型走**零训练探底**路线(§3.1),
连关系模型也不训。NIAH train 因此只用于构造 §3 的 0B-2 探针。
2000-vs-500 的原问题不再适用 —— **记消解理由而非留空**;若 §3.6 的训练路径被启动,本条恢复为待决项并须在看 dev 结果前重填。

---

## 2. 冻结的关系 schema

### 2.1 三类关系 + UNKNOWN

| 关系 | 监督来源 | 生成方式 | UNKNOWN 条件 |
|---|---|---|---|
| `CLAIM_SUPPORTS` | 去污染 VitaminC(official test 不动) | 关系模型预测 | 模型预测为该类 |
| `CLAIM_REFUTES` | 同上 | 关系模型预测 | 同上 |
| `SAME_SOURCE` | `source_parent_id` | **确定性规则**,不预测 | 无 —— 规则可判则判,不可判则各自为父 |

冻结的不变量:

1. **无法确定时输出 UNKNOWN,不得强制建边。**
2. 每条预测边保存 `confidence`、模型版本、premise/hypothesis 文本 hash。
3. `TIME_MISMATCH` / `CONDITION_MISMATCH` **不进本轮**。
4. 正式 dev/test 建图**不得读取** `utility_grade`、`harm_type`、gold `answer_cluster_id`、gold answers、official evidence flag。

### 2.2 四条件的语义重映射(新增,D1=A 的具体含义)

| 门条件 | Graph 1.0 | Graph 2.0 |
|---|---|---|
| 1. c 抽出了有效答案 | `is_valid_answer(raw)` | c 至少有一条 SUPPORTS 边 |
| 2. 存在竞争簇 | 任一 canonical 答案不同的簇 | 见 `conflict_mode` |
| 3. `support(K′) − support(K) ≥ margin` | 不变 | 不变(整数票) |
| 4. `support(K) ≤ support_cap` | 不变 | 不变(整数票) |

两条 Graph 1.0 里不存在、**必须新冻结**的语义:

1. **c 支持多个互不蕴含的簇** ⇒ c 视为"未表态",条件 1 判假,**不可踢**。保守方向,符合承诺 3。
2. **c 的"自己的簇"** = c 支持的簇中 `independent_support` 最大者。规则 1 已排除歧义情形。

`conflict_mode` 冻结两值:

- `distinct_cluster`(**主口径**):竞争簇 = 任一其他 claim 簇。聚类已按互蕴含合并,不同簇按构造即互不蕴含,
  实现上不需要额外蕴含检查 —— 与 Graph 1.0 结构逐条平行,唯一变量确实只有边的构造。
- `refutes_edge`(**消融臂**):竞争簇还须有成员向 c 的 claim 发出 REFUTES 边。
  这一臂使 Block 3 的"去掉 CLAIM_REFUTES"消融有实质含义 —— 主口径下 REFUTES 边只被记录不被门消费。

### 2.3 三条构造规则(冻结)

1. **Graph 2.0 = Graph 1.0-lenient + NLI。** lenient 等价保留为 NLI 之前的确定性预合并,不被替换 ——
   主对照就是 Graph 1.0-lenient,保留它使增量变量**纯粹是 NLI**。
2. **claim–claim 聚类比较整句 hypothesis,不比裸答案串。** 模板冻结为
   `The answer to the question "{Q}" is {A}.`;QA2D 式转换器作预注册消融。
3. **parametric 自答不成 claim 节点**(承诺 5:模型先验留在重排、禁入门内)。

### 2.4 argmax 主口径,无阈值 [新增]

**主口径 = argmax,不设 τ。** UNKNOWN 是模型预测的一个**类别**,不是阈值产物 ⇒ 踢人路径上零绝对刻度,
保住与 ArbGraph"绝对 credibility 阈值 vs 池内相对整数票"的差异化轴。

**平局裁决(冻结):** 平局按 UNKNOWN → REFUTES → SUPPORTS 的顺序解,**绝不落到 SUPPORTS**。
理由:SUPPORTS 边正是让候选**可被踢**的条件,抛硬币不得制造一条。失效方向=闭嘴(承诺 3)。
浮点 softmax 下平局近乎不可能,但规则必须确定且明写,不能由枚举声明顺序偶然决定。

**τ-gating 降级为预注册补救:** 仅当 argmax 下 Gate 0B-1 的 REFUTES precision < .85 时启用;
τ 只在 VitaminC official **dev** 上按 coverage/precision 目标冻结一次,之后不许按 NIAH 结果调;
启用则报告**必须写明"补救措施被触发"**。

---

## 3. 冻结的指标与守卫

### 3.0 ContractNLI 移出关系监督集 [实质修改,附理由]

[stanfordnlp/contract-nli](https://github.com/stanfordnlp/contract-nli) 是 **17 条固定假设 × 607 份 NDA** 的
文档级三分类(Koreeda & Manning, Findings of EMNLP 2021),假设形如 "Some obligations of Agreement may survive
termination.",与"段落是否支持 'Q 的答案是 X'"无结构相似性;v1 已量到 ContractNLI 迁移 **−0.134**。
而 Gate 0B 原文要求"两数据集必须**分别**通过" ⇒ 保留它等于要求在一个与任务无关的法律域上打到 REFUTES precision ≥ .85,
是自设阻塞。

VitaminC 反向成立:45 万 claim-evidence 对取自 10 万+ Wikipedia 修订,**一对近乎逐字相同的证据只有一处事实被改动,
一条支持一条不支持**(Schuster, Fisch & Barzilay, NAACL 2021)—— 与 injector 造的孪生同构。

### 3.1 关系模型:零训练探底优先

R012 为**一次 sweep**,三个现成 checkpoint 同场跑,不串行:

| 臂 | 模型 | 角色 |
|---|---|---|
| 主 | `tals/albert-xlarge-vitaminc-mnli`(~59M) | 三类原生对齐,域对口,开销可忽略 |
| 对照 | MiniCheck-FT5(770M) | LLM-AggreFact <1B SOTA;二分类,需否定 claim 双向探测,单独一步 |
| 上界 | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | 通用 NLI 上界,非域内 |

**为什么必须同场:** 只跑一臂时,未过 Gate 只能得出"albert 不够",**无法排除"任何零训练模型都不够"** ——
而这个区分正是决定 §3.6 训练路径是否启动的唯一依据。

**LLM teacher 边界:** MiniCheck 由 GPT-4 合成数据训练。它作为**系统组件**(边预测器)不违反零新增人工标注约束;
禁止的是把它的输出当 primary label。Gate 0B 的验收标签仍只来自 official test 与 deterministic provenance。

### 3.2 Gate 0B 拆两层 [实质修改]

原 Gate 0B 只在 official NLI 数据上验收。洞:**一个在 VitaminC official test 上过关的模型,
完全可能在"NQ 段落 + 模板 hypothesis"上无用。**

**0B-1 外部效度**(official labels,一次性,VitaminC official test):沿用原阈值**不放宽** ——
REFUTES precision ≥ .85、support/refute macro-F1 ≥ .80、non-UNKNOWN coverage ≥ .80、SUPPORT/REFUTES 各 ≥ .70。
某一类在评测集中**完全缺席**时该项计 0 并判 FAIL(那是划分错误,不是"没什么可测所以通过")。

**0B-2 任务效度**(deterministic provenance,零新增标注):从 NIAH **train** split 的 mutation log 生成四类对:

| premise | hypothesis | 标签 | 确定性依据 |
|---|---|---|---|
| needle | gold claim | SUPPORTS | injector 已验证 gold alias 在 needle 中恰好出现一次 |
| `cf::needle` | replacement claim | SUPPORTS | mutation 定义 |
| `cf::needle` | gold claim | **REFUTES** | 单答案假设 + 同机械类别异值替换 |
| needle | replacement claim | **REFUTES** | 同上 |

UNKNOWN 的 primary gate 只能用 0B-1 的 official NEI 类;跨 query 配对的"推定 UNKNOWN"只作 secondary proxy 报
abstention rate,不进 gate。

### 3.3 0B-2 阈值从选择器需求反推 [新增]

| 失效 | 后果链 | 指标 | 阈值 |
|---|---|---|---|
| `cf → gold claim` 误判 SUPPORTS | 毒进 gold 簇 ⇒ 条件 2 失效 ⇒ 不可踢 ⇒ harm 不降 | twin REFUTES accuracy | **≥ .70** |
| 含 gold 的段漏判 SUPPORTS | gold 票低估 ⇒ needle 孤立(support=1≤cap) ⇒ 被踢 ⇒ recall 掉 | gold-passage SUPPORTS recall | **≥ .85** |

精确口径(避免歧义):

- **twin REFUTES accuracy** = 在 `cf_gold` 与 `needle_replacement` 两类对上预测 == REFUTES 的比例。
- **gold-passage SUPPORTS recall** = 在 `needle_gold` 类对上预测 == SUPPORTS 的比例。
- **两者的 UNKNOWN 一律计失败。** 否则"把什么都判 UNKNOWN"的模型能刷爆前者 —— 与 G-FC 必须做联合门是同一个可操纵性。
- `cf_replacement` 类对作背景报告,**不进任何阈值**。

`.70` 的依据:当前池内 `1 − missed_conflict` = .57–.69,低于 .70 连"不比现状差"都保证不了。

**S6 纪律(写死):** 0B-2 是**隔离对探针**,不许外推到池。池级判定只认 E1 `cluster_eval`,G-FC 护栏在那一层生效。

### 3.4 SAME_SOURCE:无条件修复 [实质修改]

**发现:** dpr-w100 是 **100 词切分**语料,一个 Wikipedia 条目对应**多个 `document_id`**,而
[base_loader.py:56](../../src/evidence_rag/materializer/base_loader.py#L56) 把 title 拼进了 `document.text` 首段。
故现行 `len({candidate.document_id})` 会把同一条目的多段算成多张独立票 —— 它连同条目转抄都拦不住,
而检索恰恰倾向于把同条目相邻 passage 一起召回。selector.md §3 那句
"`document_id` 去重只能拦共享出处的转抄"**高估了现有实现**。

**决策形式(重要):无条件采用,不设阈值分支。** 数 `document_id` 违反 `independent_support` **自身的定义**,
是正确性缺陷不是可调旋钮;用碰撞率阈值决定改不改,等于让正确性取决于缺陷有多严重。故:

- 冻结参数 `support_unit ∈ {"document", "parent"}`,**主口径 = `parent`**;`document` 保留仅为复现 S1–S6 旧臂。
- top-20 同 parent 碰撞率照测照报,但它是**背景量**,不是决策门。
- Graph 1.0-lenient 须补跑一条 `support_unit=parent` 基线臂 —— 主对照两侧必须同 `support_unit`,
  否则 SAME_SOURCE 修复的收益会混进 NLI 的账上。
- 缺 sidecar 时请求 `parent` 必须**报错**,不得静默退回 `document`:静默退回会让一整轮实验测的是旧计票,
  而没有任何指标会发现。
- 未解析的 document 各自为父 —— 缺条目**只可能漏合并两个相关来源,绝不可能错合并两个真正独立的来源**。

**⚠ 依赖链(高亮):** title 解析不只修计票,它还是 §4 五轴审计**第二轴(父页面零重叠)的硬性前置**。
**即使 `support_unit` 保持 `document`,解析本身也不可省 —— 没有它 sealed 600 建不出来。**

### 3.5 G-FC — false_conflict 护栏 [来自 S6]

**为什么要它:** false_conflict 是 E2 那 −4.8pp recall 代价的**机制通道**(S2),而 S6 证明它**极易被推高**:
8B+decoupled 把它从 .49 抬到 .88,同时 missed 的改善还没确立。

**主口径 = 固定分母。** `N_eligible` = 注入题中"needle 在窗口内 **且** 窗口内另有至少一段文本含 gold 别名"者
—— 这两个条件**只依赖池与文本,不依赖任何系统**,故两臂恒同。弃权的段计为**不制造冲突**;
**needle 自己弃权时计 False 而非剔出分母**(剔出会让分母重新依赖抽取器,正是本指标要躲开的陷阱)。
**次口径 = 现行条件性 rate**,并列报告以对照 S1–S6 历史数字。两个口径都必须同时报 rate 与绝对题数。

**必须联合判读:** 主口径把弃权算作"不制造冲突",所以**一个把什么都判 UNKNOWN 的系统能刷爆 G-FC**。
故 G-FC 必须与 `needle_gold_recovery` 和 G-AB 的 abstention rate **三者联合**判定,预注册为**联合门**。

**阈值规则(冻结):** `δ = max(0.05, R003 在固定分母口径下算出的配对 MDE)`。守卫不能定得比能检出的还紧;
`0.05` 的下限依据是 S6 那次 +38pp —— δ=.05 拦下它绰绰有余。
若 R003 算出 MDE > .05,必须**在冻结前**加样本或放宽 δ,**绝不事后调**。
形式:Graph 2.0 的主口径 false_conflict 点估计不得超过基线 + δ,且 95% CI 下界不得高于基线点估计。

**基线已实测(R001,2026-07-31,job 18225682,commit 10289bc):**

| | 值 |
|---|---|
| `lenient.fixed_false_conflict` 基线 | **0.4355**(429 / 985) |
| 95% Wilson CI | [0.4049, 0.4667] |
| `N_eligible`(固定分母) | **985** |
| 对照:条件性口径 | 0.5132(429 / **836**) |

**分子完全相同(429),差异全部来自分母。** 条件性口径剔出的 149 道题在固定口径下计为"未制造冲突"
—— 正是该口径存在的理由,也说明两个口径的差不是重新打分造成的。

**δ 的当前取值:暂定 0.05,待 R003 的配对 MDE 确认。** 依据:在 p=.4355、n=985 上,
**非配对**正态近似的 MDE 为 **.063(power .80)/ .072(power .90)**,均高于 .05 下限。
配对检验(同一批 query,sign-flip)比这更灵敏,真实 MDE 取决于两臂的 discordance rate,
而那需要一条真实的 Graph 2.0 臂才能测,现在还不存在。
故:**R003 跑出配对 MDE 之前,δ 不得视为已冻结**;若配对 MDE > .05,按上面的规则 δ 随之上调。

**判定阈值(数值形式,R003 确认后生效):** Graph 2.0 的固定分母 false_conflict
点估计 ≤ **0.4855**(= .4355 + .05),且其 95% CI 下界 ≤ **0.4355**。

### 3.6 G-AB — 弃权必须被度量 [来自 S6]

准确版本:**抽取层名义上有 NONE 分支,但弃权不被度量、也没有守门**,所以它被 target priming 压掉时,
没有任何东西拦住实验。故冻结:abstention / UNKNOWN rate 与 edge coverage 是**一等报告项**;
Gate 0B 未过时**只修 Relation Builder**,不得进入 Selector 确认性测试;
official test 失败后不得反复调参再重测并仍称"一次性确认"。

### 3.7 G-PQ — per-query outcomes 必须落盘 [来自 S6 的操作性缺陷]

任何进入 Graph 2.0 结论的运行,**必须同时落盘 per-case 行(含原始抽取答案),使重打分与配对检验成为 CPU 级操作**。
已实现:`cluster_eval_cli --dump` + `cluster_rescore_cli`(含 per-query 映射,直接喂 `paired_metric`)。

另:**CI 重叠 ≠ 无效应**。凡未做配对检验的指标,报告表述只能是"未确立",不得写成"无改善"或"有改善"。

### 3.8 训练路径(仅当 0B-1 或 0B-2 未过才启动)

预注册,不看结果不改:基座 `microsoft/deberta-v3-base`,脚手架用 sentence-transformers CrossEncoder 三类范式;
VitaminC 主训(revision-family 去污染,official test 一动不动,删除清单存档)+ NIAH train 的 mutation-log 对做域适配;
全链路按 parent page + synthetic family 做 5-fold OOF;三 seed 13/42/73;official test 只跑一次。
硬约束:NIAH 域适配对的 parent page 必须与 sealed-600 零重叠。
**副作用:** 零训练路线下模型从未见过 NIAH 语料 ⇒ passage-hash 泄漏轴天然为空。

---

## 4. 数据与划分冻结

- **fresh NIAH sealed 600** 构建算法沿用 TRAINING_PLAN §4.2 全文。
- **规模反推:** dev 的 2000 采样落 1479 注入,skip rate .261。落 600 注入题需起始 `600/(1−.261) ≈ 812` 合格 query,
  **取 900 留余量**。写进 manifest;不得跑到一半发现不够再补(补样本 = 看结果后改数据)。
- **五轴泄漏审计:**

| 轴 | 判据 | 来源 |
|---|---|---|
| query | query_id 与规范化 query 文本双重零重叠 | queries.jsonl |
| 父页面 | 规范化 title 零重叠 | documents.jsonl 首段(§3.4 sidecar) |
| answer entity | 规范化 gold value 零重叠 | qrels answers |
| passage hash | gold passage 文本 sha256 零重叠 | documents.jsonl |
| synthetic family | `(gold_value, replacement_value, 机械类别)` 三元组零重叠 | mutation log |

- **检索器冻结 = bm25**(与 E2 同):Graph 2.0 的主张是"给定固定候选池,选择器更可靠",固定候选池正是冻结的模块接口;
  换检索器会把池的组成变成混淆变量,且 E2 exact 臂的 dump 只有在池相同时才可复用。
  队友的 StrongBM25/hybrid 作**独立泛化臂单列一张表**,不与主表合并、不进 C1 判定。
- **语料重建(不共享现有 100k 子采样):** 现有语料的 gold doc 是按 dev 那 2000 query 的 qrels 选进去的,
  sealed-600 新 query 的 gold 大概率不在 ⇒ 共享方案本身不成立。按 sealed-600 的 qrels 保留全部 gold doc +
  重新蓄水池采样 distractor,seed 固定并记 hash。
- **现状缺口(必须写进报告):** 目前**没有 sealed test** —— S1–S6 全部结论都在 dev 上。
- **不看结果承诺:** manifest 与 hash 在任何训练/dev 实验之前冻结。
- FinanceBench 不进入 Selector V2 的任何环节(团队约定)。

---

## 5. 主张、统计与样本量

### 5.1 C1 重写(D1=A 下)

主对照 = **同池配对的 Graph 2.0 (gate-on) vs Graph 1.0-lenient (gate-on)**,唯一变量 = 边的构造;
外加 gate-off 作绝对参照。三联判定:

- ΔHarmful@10 ≤ −0.02 且双侧 95% CI 上界 < 0
- Required Recall@10 单侧非劣下界 ≥ −0.01(**两个基准,见 5.2**)
- NDCG@10 单侧非劣下界 ≥ −0.01
- G-FC:固定分母口径 false_conflict 不得劣化超 δ(§3.5)

三项必须同时满足,不得用次级指标的改善替代失败的主指标。

### 5.2 recall 非劣必须对两个基准同时报 [冻结]

- 口径 (a) vs Graph 1.0-lenient:证明"换边没让事情变糟"。
- 口径 (b) vs gate-off:**这才是真正要闭合的守卫**。Graph 1.0-lenient 在此为 −3.6pp,**FAIL**。

**头条判据是 (b),不是 (a)。** §2.2 的机制预测说的正是 (b) 应当收窄:needle 不再需要自己抽出 gold 字符串,
只需支持池内已被提出的 gold claim ⇒ `support(K)` 升过 `support_cap` ⇒ 结构性不可踢;
而 cf 的 replacement claim 仍只有自己支持 ⇒ margin 变大 ⇒ 对毒更易 fire。
**预测:harm↓ 与 recall↑ 同向发生。** 若实测为"harm↓ 但 recall 也↓",方案的机制假设**被证伪** —— 如实写。

**只报 (a) 通过就宣布成功,等于把未闭合的守卫藏起来。禁止。**

### 5.3 样本量与 power

沿用 TRAINING_PLAN §4.3:统计单位 = query;baseline exposure .68、目标 .48、双侧 α=.05、power=.90,
保守双比例正态近似 ≈ 每组 126 query;实际主比较在同一 600 题上配对完成。公式、输入参数与可复现脚本必须存档。
**新增:G-FC 需要自己的 MDE**(§3.5),在 R001 基线实测后按固定分母口径补一份 power/MDE 计算。

### 5.4 废除三-seed ensemble [实质修改]

**greedy decoding 已核实(2026-07-30):** [composition.py:336](../../src/evidence_rag/composition.py#L336)
用 `GraniteLLMClient()` 无参构造 → [granite.py:39](../../src/evidence_rag/generator/granite.py#L39) `temperature = 0.0`
→ [granite.py:121](../../src/evidence_rag/generator/granite.py#L121) `do_sample = False`,`num_beams` 未设(HF 默认 1)。
纯 greedy argmax,无采样无 beam。

> 由于 D1 决议采用零训练 NLI 边构造与确定性的四条件门控,选择器与关系层均无学习参数;抽取层经核实采用 greedy argmax
> 解码,不含采样。全链路**无随机源**,seed 在本系统中没有作用点,因此废除原计划的 3-seed ensemble 机制。

**不得写成"100% deterministic / 逐位可复现"。** `dtype="auto"` + `device_map="auto"` 下模型以 bf16/fp16 运行,
GPU kernel 选择与浮点规约顺序可在近似平局处翻转 argmax ⇒ 跨硬件重跑可能出现少量不同答案。
准确表述是"**无随机源**",不是"可复现到位"。

**替代物(取代而非删除):** **rerun-stability 检查** —— 同硬件重跑 100 题抽取,报答案级一致率与翻转样例。
比 seed ensemble 更贴近真实风险源,成本近零。列为 M4 冻结前的必做项。

若 §3.8 训练路径被启动,三 seed 条款**恢复生效**(仅对关系模型)。

---

## 6. Gate 0A 审计清单(退出判据,零违规)

- [x] §1 的 D1、D2 已填(D1=A;D2 记消解理由)
- [x] §2 关系 schema、两条新语义、`conflict_mode`、argmax 与平局裁决已冻结
- [x] §3 Gate 0B 两层、0B-2 阈值与反推依据、SAME_SOURCE 决策形式已冻结
- [x] §4 检索器、语料、规模反推、五轴判据已冻结
- [x] §5 C1 重写、recall 双基准、seed 条款处置已冻结
- [ ] **§3.5 的 G-FC 基线与 δ 已带数值(依赖 R001)** ← 转 FROZEN 的唯一剩余阻塞
- [ ] primary label 的 provenance 只能是 `official` 或 `deterministic_rule`
- [ ] split / group / hash 零重叠(五轴)
- [ ] 每题 candidate count、Top-20 ID、utility range、derived flags 一致
- [ ] counterfactual mutation 可逆,且 gold alias 无残留
- [ ] fresh 600 manifest 在运行前冻结,hash 已记录
- [ ] 不存在人工标注待办(零新增人工标注约束)
- [ ] 本文件协议版本号 `g2-proto-1` 与 hash 已记入 [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md) R000

任何一条违规:停止,重建数据或补齐决定,**不得带着 TODO 进 M1**。

---

## 7. M0 的四个 Run

| Run | 内容 | 依赖 | 产物 |
|---|---|---|---|
| R000 | 冻结 protocol / schema / splits(本文件转 FROZEN) | R001 | 协议版本 + hash |
| R001 | **G-FC 基线实测**(按 §3.5 固定分母口径重算 Graph 1.0 + lenient) | 一轮带 `--dump` 的 3B+single E1 | 基线值 + δ,回填 §3.5 |
| R002 | ~~冻结 NIAH train 规模~~ | — | **已消解**(§1 D2) |
| R003 | paired Monte Carlo 样本量 sensitivity + G-FC 的 MDE | R001 | power / MDE 脚本与结果 |

**关键路径 = R001。** 工具已就位;只差一次 GPU 抽取轮次。

---

## 8. 未决事项汇总

1. **G-FC 的基线与 δ** —— 依赖 R001,是转 FROZEN 的唯一剩余阻塞。
2. **sealed 600 尚未构建** —— 依赖 §3.4 的 title sidecar(已实现)与语料重建。
3. 门决策记录是否升级进 `PipelineRun` trace —— 契约变更,另走流程,不阻塞 M0。
5. **修订案 A2 待起草** —— 0B-1 在二分类空间下的定义。0B-1 已于 2026-08-04 挂起(§9.11),
   在 A2 落地前 Gate 0B 只有任务层证据,且 UNKNOWN 无 primary gate、§9.5a 的补救无扳机。
4. ~~修订案 A1(§9)待批准~~ —— **已于 2026-08-03 批准,g2-proto-2 生效。**
   §9.10 尚余两项未完成:关系模型输出空间的代码改动(§9.1)与 G 模块的后端/阈值复用共识。
   **在代码改动落地前,`relations/` 仍按三类实现运行**,故 R012 的三类读数继续可复现。

---

## 9. 修订案 A1 — 关系判定改为二分类支持判断

**提出日期:** 2026-08-03 | **批准日期:** 2026-08-03 | **状态:** **已批准,g2-proto-2 生效**

批准不改变 §9.0 的披露:本修订仍是在看到 R012 判 FAIL 之后提出的,该事实随协议长期保留。

### 9.0 诚实声明:本修订是在看到结果之后提出的

本文件第 22 行写着:

> 冻结之后任何改动都必须新开协议版本号、写明原因与时间,**不得在看到结果之后追加或调整**。
> 这条是 S4/S5 两次自我推翻换来的纪律。

**本修订正是在看到 R012 判 FAIL 之后提出的,并且它会使门更容易通过。** 这一点不因下文理由充分而消失,
必须写在最前面而不是附注里。事实序列如下,不作美化:

| 时间 | 事件 |
|---|---|
| 2026-07-30 | 设计文档写下"主口径下 REFUTES 边只被记录不被门消费";§3.3 写下"UNKNOWN 一律计失败"及其防刷分理由 |
| 2026-08-01 | R012 执行,三类口径判 **FAIL**(job 18235972) |
| 2026-08-03 | R012b rung 1/2 + `--dump` 得到逐类混淆矩阵 |
| 2026-08-03 | 组内提出"关系判定能否改二分类";**此后**才回查设计文档,发现门不消费 REFUTES。
即修订动机并非从结果中挖出,但**核查发生在结果已知之后** |
| 2026-08-03 | **在提出修订之前**已算出二分类下 twin 四种组合全过(albert **.9980**)——即本修订的提出者
已知它会使 twin 项从 FAIL 变 PASS。(该项**并未**使 Gate 整体通过,见 §9.5) |

**本文件的作者无法主张"未受结果影响"。** 能主张的只有:下述三条依据各自独立可查,且**依据本身早于数据**。
读者应当以此为限来判断本修订的可信度。

**在此前提下,以下三项事实与上述披露同样成立,应当一并权衡(而非在后文分散呈现):**

1. **依据早于数据且可用 git 独立核验。** "主口径下 REFUTES 边只被记录不被门消费"写在
   commit **`10a1314`**(2026-07-30,设计文档),彼时 Gate 0B 一个数字都不存在。
   审阅者不必采信本文件的自述,`git show 10a1314` 即可自行确认。
2. **本修订未产生任何一个新的通过者。** argmax 口径下四种(臂 × rung)组合**仍全部未过**
   `gold_supports_recall`(最高 .7942 < .85)。修订把 FAIL 的原因从两项收敛到一项,
   **没有把任何 FAIL 变成 PASS**(§9.5)。这一条是可证伪的:任何人重跑 dump 即可检验。
3. **本修订把推翻自己的权力交给了尚未产生的数据。** §9.8 指定 MiniCheck-FT5 与 rung 3 为出样检验,
   二者结果在起草时均不存在,且明文规定"若出样结果与本修订预期相悖,以出样结果为准"。

**这三条不抵消 §9.0 的披露,它们各自独立。** 本修订的可信度应由"依据是否真的早于数据"、
"是否真的没有制造通过者"、"出样检验是否真的被执行并被采信"三问共同决定,而不由本文件的措辞决定。

### 9.1 修订内容

| | 现行(g2-proto-1) | 修订后(g2-proto-2) |
|---|---|---|
| 关系模型输出空间(§2.1) | SUPPORTS / REFUTES / UNKNOWN 三类 | **SUPPORTED / NOT-SUPPORTED 二类** |
| 0B-2 twin 指标(§3.3) | `twin REFUTES accuracy` = 预测 == REFUTES 的比例,UNKNOWN 计失败 | **`twin NOT-SUPPORTED accuracy`** = 预测 != SUPPORTED 的比例 |
| 0B-2 gold 指标(§3.3) | `gold-passage SUPPORTS recall` | **不变** |
| 阈值 | twin ≥ .70,gold ≥ .85 | **不变**(依据见 §9.5) |
| 联合门 | 两项须同时过 | **不变,且成为防刷分的唯一机制**(见 §9.3) |

`CLAIM_REFUTES` 作为 schema 中的边类型**保留但不再由关系模型产出**,直至有二分类以外的证据需要它。

### 9.2 依据一:门在主口径下从不消费 REFUTES(设计文档,2026-07-30)

§2.3 与设计文档 §2.3 同一句、逐字一致:

> `refutes_edge`(**消融臂**):竞争簇还须有成员向 c 的 claim 发出 REFUTES 边。……
> **主口径下 REFUTES 边只被记录不被门消费。**

且 `independent_support(K)` 的定义只数 **SUPPORTS** 边;主口径 `conflict_mode=distinct_cluster`
用的是簇成员关系,不是 REFUTES 边。**三类中的第三类,主口径一次都没读过。**

### 9.3 依据二:§3.3 要求三类的真实理由是防刷分,而联合门已覆盖它

这条是本修订必须正面回应的,因为它比依据一更硬。§3.3 原文:

> **两者的 UNKNOWN 一律计失败。** 否则"把什么都判 UNKNOWN"的模型能刷爆前者
> —— 与 G-FC 必须做联合门是同一个可操纵性。

即:三类不是为了让门消费 REFUTES,是为了让 twin 指标不可刷。**这个担心是对的。**

回应:两种退化策略在**联合门**下都已被挡住,不需要额外靠三类来挡——

| 退化策略 | twin(二分类) | gold_supports | 联合门 |
|---|---:|---:|---|
| 全判 UNKNOWN / NOT-SUPPORTED | 1.000 | **0.000** | **挡住** |
| 全判 SUPPORTED | **0.000** | 1.000 | **挡住** |

§3.3 自己把这个可操纵性类比为 "与 G-FC 必须做联合门是同一个",而联合门正是既有设计。
故三类在此处是**冗余的第二道防线**,不是唯一防线。

**但要如实指出该冗余防线并非全无价值:** 三类能区分"弃权多"与"承诺错",而二分类不能。
弃权多会导致图稀疏——不过那恰好由 `gold_supports_recall` 度量,仍在联合门内。

### 9.4 依据三:同仓库 G 模块已在生产上采用二分类,且有六臂实测

`src/evidence_rag/generator/nli.py` 的生产默认值与其理由(该文件早于本修订):

> `DEFAULT_NLI_BACKEND = "true"` —— TRUE is the default because it won the G1 six-arm triage on both
> axes that matter — ASQA entailment recall **0.747** and derived citation precision **0.966**,
> vs MiniCheck 0.620 / 0.886 — and its probability sweep is usable where **Granite's degenerates**.

该文件另有注释称 TRUE 与 MiniCheck 为 "the two **binary (support-probability) backends**"。
即:**同一仓库内,对同一子问题(证据是否支持某 claim),G 模块经六臂实测已选定二分类后端。**
现状是两个模块对同一子问题给了不同答案,而只有 G 那边有实测支撑。本修订使二者收敛。

### 9.5 阈值处置:不变,但必须承认它变得几乎不可区分

`.70` 的原始依据(§3.3)是:"当前池内 `1 − missed_conflict` = .57–.69,低于 .70 连'不比现状差'都保证不了。"
**这条推理不依赖三类还是二分类**——它是"必须胜过现状池的冲突检出率",故阈值沿用 `.70`,不上调。

但必须写明代价。逐 pair dump 实测(2026-08-03,`results/gate0b/dump-{template,qa}.jsonl`):

| 臂 / rung | twin(三类,现行) | **twin(二分类,修订后)** | gold_supports_recall |
|---|---:|---:|---:|
| albert / rung 1 `template` | .6736 ✗ | **.9980** ✓ | .1916 ✗ |
| albert / rung 2 `question_answer` | .7602 ✓ | **.9871** ✓ | .3635 ✗ |
| DeBERTa / rung 1 | .6376 ✗ | **.8689** ✓ | **.7942** ✗ |
| DeBERTa / rung 2 | .5547 ✗ | **.9008** ✓ | .6651 ✗ |

**twin 项已接近饱和,几乎不再有区分力**,它从"选型判据"退化为"下限守卫"。0B-2 的实际选型压力
**此后完全落在 `gold_supports_recall ≥ .85` 一项上**,而四种组合中最高者仅 **.7942**
(DeBERTa / rung 1,且该臂另背大小写减分,见 hpc-run-log R012 AFTER)。

**修订后的门并未变成谁都能过 —— 四种组合仍无一通过。** 这是本修订不构成放水的直接证据:
它把 FAIL 的原因从两项收敛到一项,并未产生任何一个通过者。

### 9.5a A1 的适用范围严格限于 argmax;引入阈值须另开修订 [防护条款]

上一段"四种组合仍无一通过"**成立的前提是 §2.4 的 argmax 主口径、无阈值**。本条款存在的理由是
一个已经发生的事实,记录如下,不作淡化:

2026-08-03,在 A1 起草**之后**,对 dump 中的 `P(SUPPORTS)` 做了一次全量阈值扫描(原意是诊断
"概率质量是否堆在决策边界"),结果显示 **DeBERTa / rung 1 在 θ ≤ .10 时两项同时通过**
(θ=.05:gold .9015 / twin .7374;θ=.10:gold .8716 / twin .7897)。

**该阈值不得采用,A1 也不得被解读为允许引入阈值。** 三条理由:

1. 采用它会使 §9.5 上一段当场变假,**A1 赖以自证不放水的唯一直接证据随之作废**。
2. θ 是直接对着 Gate 结果调出的自由参数,除"它能过"外无任何独立依据 ——
   其性质比 A1 本身严重一个量级。
3. **实质理由:** θ=.05 意为"有 5% 概率即判支持",在真实 20 段池中将产生大量假支持边,
   而该失败恰由 §3.5 G-FC 护栏承接。§3.3 的 **S6 纪律**写死 0B-2 是隔离对探针、不许外推到池,
   故在 0B-2 上把阈值调到过关,**不构成系统可工作的任何证据**,只是把失败移交给下游护栏。
   通过窗口 θ∈(.05, ~.15) 亦位于量程边缘,鲁棒性存疑。

**第四条理由(2026-08-03 补,来自代码而非本文件):** 阈值补救**本就是预注册过的**,但其触发条件不是本次情形。
`src/evidence_rag/relations/predictor.py` 的模块 docstring(早于本修订)写明:

> argmax is the PRIMARY head: there is no confidence threshold anywhere on the path that can drop a
> candidate, so the gate carries no absolute cross-domain scale. That is the property separating this
> design from credibility-threshold arbitration, and it is why UNKNOWN is a predicted class rather than
> "score below tau". **Threshold gating exists only as a pre-registered remedy if Gate 0B REFUTES
> precision falls short**, and invoking it must be declared in the report.

即预注册允许引入阈值的**唯一触发条件是 REFUTES precision 不足**。实测 0B-1 的 `refutes_precision`
albert **.9026**、DeBERTa **.9127**,**两臂均高于 .85 阈值 —— 触发条件根本没有满足。**
以 `gold_supports_recall` 不足为由引入阈值,**不在该预注册补救的范围内**,属新增自由参数。

同段还指出 argmax 无阈值是"使本设计区别于 credibility-threshold 仲裁"的**定义性属性**。
A1 改二分类后该属性仍成立(二分类 argmax 依然不含阈值),但**一旦引入 θ 即告丧失**——
这不只是纪律问题,是设计身份问题。

**若日后确需操作阈值**(G 模块的二分类后端即带阈值,故这是合理的未来方向),必须满足全部四条:

- 另开修订案(A2),不得并入 A1;
- θ **在标定集上选定**,评测集不得参与。既有机制可用:`ProbePair.group`(synthetic family)
  本就是为 leakage 切分而设,按 family 切标定半 / 评测半即可,同一 family 不得跨界;
- θ 在**看到评测半结果之前**写死并预注册;
- 2026-08-03 这条全量扫描曲线**已被污染,只能作为刻画呈现,不得用于选定 θ**。

**可正当报告的部分:** 该扫描给出一条独立于门成败的发现 —— 三类 argmax 严重低报了模型已有的信息。
albert / rung 1 的 gold_supports_recall 由 argmax 的 **.1916** 升至 θ=.05 的 **.7018**(3.66×),
rung 2 由 .3635 升至 .7799(2.15×)。即该 checkpoint 的失败**不是"判不出",而是"不肯承诺"**。
这一点对 §3.8 是正面信号:概率质量已在正确位置的模型比真正混淆的模型更易微调。

### 9.6 本修订不改变什么

1. **R012 的三类 FAIL 是预注册主结果,永久保留,不因本修订改写或撤回。**
2. 二分类读数一律标注为 **2026-08-03 事后重分析**,任何场合不得单独作为"Gate 0B 结果"呈现;
   两种口径必须并列,并附 §9.0 的时间线。
3. §3.5 G-FC、§3.6 G-AB、§3.7 G-PQ 三条护栏不动。
4. S6 纪律不动:0B-2 仍是隔离对探针,不许外推到池。

### 9.7 代价与因此失效的条目

- **`conflict_mode=refutes_edge` 消融臂失去可执行性**(无 REFUTES 边可用)。
- **TRAINING_PLAN Block 3 的"去掉 CLAIM_REFUTES"消融失去实质含义。**
- **EXPERIMENT_TRACKER 的 R036(CLAIM_REFUTES 消融)整行变空**,须显式标 N/A 并注明因本修订消解。
- **叙事代价**:"带类型边的关系图"弱化为"支持计数图"。如实记录:主口径本就只用 SUPPORTS,
  本修订**暴露**了这一点而非造成它。这对 report 是减分项,但比被 reviewer 问出来好。

### 9.8 出样验证(本修订可信度的真正来源)

本修订由 albert / DeBERTa 两臂的数据促成,故这两臂**不能**用来验证它。以下两项在提出本修订时
其结果尚不存在,构成真正的出样检验,**须在其结果产生后复核本修订是否仍然成立**:

1. **MiniCheck-FT5 臂**(预注册三臂中从未上场的一臂)。它原生二分类,在本修订下无须双向探测即可直接上场
   ——这本身是本修订的一项独立收益。
2. **R012b rung 3(QA2D)。**

若二者在二分类口径下仍双双卡在 `gold_supports_recall`,则本修订不构成"改指标换通过",证据链完整。
**若出样结果与本修订预期相悖,以出样结果为准。**

### 9.9 利益冲突声明

本修订使门更容易通过,而本项目在门通过一事上有直接利益(§3.8 训练路径的启动与否、报告的成败)。
提出者对此有明确认知。缓解措施为 §9.0 的时间线披露、§9.6 的主结果保全、§9.8 的出样检验三项。
**审阅者应默认本修订带有确认偏误,并据此加重审查。**

### 9.10 批准所需

- [x] 项目负责人批准并将本文件版本号改为 `g2-proto-2` —— **2026-08-03 完成**
- [x] `EXPERIMENT_TRACKER.md` 的 Protocol 行同步改版本号,R036 标 N/A —— **2026-08-03 完成**
- [x] R012b 的 AFTER 中两种口径并列 —— **2026-08-03 完成**
- [ ] `relations/models.py` 的输出空间与 `relations/gate0b.py` 的指标实现按 §9.1 修改(TDD)
      —— **未完成,且按 §9.10a 必须推迟。** 波及 `predictor.py` 的三类 tie-break、
      `gate0b.py` 的 `LABEL_ORDER` 三元组、`task_probe.py` 的 REFUTES 标签、
      `vitaminc.py` 的标签映射与 `graph.py`。

### 9.11 0B-1 挂起,待修订案 A2 [裁决 2026-08-04]

**A1 只改了 0B-2,没有规定 0B-1** —— 而 0B-1 的五项冻结阈值建在三类分割上。实现落地后实测:
用一个**完全正确**的二分类打分器对 VitaminC gold 打分,得

| 指标 | 值 | 状态 |
|---|---:|---|
| `refutes_precision` | 0.0 | 自动 FAIL —— 模型永不预测 REFUTES |
| `refutes_coverage` | 0.0 | 自动 FAIL —— 同上 |
| `macro_f1` | 0.5 | 自动 FAIL —— REFUTES 的 f1 结构性为 0,拉死均值 |
| `non_unknown_coverage` | 1.0 | **空洞 PASS** —— 模型永不预测 UNKNOWN,该项什么都没测 |
| `support_coverage` | 1.0 | 唯一仍在测量的一项 |

**裁决:0B-1 挂起,标 N/A,待 A2。** 即 §9.1 讨论过的三个选项中的 (c)。
`external_report`、其五项阈值与 `vitaminc.py` **逐字节保留不动**;实现中以
characterisation 测试钉住该不一致,并在 sweep payload 中加 `external_tier_status` 字段,
使退化数字**不会不带标记地流传**(该字段不选择任何一个选项,A2 落地后移除)。

**挂起的代价必须与裁决同时记录,它不是免费的:**

1. **验收证据只剩任务探针。** 而 §3.2 设两层的理由正是"探针单独可被合成 mutation 模式刷" ——
   挂起 0B-1 就是暂时接受这个风险。
2. **UNKNOWN 失去 primary gate。** §3.2 明文:"UNKNOWN 的 primary gate 只能用 0B-1 的 official NEI 类"。
   叠加 A1 使 NOT_SUPPORTED 吞并 UNKNOWN(§9.1 实现裁决 1),弃权度量在本轮**没有主门可依**。
3. **§9.5a 的预注册补救失去扳机。** 该条款规定阈值补救"仅当 0B-1 的 REFUTES precision < .85 时启用",
   而该量此后**永远不可求值** —— 设计里唯一的预注册逃生口因此悬空。§9.5a 关于"触发条件从未满足"
   (实测 .9026 / .9127)的论证,其依据自此没有后继测量。

**A2 必须解决的问题(待起草):** 0B-1 在二分类输出空间下如何定义。§9.1 记录的三个选项各有代价 ——
(a) 二值化 0B-1 的 gold:`.85/.80/.70` 是在三类量上标定的,沿用即**静默重标定**,且
`non_unknown_coverage` 无后继;(b) 0B-1 保持三类、跑原生头:**认证外部效度的函数不再是建边的函数**,
直接抵消 §3.2 设两层的理由;(c) 永久取消该层:需重新论证 §3.2。
**A2 与 A1 同规格**:须写明时间线、利益冲突、以及是否使任何已 FAIL 的项变 PASS。

### 9.10a 代码改动的排序约束:必须在 R012b 阶梯跑完之后 [2026-08-03]

**§9.1 的代码改动不得在 R012b rung 3 完成之前落地。** 这不是保守,是它会**破坏 R012b 自身的可比性**:

1. ~~**已生成的 pair 文件会失效。** `task_probe.py` 为孪生对产出 `REFUTES` 标签,而
   `gate0b.py` 以 `RelationLabel(row["label"])` 读回。输出空间一改,现有 pair 文件直接解析失败。~~

   **[2026-08-04 更正 —— 本条为事实性错误,已由实现实测推翻。]** A1 §9.1 **保留** `CLAIM_REFUTES`
   于枚举中,且两个 0B-2 指标都是对 **SUPPORTS** 定义的、从不读 gold 的 REFUTES 列。故旧 pair 文件
   **照常解析,且新旧文件产出逐字相同的报告**(已由 characterisation 测试钉住)。
   **rung 3 不需要重新生成 pair 文件即可与 rung 1/2 可比。**
   本条推理作废;下列第 2、3 条不受影响,推迟本身仍然正确。
2. **rung 1/2 已在三类 argmax 下跑完**(job 18246568 / 18246569)。rung 3 若在二分类下跑,
   twin 指标两边口径不同,增量无法解读。
3. **R012 是预注册主结果**,其三类读数须保持可原地复现,不应退化为"需 checkout 旧 commit"。

**正确顺序:** rung 3 在现行三类实现下跑完 → 三级阶梯完整 → 再落地 §9.1 →
之后如需二分类读数,**从各 rung 的 `dump-*.jsonl` 重算**(重算与"模型原生二分类"在
collapse 语义下等价,且不动任何已产生的实验数据)。

**例外:** 若 MiniCheck 臂(§9.8 的出样检验之一)先行就绪,它**原生二分类**,应作为独立臂
单独报告,不并入 R012b 的三级阶梯比较。
- [ ] G 模块就后端与阈值复用达成一致(负责人:待定)
