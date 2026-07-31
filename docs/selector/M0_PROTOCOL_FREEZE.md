# Graph 2.0 — M0 协议冻结与预注册

**日期:** 2026-07-30 | **状态:** **DRAFT — 决定已全部填,等 R001 的 G-FC 基线数值后转 FROZEN** | **协议版本:** g2-proto-1

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
