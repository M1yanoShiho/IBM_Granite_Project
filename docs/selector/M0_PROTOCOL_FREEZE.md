# Graph 2.0 — M0 协议冻结与预注册

**日期:** 2026-07-30 | **状态:** **DRAFT — 决定已全部填,等 R001 的 G-FC 基线数值后转 FROZEN** | **协议版本:** **g2-proto-4**(2026-08-06 经修订案 A3 升版,见 §11)

**版本沿革:** `g2-proto-1` → `g2-proto-2`(A1,2026-08-03,§9:关系判定改二分类)→ `g2-proto-3`(A2,2026-08-06,§10:**把 A1 收窄至判读口径** —— 模型恢复产三类,门与 0B-2 仍读二类)→ **`g2-proto-4`**(A3,2026-08-06,§11:**§3.8 训练基座重新选型** —— 预注册基座只发布 `.bin`,在 `torch < 2.6` 下不可加载)。**A2 是对 A1 的部分撤回,不是澄清**;A1 的记录逐字保留于 §9。

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
| 对照 | MiniCheck-FT5(770M) | LLM-AggreFact <1B SOTA;**原生二分类,无须双向探测**(见下方更正) |
| 上界 | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | 通用 NLI 上界,非域内 |

**[2026-08-06 更正,A2 §10.7(B)]** 本表原写 MiniCheck"二分类,需否定 claim 双向探测,单独一步",
**该记述已被 R012c 的实现证伪,且它曾是 MiniCheck 被降级的理由。** 两点更正:

1. **双向探测从来不必要。** 二分类正是 A1 之后的判读口径,无 REFUTES 需反推。原句是 pre-A1 的记录。
2. **真正的阻塞是架构。** `lytang/MiniCheck-Flan-T5-Large` 是 `T5ForConditionalGeneration` 且**无 `id2label`**,
   `AutoModelForSequenceClassification` 根本打不开,`LABEL_ORDER` 对它无意义。故走独立的二分类注册表
   (`relations/minicheck.py`),两个注册表按架构分发。
3. **A2 之后的新增后果:** 原生二分类的臂**无三类输出,因此不具备 0B-1 认证资格**(§10.6 第 3 条)。

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
| `cf::needle` | gold claim | **NOT_SUPPORTED** | 单答案假设 + 同机械类别异值替换 |
| needle | replacement claim | **NOT_SUPPORTED** | 同上 |

**[2026-08-06,A2 §10.7(B)]** 后两行原标 `REFUTES`。改为 `NOT_SUPPORTED` 的裁决理由是**探针 gold 标签跟随指标** ——
twin 指标自 A1 起即为 `predicted != SUPPORTS`,而 `task_probe.py` 已产 `TWIN_NOT_SUPPORTED`。
两类对在语义上确实是真矛盾,但**指称必须与实际计算的量一致**:§9.12 那一整轮麻烦的根源正是文本指称与实际量脱节。

UNKNOWN 的 primary gate 只能用 0B-1 的 official NEI 类;跨 query 配对的"推定 UNKNOWN"只作 secondary proxy 报
abstention rate,不进 gate。

### 3.3 0B-2 阈值从选择器需求反推 [新增]

| 失效 | 后果链 | 指标 | 阈值 |
|---|---|---|---|
| `cf → gold claim` 误判 SUPPORTS | 毒进 gold 簇 ⇒ 条件 2 失效 ⇒ 不可踢 ⇒ harm 不降 | twin NOT_SUPPORTED accuracy | **≥ .70** |
| 含 gold 的段漏判 SUPPORTS | gold 票低估 ⇒ needle 孤立(support=1≤cap) ⇒ 被踢 ⇒ recall 掉 | gold-passage SUPPORTS recall | **≥ .85** |

精确口径(避免歧义;A1 §9.1 改指标定义、A2 §10.2 定折叠点,本段为二者落地后的**当前**口径):

- **twin NOT_SUPPORTED accuracy** = 在 `cf_gold` 与 `needle_replacement` 两类对上 `predicted != SUPPORTS` 的比例。
- **gold-passage SUPPORTS recall** = 在 `needle_gold` 类对上 `predicted == SUPPORTS` 的比例。
- **三类输出先按 §10.2 折叠再计分。** 折叠**取 max**,等价于三类 argmax 换标签;
  **取和等价于 `P(SUPPORTS) > .5`,是被 §9.5a 禁止的阈值**。
- **因此 UNKNOWN 在 twin 上计成功、在 gold 上计失败。** 这个不对称是 A1 的**已知代价**(§9.3:
  二分类判读不能区分"弃权"与"承诺相反"),不是笔误。**弃权本身仍被度量** —— 模型自 A2 起恢复产 UNKNOWN,
  `unknown_rate` 按 §3.6 并报。
- **防刷分不再靠"UNKNOWN 一律计失败",而只靠联合门**(A1 §9.3):全判 NOT_SUPPORTED 者
  twin 1.000 / gold 0.000,全判 SUPPORTS 者反之,两种退化策略都被挡住。
  **故两项不得单独报告,也不得单独设阈。**
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

预注册,不看结果不改:基座 **`cross-encoder/nli-deberta-v3-base`**
(**A3 / §11 改**;原文为 `microsoft/deberta-v3-base` —— 该 checkpoint 只发布 `pytorch_model.bin`,
在本项目 `torch < 2.6` 的 pin 下**不可加载**,故原配方按原文不可执行),
脚手架用 sentence-transformers CrossEncoder 三类范式;
VitaminC 主训(revision-family 去污染,official test 一动不动,删除清单存档)+ NIAH train 的 mutation-log 对做域适配;
全链路按 parent page + synthetic family 做 5-fold OOF;三 seed 13/42/73;official test 只跑一次。
硬约束:NIAH 域适配对的 parent page 必须与 sealed-600 零重叠。

**预注册的应急臂(A3 新增,§11.5 裁决):** 若三 seed 训完后 `gold_supports_recall` 仍未过 `.85`,
**允许且仅允许**改用同族同配方的 **`cross-encoder/nli-deberta-v3-large`**
(同为 SNLI + MNLI、Apache-2.0、语义相同的三类头,**仅规模不同**:24×1024 vs 12×768)重跑三 seed,
且**两者结果必须并列报告,不得只报后者**。除规模外任何一项配方改动仍须另开修订。
写在此处的目的与 §10.11 相同:该升级一旦发生,它是**被预先描述过的路径**,而不是一次事后换模型。

**§3.8 的四处空白已补齐并冻结 [2026-08-06 裁决,R013 开跑前]。** 实现 R013 时发现 §3.8 有四处未写明,
逐条裁决如下。**四项均为预注册决定,在第一次训练之前定死;此后任何改动须另开修订。**

**(a) 孪生对的训练标签 = `REFUTES`。** mutation-log 的孪生行 gold 标 `NOT_SUPPORTED`,而 A2 把它降为**派生标签**
—— 三类头没有对应 logit,故训练目标必须另行指定。裁决取 `REFUTES`,理由:那两类对(`cf::needle` × gold claim、
needle × replacement claim)在语义上**确实是真矛盾**(单答案假设 + 同机械类别异值替换);教模型区分"反驳"与"未知"
学到的推理更细;而**折叠时它依然计为不支持**,0B-2 两个指标(只问是不是 SUPPORTS)逐字不受影响。
**附带效果须一并记明:** UNKNOWN 此后**只由 VitaminC 的 NEI 类训练** —— 那正是 0B-1 需要它的地方 ——
而在决定 `gold_supports_recall` 的那些对上不被强化。这使 §10.11 的"UNKNOWN 质量堆积"风险降低,
**但不消除**,§10.11 的预注册应急路径继续有效。

**(b) 超参 —— 按公开标准值冻结,不调。**

| 参数 | 值 | 依据(与本任务结果无关) |
|---|---|---|
| optimizer | AdamW | 标准 |
| learning rate | **2e-5** | DeBERTa-v3-base 微调的公开标准值 |
| schedule | linear decay,warmup ratio **0.06** | DeBERTa / BERT 系列常规 |
| batch size | **32** | A100 40GB 切片在 184M × seq 256 下宽裕 |
| epochs | **2** | 基座已在 SNLI + MNLI 上训过,本轮是**继续微调**而非从零学 NLI;VitaminC train 约 37 万行,2 epoch 已足,更多起过拟合风险 |
| max_length | **256** | premise 是单段 ~100 词、hypothesis 是短模板;512 白烧一倍算力 |
| weight decay | **0.01** | 标准 |
| max_grad_norm | 1.0 | 标准 |
| precision | **bf16** | A100 原生支持 |

**必须写明的两点:** (1) 每个值的依据都是**该架构 / 该数据规模的公开惯例**,**没有一个是按"哪个更可能过门"挑的** ——
后者正是 §9.0 记录的那个病。(2) **本轮不存在可以调超参的合法面**:dev 是 §2.4 与 §11.2a 保留的唯一标定面,
(c) 又确认主训只用 train。故"取公开默认值并冻结"不只是纪律,**是唯一可行的做法**。
bf16 与 §5.4 的表述一致:全链路**无随机源**,但 bf16 下浮点规约顺序可在近似平局处翻转 argmax,
**不得声称"逐位可复现"**。

**(c) "VitaminC 主训"= 只用 train split,不含 dev。** dev **仅**用于跑去污染守卫,不进训练。
理由:dev 是 §2.4(τ 标定)与 §11.2a(基座选型的测量面)保留的**唯一合法标定 / 选型面**,折进训练即烧毁。

**(d) `id2label` 映射检查取严格拦截。** 三类头的标签映射一律**按名解析**;遇到
`LABEL_0/1/2`、二类头、重名、id 不是 `{0,1,2}` 一律**报错拒绝,不得回退、不得猜、不得自动补名**。
理由与 §11.9 第 4 项同源:猜一个顺序**不会报错**,只会静默产出可信的错数字。
**本项另补上 §11.9 第 6 项的一个洞:** 该项当时只验了 `CrossEncoder(num_labels=3)` **能构造**,
**没验构造之后 `id2label` 变成了什么** —— `num_labels` 会传入 `PretrainedConfig`,可能把具名映射替换成
`LABEL_0/1/2`。补验后若名字丢失,**拒绝执行而非重新附上名字**:那时 §11.6"头无须重新初始化"的前提已存疑,
属协议问题,不是代码可以自行处置的。

**副作用(A3 改写):** ~~零训练路线下模型从未见过 NIAH 语料 ⇒ passage-hash 泄漏轴天然为空。~~
所选基座已在 SNLI + MNLI 上训过,**该轴不再天然为空,必须实测**(§11.9 末项)。
NIAH 语料本身仍未被基座见过,但这一点**从此需要证据,而不再是构造保证**。

---

## 4. 数据与划分冻结

- **fresh NIAH sealed 600** 构建算法沿用 TRAINING_PLAN §4.2 全文。
- **规模反推:** dev 的 2000 采样落 1479 注入,skip rate .261。落 600 注入题需起始 `600/(1−.261) ≈ 812` 合格 query,
  **取 900 留余量**。写进 manifest;不得跑到一半发现不够再补(补样本 = 看结果后改数据)。

  **[2026-08-06 实测,构建完成后回填 —— 反推略乐观,余量吸收掉了:]**
  实际 `attempted 816`(非 812)、`realised_skip_rate` **0.2647**(非 .261)、`sealed 600`、`pool 900`。
  **差值来自本条的 `.261` 是在 2000 条*未经泄漏过滤*的 dev query 上量的**,而 900 是在过滤**之后**取的,
  两者不是同一个总体。**泄漏过滤自身的拒绝率此前从未量过,现已量到:**
  `considered 6515` 中拒绝 `4407`(≈68%)—— `leak_query_id 2000` / `leak_parent_page 1213` /
  `leak_answer_entity 1194`。**它没有在 `select_pool` 挂,但那是余量吸收的结果,不是反推准确。**
  **谁若按 812 取值将不足。** 本条保留原文以存证,下次重建须按 816 与 .2647 重算。
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
- [ ] 本文件协议版本号 `g2-proto-4` 与 hash 已记入 [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md) R000

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
2. ~~**sealed 600 尚未构建**~~ —— **已于 2026-08-06 构建**(`runs/niah-sealed600/`,`sealed 600` /
   `pool 900` / `documents 100600`,实测数字回填于 §4)。**Gate 0A 审计判 `INCOMPLETE`,exit 1,
   且这是正确结果**:五个泄漏轴**全部零重叠且两侧均非空**(query 1200/7798、parent_page 2148/20579、
   answer_entity 600/4247、passage_hash 2884/30054、synthetic_family 600/2679),
   manifest 六个产物哈希吻合,标签 provenance 与 600 个反事实的可逆性、无 gold 别名残留均通过;
   `utility_range` 判 `not_applicable`(D1=A 下无生产者,系 TRAINING_PLAN §5 的 LightGBM 时代遗留);
   **仅 `candidate_windows` 未评** —— §6 第 3 项需要冻结的 bm25 Top-20 窗口,而检索尚未跑。
   **待 bm25 跑完带 `--candidates` 重审方可能转 PASS;本实现不产生"带脚注的 PASS"。**
3. 门决策记录是否升级进 `PipelineRun` trace —— 契约变更,另走流程,不阻塞 M0。
4. ~~修订案 A1(§9)待批准~~ —— **已于 2026-08-03 批准,g2-proto-2 生效。**
   §9.10 的代码改动已于 2026-08-04 落地(commit `221c34a`),并由 A2 **部分回退**(§10.2)。
   **尚余一项未完成:G 模块的后端/阈值复用共识** —— 且 A2 已使该项的前提改变(§10.6 第 4 条)。
5. ~~修订案 A2(§10)待批准~~ —— **已于 2026-08-06 批准,g2-proto-3 生效。**
   0B-1 恢复、§3.8 配方冲突消解,**R013–R015 与 R020 的阻塞同时解除**。落地进度见 §10.10。

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

**[2026-08-06 后续 —— 本清单四条已反转三条,见 §10.4。原文一律不改写。]**
A2 把 A1 收窄至判读口径后模型恢复产三类,故前三条(`refutes_edge` 消融臂、TRAINING_PLAN Block 3、
tracker R036)**全部恢复**。**第四条不反转** —— 门仍然只读 SUPPORTS,本节"暴露而非造成"的判断仍然成立。

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
- [x] `relations/models.py` 的输出空间与 `relations/gate0b.py` 的指标实现按 §9.1 修改(TDD)
      —— **2026-08-04 完成**(commit `221c34a`)。§9.10a 的推迟条件在 rung 3 与 R012d 收口后清空,
      随即落地。波及 `predictor.py` 的 tie-break、`gate0b.py` 的 `LABEL_ORDER` 折叠、
      `task_probe.py` 的孪生标签与 `graph.py`,均已改;**`vitaminc.py` 与 `external_report`
      逐字节未动** —— 0B-1 按 §9.11 挂起。
      **折叠取 max 而非求和**(求和等价于 `S > .5`,是被 §9.5a 禁止的阈值);
      该等价性已由 `cli/recompute_binary --against` 对四个 dump 八格全部实证复现。
- [ ] G 模块就后端与阈值复用达成一致(负责人:待定)

**本清单曾于 2026-08-04 一度过时** —— 上面第 4 项在实现已落地、且同文件 §9.11 已记录该事实之后,
仍写着"未完成,必须推迟"。由 MiniCheck 臂的实现者发现并报告,本行为其更正记录。

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

**[2026-08-05 后续 —— A2 已起草,见 §10,本节保留原文不改写。]** A2 取的**不是**上述三个选项中的任何一个:
三者都默认"A1 改输出空间"这一前提,而 A2 直接**收窄该前提**(§10.1)。收窄后模型恢复产三类、
门与 0B-2 仍读二类,于是 0B-1 沿用原生三类头、五项阈值逐字不动 —— 上面 (a) 的静默重标定与
(c) 的重新论证都不再需要。(b) 的那条代价("认证的函数不再是建边的函数")在收窄下亦大幅缩小:
两者只差一个**零参数的冻结折叠**,且该等价性由 `cli/recompute_binary --against` 机器可验。
**本节记录的三条代价,A2 批准后全部消解(§10.4);批准前仍然生效。**

### 9.12 A1 的覆盖面从未被系统审计 —— 一次扫描又找出数处 [2026-08-05]

**0B-1 的失效(§9.11)不是孤例。** A1 起草时只逐条改了它明确点到的地方(输出空间、0B-2 twin 指标),
**没有对协议全文做过一次"哪里还假设三类"的排查**。0B-1 是在实现落地时撞出来的;
下面这批是 2026-08-05 一次 `grep 三类|REFUTES|UNKNOWN` 扫出来的,**同样是临时撞见而非系统审计**。

**(a) 实质冲突,须裁决 —— 目前只有一处,但它挡着 §3.8:**

- **§3.8 训练路径的配方仍写"脚手架用 sentence-transformers CrossEncoder **三类范式**"。**
  A1 已把输出空间改成二类,§9.7 列举 A1 作废项时**没有提到 §3.8**。
  ⇒ R013 一开跑就会撞上:**按预注册训三类头,还是按 g2-proto-2 训二类头?**
  训三类则与验收口径不一致;训二类则偏离预注册配方。**必须在训练开始前裁决,不得边训边定。**

**(b) 仅文本过时,实现已正确,但协议读起来会误导:**

- **§2.4 平局裁决**写"按 UNKNOWN → REFUTES → SUPPORTS 的顺序解" —— 二分类下该序列无指涉。
  实现以 `PREDICTED_LABELS` 表达且顺序正确(仍绝不落到 SUPPORTS),**文本待改**。
- **§3.6 G-AB** 把"abstention / **UNKNOWN rate**"定为一等报告项 —— 实现已改名 `not_supported_rate`。
  实质代价(弃权与反对不再可分)已记于 §9.1 实现裁决 1,**此处仅文本待改**。
- **§3.5 G-FC** 的防刷分论证写"把什么都判 UNKNOWN 的系统能刷爆 G-FC" —— 退化策略在二分类下依然存在
  (全判 NOT_SUPPORTED),论证成立,**仅指称待改**。

**(c) 已记录在案,不重复:** §3.2 的"UNKNOWN 的 primary gate 只能用 0B-1 的 official NEI 类"
与 §2.4 的 τ-gating 触发条件(0B-1 REFUTES precision < .85),两者均见 §9.11 的代价清单。

**由此得到的处置:**

1. **A2 的范围必须包含一次协议全文审计**,而不只是解决 0B-1。两处缺口同源:
   A1 只改了自己点到的地方。
2. **§3.8 在 (a) 被裁决之前不得启动。** Gate 0B 的失败已解锁它,但解锁不等于配方就绪。
3. **本节自身的局限如实声明:** 上表由一次关键词扫描得到,**不是穷尽审计**。
   A2 起草时须重做,并把"改一处输出空间会波及协议何处"这件事本身写成检查表 ——
   这已经是同一类疏漏的第二次和第三次。

**[2026-08-05 后续 —— 审计已重做,见 §10.7;检查表见 §10.8。本节保留原文不改写。]**
重做的结果与本节的自我评估一致,但方向出乎本节预料:
**(a) 的实质冲突由 A2 的收窄直接消解**(§3.8 配方逐字不动);
**(b) 判为"须改"的三条 —— §2.4、§3.6、§3.5 —— 收窄后全部自动正确,一条都不必改**;
**(c) 的两条同样自动正确**。而重做的审计另找出本节**未发现**的数处,其中两处是协议内部自相矛盾
(§3.3 的精确口径与 §9.1 冲突、§3.2 的 0B-2 表标签与 `task_probe.py` 冲突),
一处是与 A1 无关的独立事实错误(§3.1 臂表对 MiniCheck 阻塞原因的记述,已由 R012c 证伪)。
**即:本节的清单既有多余项也有遗漏项,"扫描不是审计"这句自述是准确的。**

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

---

## 10. 修订案 A2 — 把 A1 收窄至判读口径,并恢复 0B-1

**提出日期:** 2026-08-05 | **批准日期:** 2026-08-06 | **状态:** **已批准,`g2-proto-3` 生效**

批准不改变 §10.0 的披露:本修订仍是在看到 R012 与 R012c 的结果之后提出的,该事实随协议长期保留。

**本修订是对 A1 的部分撤回,不是澄清、也不是补丁。** A1 已批准生效、协议已升 `g2-proto-2`,
收窄它必须再升一版,并把"**A1 的结论超出了它自己的依据**"这句话写在最前面而不是脚注里。

### 10.0 诚实声明:本修订同样在看到结果之后提出,但方向与 A1 相反

按 §0 的冻结规则,本修订与 A1 同样需要先做披露。事实序列如下,不作美化:

| 时间 | 事件 |
|---|---|
| 2026-08-01 | R012 执行(job 18235972,n_external=55197)。**0B-1 五项在三类下跑完并录入:albert 5/5 全过,DeBERTa 挂 3 项** |
| 2026-08-03 | A1 提出并批准,升 `g2-proto-2`。A1 §9.1 改的是**关系模型的输出空间** |
| 2026-08-04 | A1 的代码改动落地(commit `221c34a`)。实测发现 0B-1 五项中三项退化、一项空洞,**该层挂起标 N/A**(§9.11) |
| 2026-08-05 | MiniCheck 臂(R012c,job 18269630-32)读出,**九格无一过门**,族级断言成立,§3.8 训练路径解锁 |
| 2026-08-05 | 关键词扫描发现 §3.8 的配方仍写三类范式,与 A1 冲突,**R013 因此不得启动**(§9.12) |
| 2026-08-05 | 本修订提出 |

**本修订的提出者同样无法主张"未受结果影响"。** 但有一条与 A1 不同、且可独立核验的不对称性,应当一并权衡:

- **A1 使门更容易通过;A2 恢复了一整层门。** 更具体:A2 恢复的 0B-1,**其唯一拒绝的臂正是 0B-2 上最接近通过的臂**
  —— DeBERTa 的 `gold_supports_recall` .7942 是九格最高值,而它在 0B-1 挂三项
  (macro_f1 .758 / non_unknown_coverage .675 / refutes_coverage .543)。恢复该层使 Gate 0B **更难**通过。
- **0B-1 的这批数字产生于 2026-08-01,早于 A1 存在。** A2 恢复的是已经量到的读数,**不产生任何新数字**。

**但必须同时写明本修订确实服务于本项目的利益:** 它解除 R013 的阻塞(§3.8 配方冲突消解)
与 R020 的阻塞(0B-1 复活),而这两项正是项目当前想推进的。**两面都成立,读者应当同时看见。**

### 10.1 依据:A1 的结论超出了它自己的依据

A1 的三条依据中,**两条只支持消费侧结论**:

| A1 依据 | 它实际证明的 | 它被用来支持的 |
|---|---|---|
| §9.2 门在主口径下从不消费 REFUTES | 门**读**二类即可 | 模型**产**二类 |
| §9.3 联合门已覆盖防刷分 | 0B-2 指标**读**二类即不可刷 | 模型**产**二类 |
| §9.4 G 模块已用二分类后端 | (唯一的生产侧论据,见下) | 模型**产**二类 |

**§9.4 是唯一的生产侧依据,而它不能被照搬。** `generator/nli.py` 的二分类后端是**带阈值**的
—— §9.4 自己引的那段就写着 "its probability sweep is usable"。而 §9.5a 明文禁止本路径引入阈值,
并把"argmax 无阈值"列为**使本设计区别于 credibility-threshold 仲裁的定义性属性**。
**A1 援引了一个自己不能照搬的先例:该先例成立的前提,恰恰是本路径禁止的那件事。**

因此,证据支持的命题是"**门与 0B-2 读二类**",而 A1 写成了"**模型产二类**"。
**0B-1 塌掉(§9.11)与 §3.8 配方冲突(§9.12(a))都是这一步越界的直接后果** ——
若 A1 当初只改判读口径,这两个问题一个都不会出现。

### 10.2 修订内容

| | A1 之后(`g2-proto-2`) | A2 之后(`g2-proto-3`) |
|---|---|---|
| 关系模型**输出空间**(§2.1) | SUPPORTS / NOT_SUPPORTED 二类 | **恢复三类** SUPPORTS / REFUTES / UNKNOWN |
| **门**的判读口径(§2.2 条件 1) | 只读 SUPPORTS | **不变** —— 只读 SUPPORTS |
| **0B-2 twin 指标**(§3.3) | `predicted != SUPPORTS` | **不变** |
| **0B-2 gold 指标**(§3.3) | `predicted == SUPPORTS` | **不变** |
| 0B-2 两项阈值 | twin ≥ .70,gold ≥ .85 | **不变** |
| 联合门 | 两项须同时过 | **不变,仍是防刷分的唯一机制** |
| **0B-1**(§3.2) | 挂起标 N/A(§9.11) | **恢复,五项阈值逐字不动** |
| 折叠发生的位置 | checkpoint adapter | **后移至两个消费点**(0B-2 指标、建边) |
| §3.8 训练配方 | 与输出空间冲突,R013 阻塞 | **冲突消解,配方逐字不动** |

**一句话形式:折叠仍然发生,只是发生得更晚 —— 从"模型只会说两类"改成"模型说三类,门只听两类"。**

`NOT_SUPPORTED` 由**预测标签**降为**派生标签**:它仍是 0B-2 与建边处的判读结果,不再是模型的输出类别。

### 10.3 A1 的哪些主张存活 —— 全部核心主张

1. **门在主口径下从不消费 REFUTES**(§9.2)—— 存活,本修订不改变它。
2. **联合门是防刷分的唯一机制**(§9.3)—— 存活。两种退化策略仍被 §9.3 那张表挡住。
3. **twin 指标在二值判读下已近饱和,退化为下限守卫**(§9.5)—— 存活,.8689–.9980 的读数不变。
4. **选型压力完全落在 `gold_supports_recall ≥ .85` 一项**(§9.5)—— 存活,九格最高仍 .7942。
5. **§9.5a 的阈值防护条款** —— 存活,且**其扳机恢复可求值**(见 §10.4)。
6. **§9.0 的披露、§9.6 的主结果保全、§9.8 的出样检验** —— 全部存活,不因本修订改写或撤回。

**R012 的三类 FAIL 仍是预注册主结果,A1 §9.6 第 1 条继续有效。**

### 10.4 §9.11 与 §9.7 的代价清单大部分反转

**§9.11 记录的三条挂起代价全部消解:**

| §9.11 的代价 | A2 之后 |
|---|---|
| 1. 验收证据只剩任务探针,§3.2 设两层的理由被架空 | **消解** —— 0B-1 复活,两层结构恢复 |
| 2. UNKNOWN 失去 primary gate | **消解** —— 模型重新产 UNKNOWN,§3.2 的 official NEI 类重新是它的主门 |
| 3. §9.5a 的预注册补救失去扳机 | **消解** —— `refutes_precision` 重新可求值(实测 albert .9026 / DeBERTa .9127,**触发条件仍未满足**) |

§9.1 实现裁决 1 的代价(弃权与反对不再可分)**同样消解**:§3.6 G-AB 拿回它的分解。
R012 已实测 `unknown_rate` albert **.482** / DeBERTa **.176** —— 该分解一直存在,只是被 A1 折掉了。

**§9.7 的代价清单四条反转三条:**

| §9.7 的代价 | A2 之后 |
|---|---|
| `conflict_mode=refutes_edge` 消融臂失去可执行性 | **恢复可执行** |
| TRAINING_PLAN Block 3 的"去掉 CLAIM_REFUTES"消融失去实质含义 | **恢复实质** |
| EXPERIMENT_TRACKER R036 整行变空 | **恢复** —— 该行原文即写"如 A1 日后被推翻则本行恢复" |
| 叙事代价:"带类型边的关系图"弱化为"支持计数图" | **不反转** —— 门仍然只读 SUPPORTS。A1 §9.7 说本修订"**暴露**了这一点而非造成它",这句判断是对的,A2 不改变它 |

### 10.5 是否使任何已 FAIL 的项变 PASS(§9.11 规定 A2 必答)

**Gate 0B 整体:不变,仍 FAIL。** 九格 `gold_supports_recall` 无一达到 .85(最高 .7942);
该指标在两种判读下逐字节相同,本修订不触及它。

**0B-1:从"挂起 N/A"回到 R012 的实测读数。** 分四点精确陈述,不合并:

1. §9.11 表中的 `0.0 / 0.5 / 1.0` 是**退化产物**,从未作为 Gate 判定发布 —— 该层当时标 N/A,
   且 `external_tier_status` 字段正是为阻止这些数字不带标记地流传而设。
2. 恢复后 0B-1 回到 **2026-08-01 已记录的读数**:albert 五项全过,DeBERTa 挂三项。
3. 故本修订**不产生任何新数字**,也**不改变 Gate 0B 的整体判定**;它使一个曾被挂起的层重新可判,
   而该层的读数早于 A1 存在。
4. **反向效果必须同时记录:** 该层拒绝的唯一的臂 DeBERTa,正是 0B-2 上最强的臂。

**结论:本修订不制造通过者,并使门整体更严。** 这一条与 A1 §9.0 第 2 点同样可证伪 ——
任何人重跑 R012 的 external pairs 即可检验。

### 10.6 代价与新增义务

1. **这是对已批准修订的部分撤回。** 协议一周内三个版本号(`g2-proto-1` → `-2` → `-3`),
   其中两次由同一处疏漏引发。这本身是过程质量的减分项,如实记录,不淡化。
2. **代码须改。** 改动面在 §10.7(D) 逐项列出,非零。
3. **天然二分类的臂不具备 0B-1 认证资格。** MiniCheck-FT5 无三类输出,过不了外部效度层
   (R012c 本就按 `EXTERNAL_PAIRS=none` 跑,无该层读数)。它已在 0B-2 上失败、不是候选,
   故本条**目前是假设性代价**;但将来若要以原生二分类模型作生产关系模型,该模型将**无法被 0B-1 认证**,
   届时须另开修订。**必须显式声明,不得默认。**
4. **§9.4 的"两模块收敛"论据作废。** A1 曾主张本修订使 R 与 G 两模块对同一子问题收敛;
   收窄后两者重新分歧(G 产二类带阈值,R 产三类无阈值)。**该分歧有实质理由**(§9.5a 的定义性属性),
   但 §9.4 那段论证不再可用,§9.10 最后一项待办的前提也随之改变。

### 10.7 A1 覆盖面的全文审计(§9.12 第 3 点要求重做)

§9.12 自述其清单"由一次关键词扫描得到,**不是穷尽审计**"。本节重做该审计。
**审计范围:** 本文件全文、`TRAINING_PLAN.md`、`selector.md`、
`../superpowers/specs/2026-07-30-graph-2.0-relation-layer-design.md`、`EXPERIMENT_TRACKER.md`、`src/`、`tests/`。
**检索式:** `三类|三分类|REFUTES|UNKNOWN|NEI|macro.f1|non_unknown|CrossEncoder`。

**审计的第一个结论是:收窄使多数条目自动消解,包括 §9.12 判为"须改"的三条中的全部三条。**

**(A) 收窄后自动正确,无须改动:**

| 位置 | §9.12 的原判 | A2 之后 |
|---|---|---|
| §1 D1 "NLI 关系边 + UNKNOWN" | **未发现** | 自动正确 |
| §2.1 标题"三类关系 + UNKNOWN"、`CLAIM_REFUTES` 行"关系模型预测"、不变量 1 | **未发现** | 自动正确 |
| §2.4 平局裁决 "UNKNOWN → REFUTES → SUPPORTS" | (b) 判须改 | **自动正确** —— `PREDICTED_LABELS` 恢复三类后字面等于原序 |
| §2.4 τ-gating 的触发条件 | (c) 记于代价清单 | **自动正确** —— 扳机恢复 |
| §3.2 0B-1 五项阈值 | §9.11 挂起 | **自动正确,逐字不动** |
| §3.2 "UNKNOWN 的 primary gate 只能用 0B-1 的 official NEI 类" | (c) 记于代价清单 | **自动正确** |
| §3.5 G-FC "全判 UNKNOWN 的系统能刷爆 G-FC" | (b) 判须改 | **自动正确** |
| §3.6 G-AB "abstention / UNKNOWN rate 是一等报告项" | (b) 判须改 | **文本自动正确**;但须补实现,见 (D) |
| §3.8 训练配方"CrossEncoder 三类范式" | (a) **实质冲突,挡着训练路径** | **冲突消解,配方逐字不动,R013 解锁** |
| §5.4 "若 §3.8 训练路径被启动,三 seed 条款恢复生效" | **未发现** | 随 §3.8 启动自动生效,文本不动 |

**(B) 仍须改动 —— 本节是 §10.10 第三项批准条件的清单:**

| 位置 | 改成什么 | 原因 |
|---|---|---|
| §3.1 臂表 MiniCheck 行"二分类,需否定 claim 双向探测,单独一步" | 改为记录**真实阻塞是架构**:`lytang/MiniCheck-Flan-T5-Large` 是 `T5ForConditionalGeneration` 且无 `id2label`,`AutoModelForSequenceClassification` 打不开,故走独立的二分类注册表 | **与 A1 无关的独立错误**,由 R012c 的实现证伪。该句曾是 MiniCheck 被降级的理由 |
| §3.2 0B-2 表第 3、4 行的标签列 `REFUTES` | 改为 `NOT_SUPPORTED` | 探针 gold 标签**跟随指标**(2026-08-05 裁决)。`task_probe.py` 已产 `TWIN_NOT_SUPPORTED`,协议文本对齐实现 |
| §3.3 指标名 `twin REFUTES accuracy`(表内与精确口径两处) | 改为 `twin NOT_SUPPORTED accuracy` | A1 §9.1 已改该指标而 §3.3 正文未同步 —— **协议内部自相矛盾** |
| §3.3 精确口径三条("预测 == REFUTES 的比例"、"两者的 UNKNOWN 一律计失败") | 改写为:twin = `predicted != SUPPORTS`;gold = `predicted == SUPPORTS`;**三类输出在两项上一律先按 §10.2 折叠再计分** | 同上。照 §3.3 字面计算得到的量,与实现产出的量不是同一个 |
| §6 Gate 0A 清单末项的 `g2-proto-1` | 改为 `g2-proto-3` | 版本号过时,且该项是退出判据 |
| §8 第 4 项"在代码改动落地前 `relations/` 仍按三类实现运行" | 改为记录 `221c34a` 已落地、并由本修订部分回退 | 过时 |
| §8 条目编号 1,2,3,5,4 | 重排 | 乱序 |
| §9.7 的代价清单 | 加一行指向 §10.4 的反转结果,**原文不改写** | A1 的记录须完整保留(§9.6 第 1 条的精神) |

**已随本修订起草一并落地的状态注记(非协议改动,故不入批准清单):**
§8 第 5 项已由"A2 待起草"改为"A2 已起草待批准";§9.11 与 §9.12 各已加一段
`[2026-08-05 后续]` 指向 §10,**两节原文均逐字保留**。
三处都明写"**批准前 §9.11 的代价仍然生效**",不得被读成 A2 已生效。

**(C) 与 A1 无关的独立过时(本修订不处理,单独记为待办):**

- `selector.md` 仍写关系模型监督来自 **ContractNLI**(§3.0 已移出)、底座为 `cross-encoder/nli-deberta-v3-base`
  (§3.1 已换成三臂)。**两处均非 A1 造成**,而是 §3.0 与 §3.1 的实质修改未回写。

  **[2026-08-06 更正 —— 本条后半已不成立。]** A3(§11)按预先定死的规则重新选型后,
  **§3.8 的基座就是 `cross-encoder/nli-deberta-v3-base`**,即 `selector.md` 那句**重新变成正确的**,
  无须改(§11.10 第 1 项)。**ContractNLI 那半仍然过时。**
  本条留作记录:一次修订判为"须改"的文本,可能被下一次修订改回正确 ——
  §10.8 检查表第 6 项之所以要求"每次重扫平行文档",而不是沿用上一轮的结论,原因就在这里。
- `TRAINING_PLAN.md` §3.2、Gate 0B 阈值段、Block 3 —— 按 §0"凡本文件冻结的条目以本文件为准",
  不构成冲突,但读者会被误导。
- 设计文档 §3.1 有与本文件 §3.1 臂表同源的 MiniCheck 错误陈述。

**(D) 实现清单(代码,须在本修订批准后落地):**

- [ ] `relations/models.py`:`PREDICTED_LABELS` 恢复三类并按 §2.4 平局序排列;
      `_SCHEMA_ONLY` 的成员对调 —— `NOT_SUPPORTED` 成为派生标签
- [ ] `relations/predictor.py` 中拒绝三类分数的守卫**反向**:三类成为合法输入
- [ ] `cli/gate0b.py` 的 `_collapse_to_binary` 调用点从 checkpoint adapter 后移到 0B-2 指标与建边两处;
      **0B-1 路径不折叠**
- [ ] 撤除 `cli/gate0b.py` 的 `external_tier_status` 字段与 `tests/cli/test_gate0b.py` 的对应断言
      —— §9.11 明文规定"A2 落地后移除"
- [ ] 替换 `test_0b1_is_UNRUNNABLE_after_A1_and_this_test_records_it_rather_than_fixing_it`
- [ ] G-AB 报告项补回 UNKNOWN 分解:`not_supported_rate` 在 0B-2 层仍正确,但须并报 `unknown_rate`
- [ ] 复核 `relations/task_probe.py`、`relations/graph.py`、`relations/minicheck.py` 的契约
- [ ] 折叠等价性由 `cli/recompute_binary --against` 重新对拍,**证明后移折叠点不改变任何已发布读数**
      —— **本项未完成,原因如实记录:** `results/` 在 `.gitignore` 内,逐 pair dump(`dump-*.jsonl`)
      只存在于集群,本地无法执行。**须在 bp1 上补跑,不得因其余项已过而勾掉。**
      **本地已做的替代验证(不等价,但非空):** 用 R012 的 `sweep-full.json`(job 18235972,
      pre-A1 的**原生三类**实现所产)钉住两件事 —— (1) 其五项读数在**现行 `THRESHOLDS`** 下
      仍落在同一侧(albert 五项全过、DeBERTa 挂三项),任何静默重标定都会在此暴露;
      (2) `external_report` 的字段集与 R012 发布的十个键**逐字相同**,即 0B-1 是被恢复而非被重建。
      两项见 `tests/relations/test_gate0b.py`。

**无须重跑任何实验。** §9.5 公布的二分类读数本就由"三类 argmax 换标签"得到,与折叠点后移后的读法同义;
0B-1 的读数则产生于 R012 的三类原生头。**本修订不产生、也不作废任何实验数据。**

### 10.8 可复用检查表:改一处输出空间会波及协议何处

同一类疏漏已发生三次(0B-1、§3.8、以及本次审计新发现的若干处)。故按 §9.12 第 3 点的要求把它写成检查表,
而不是再靠关键词扫描撞见。**凡修改关系模型的输出空间、标签集合或判读口径,必须逐项回答:**

1. **谁产、谁读、在哪折叠?** 写出输出空间、每个消费点的判读口径、折叠发生的位置。
   **只改判读口径即可达成目的时,不得改输出空间** —— 这正是 A1 越界的那一步。
2. **每一层验收门是否仍可求值?** 逐个 gate、逐个指标检查类别与分母是否仍存在。
   **一个恒为 0.0 或恒为 1.0 的指标是失效,不是通过。**
3. **每个阈值是在哪个量上标定的?** 若量变了而阈值不变,即**静默重标定**,须显式论证或重新标定。
4. **有没有条款以某指标的取值为触发条件?** 该指标失效则条款悬空 —— §9.5a 的扳机就是这样丢的。
5. **有没有消融臂 / tracker 行 / 训练配方以旧标签集为前提?** 逐个列出,标 N/A 或改写。
6. **上游与平行文档:** TRAINING_PLAN、selector.md、设计文档、EXPERIMENT_TRACKER 各扫一遍。
7. **代码与测试:** 枚举、常量、守卫、characterisation 测试,以及缓存与 dump 的向后解析。
8. **最后一步:把本修订的"作废条目清单"与上面 1–7 的答案逐条对照。**
   A1 的 §9.7 清单漏掉 §3.8,正是因为没做这一步。

**纪律条款:** 上列八项必须在修订案正文中留下书面答案,并由批准人核对。
理由是本项目已经反复撞到的那条教训 —— **只写在文档里、没有强制机制的义务,迟早被跳过**。

### 10.9 利益冲突声明

本修订解除 R013(训练路径)与 R020(0B-1)两处阻塞,**而这两处正是项目当前想推进的**,故存在直接利益。
与 A1 的差别必须精确陈述,且不得用来免责:

- **A1 使门更容易通过;A2 使门更严** —— 恢复一整层,且该层拒绝的正是 0B-2 上最强的臂。
- **A2 不改变 Gate 0B 的整体判定**(仍 FAIL),也不产生任何新数字。
- **但 A2 确实让项目得以继续推进。** 一个"更严但解除阻塞"的修订,其利益冲突形态与 A1 不同,**不等于没有**。

**缓解措施:** §10.0 的时间线披露、§10.5 的可证伪陈述、§10.7 的完整审计与实现清单。
**审阅者仍应默认本修订带有确认偏误,并据此加重审查。**

### 10.10 批准所需

- [x] 项目负责人批准,并将本文件版本号改为 `g2-proto-3` —— **2026-08-06 完成**
- [x] `EXPERIMENT_TRACKER.md` 的 Protocol 行与版本沿革同步;R036 由 N/A 恢复;
      R013–R015 的"配方待裁决"注记改为"配方已确认,按 §3.8 预注册原文";R020 解除 BLOCKED
      —— **2026-08-06 完成**。R020 另带一项**新的待裁决**:R012 已在三类原生头上跑过一次
      official test,该次是否消耗"只运行一次"的额度,须在 §3.8 训练出的模型上场前明确。
- [x] §10.7(B) 的协议文本改动逐项落地 —— **2026-08-06 完成**(八处)
- [x] §10.7(D) 的代码改动落地(TDD)—— **2026-08-06 完成**。逐字 CI 全树通过:
      `ruff check src tests` / `mypy src tests/typecheck.py` / `pytest tests/`。
      **落地时新增一项 §10.6 未预见的强制:** 撤掉 `external_tier_status` 后,
      天然二分类的臂若带 `--external-pairs` 跑会**无声产出**那四个退化数字 —— 警告字段没了,
      而 §10.6 第 3 条只写在文档里。故 `external_report` 现在**硬失败**:预测中出现
      `NOT_SUPPORTED` 即拒绝认证。这是"写进文档的义务必须有代码执行"的直接应用。
- [ ] **`cli/recompute_binary --against` 的对拍尚未执行** —— 见 §10.7(D) 末项,须在 bp1 补跑
- [x] §10.8 检查表的八项在本修订正文中均有书面答案 —— **随 2026-08-06 的批准一并核对**
- [ ] G 模块就后端与阈值复用达成一致(承自 §9.10 的未完成项;**A2 使该项的前提改变**,见 §10.6 第 4 条)

### 10.11 §3.8 解锁后的已知风险(不构成阻塞,但须预先写明)

**风险:三类头为 UNKNOWN 保留概率质量,而 `gold_supports_recall` 恰恰惩罚不承诺。**
§9.5a 的阈值扫描已量到该现象:albert / rung 1 的 `gold_supports_recall` 由 argmax 的 **.1916**
升至 θ=.05 的 **.7018**(3.66×),即失败是"**不肯承诺**"而非"判不出"。
零训练路线下 UNKNOWN 是模型自带的槽位,**训练一个三类头不会自动消除它**。

**预先裁定:** 若 R013–R015 训完后 `gold_supports_recall` 仍未过 .85、且失败可归因于 UNKNOWN 的质量堆积,
该事实按预注册如实报告,并**须另开修订案(编号顺延)才能改训二类头 —— 不得边训边改配方**。
*(本条原写"A3";A3 已由 2026-08-06 的训练基座修订占用,见 §11。)*

写在这里的目的只有一个:该情形一旦发生,它是**被预先描述过的结果**,而不是一个可以临场重新解释的意外。

---

## 11. 修订案 A3 — §3.8 训练基座的重新选型

**提出日期:** 2026-08-06(**v2,取代同日的 v1 草稿**) | **批准日期:** 2026-08-06 | **状态:** **已批准,`g2-proto-4` 生效**

批准不改变 §11.0 的披露:三次选型疏漏(含本修订 v1 的错误主张)随协议长期保留。

### 11.0 诚实声明:本修订存在,是因为基座选型从来没有做过

不是"做得不够好",是**没做过**。三次疏漏叠加,逐条记明,包括本修订自己的第一稿:

| # | 疏漏 | 记录 |
|---|---|---|
| 1 | `selector.md` 把基座定为 `cross-encoder/nli-deberta-v3-base`;**§3.8 换成了裸的 `microsoft/deberta-v3-base`,未记录任何理由** | 与 review-prep 已自认的"两个 checkpoint 随设计文档静默进入"是同一物种,只是这次是**基座** |
| 2 | `gate0b-review-prep.md` 论证 Granite reranker 是"最佳候选",**但四条论据比的是 albert(59M / 512),不是 §3.8 的实际基座** | 能证成换基座的那个比较**从未做过** |
| 3 | **本修订 v1 草稿(2026-08-06)声称 Granite 是"唯一能加载的那个"** | **错误。** 候选池里至少五个能加载。该稿的核心句被自己的候选调查推翻,故整节重写 |

**三次的共同形状:先有结论,再找理由,而理由没有对着正确的对象比。**
本节因此不先给结论 —— **先定规则(§11.2),再套用(§11.5)**。

### 11.1 触发事实:预注册基座在本项目环境中不可加载

本项目 pin `torch < 2.6`,该 pin 下 transformers **拒绝 `torch.load` 任何 `.bin`**(CVE-2025-32434);
`cli/gate0b.py` 的 `use_safetensors=True` 注释已记录该坑并写明**不得以放宽它来"修"失败**。

**实测(2026-08-06,HF API):`microsoft/deberta-v3-base` 只发布 `pytorch_model.bin`,无 safetensors。**

⇒ **§3.8 的预注册配方按原文不可执行。这是本修订成立的唯一必要条件,也是它主张的全部。**
"因此应当换成某个特定模型"**不从中推出** —— 那需要 §11.2 的选型。

**两条限定必须同时写明:**

1. 该结论依赖集群的 torch pin。**若集群实际 torch ≥ 2.6,`.bin` 可加载,本修订的必要性当场消失**,
   §3.8 应按原文执行。故 §11.9 把"实测集群 torch 版本"列为**第一项**硬性前置。
2. 手动把 `.bin` 转成 safetensors 是可行绕路,但使基座权重成为**本地转换产物而非官方发布物**,
   与本项目"权重指纹进 `model_version`"的做法冲突。**该路径未被采纳,但也未被否决** —— 若选型结果
   不理想,它是一个应当被重新考虑的选项,须显式论证。

   **[2026-08-06 补 —— 本条已由一次实际发生的事从假设性变成 live,必须记明。]**
   G 模块的生产验证器 `google/t5_xxl_true_nli_mixture` **同样只发 `.bin`**,在同一个
   `transformers 4.57.6 + torch 2.5.1` 组合下同样加载不了 —— G6 生成作业的三个验证臂
   365/400 全挂,根因即此。处置是**采纳了这条绕路**:以 `weights_only=True`
   (CVE-2025-32434 自身指定的缓解手段)转成 safetensors,每个输入输出文件的 sha256 写进
   `results/true_nli_safetensors_conversion.json`(`scripts/convert_bin_to_safetensors.py`)。

   **于是本条的问题变成:凭什么验证器可以转,训练基座不可以?**
   **这个不对称存在,但必须被论证,不得默认** —— **训练基座的 provenance 流进本项目要交付的模型;
   验证器是测量仪器,其权重在两种方案下都由 hash 钉死。** 前者的转换风险高一个量级。

   **该论证的强度须如实标注:它是一条合理的区分,不是一条不可反驳的区分。**
   **若有人认为它不成立,则 A3 的核心依据(§11.1"预注册基座不可加载")相应减弱** ——
   因为"不可加载"就不再等于"不可使用"。本条写在这里,正是为了让这个反驳能在内部被提出,
   而不是等它在评审时由别人提出。

### 11.2 选型规则(在看任何候选之前定死)

**Tier 0 — 硬约束,不满足即出局,不做权衡:**

| | 约束 | 理由 |
|---|---|---|
| H1 | **官方发布 safetensors** | §11.1;不接受本地转换产物 |
| H2 | **架构可被 pin 住的 transformers(`>=4.45,<5`)解析** | MiniCheck 的 `T5ForConditionalGeneration` 就是这类失败,且是实现时才撞见的 |
| H3 | **宽松许可证**(Apache-2.0 / MIT) | 可发表性 |
| H4 | **cross-encoder / 句对分类架构** | 双塔 embedding 模型不适用;§3.8 的脚手架是 CrossEncoder |
| H5 | **训练语料不含 VitaminC** | **0B-1 在 VitaminC official test 上验收。基座若已训过 VitaminC,该层当场失去意义** |
| H6 | **`id2label` 语义具名且可核验** | 一个 `{LABEL_0, LABEL_1, LABEL_2}` 的三类头,其顺序**无法从 config 判定**。`LABEL_ORDER` 的注释已写死这条:猜一个顺序**不会报错**,只会静默产出可信的错数字 |

**H5 的核验方式(2026-08-06 补,由一次实际踩中改写):** **不得以 HF `cardData.datasets` 为准。**
`tasksource/ModernBERT-base-nli` 的 `cardData` 只列 `glue` 与 `anli`,而 tasksource 的实际训练任务清单
(`sileod/tasksource` 仓库 `tasks.md`)含 **`bigbench/vitaminc_fact_verification`**。
**cardData 已被实证为不完整**,故 H5 与 C3 一律以**上游训练任务清单**为准,查不到清单者按**不通过**处理。

**Tier 1 — 预注册的排序判据,按顺序套用,前一条能分出胜负就不看下一条:**

- **C1 与 §3.8 预注册基座的血缘距离,近者优先。**
  依据:**本修订的授权范围是"修一个不可执行的配方",不是"重开一次选择"**。
  §11.1 只证成了"预注册基座打不开",没有证成"预注册的判断是错的"。
  故 `base_model` 就是 `microsoft/deberta-v3-base` 的候选,偏离最小。
- **C2 任务先验的方向:** 已带正确语义的三类 NLI 头 **>** 中性裸 encoder **>** 与任务目标相反的先验。
  依据:§3.8 的目标是三类 entailment 判定;一个被训练成"相关即高分"的模型,
  其目标函数**恰好奖励**本项目要抓的失败模式(反事实孪生与 gold claim 高度相关)。
- **C3 与 VitaminC 的污染距离,远者优先。** 具体地:训练语料**不含 FEVER** 者优先 ——
  VitaminC 有 FEVER 派生的部分,而 0B-1 在 VitaminC official test 上验收。
  *(该重叠本修订**未实测**,仅按公开语料清单作保守排序;见 §11.7 第 3 条。)*
- **C4 仅在 C1–C3 全部打平时启用:** §11.2a 的一次测量。

**Tier 2(§11.2a)— 只在需要 C4 时执行的一次测量,规则同样先定死:**

- **测量面 = VitaminC official *dev*。** 依据:§2.4 已经把该集指定为唯一合法的标定面
  ("τ 只在 VitaminC official **dev** 上……冻结一次"),它与 0B-1 的 official test、
  与 0B-2 的 NIAH 探针**三者互不相交**。
- 指标与判据在运行前写死;**只跑一次**;结果**只能作为基座选型证据报告,任何场合不得作为 Gate 0B 读数呈现**。

### 11.3 两条被明确排除的"理由",以及排除它们的代价

**(a) "回到 Granite 家族"不是选型判据。**
它是**利益相关方与叙事上的好处**,不是能力或效度论据。本修订**承认这个好处是真实的**
—— 一个 IBM Granite 项目的关系层用 Granite 基座,对报告确有帮助。
但它不能被写成技术理由,这正是 §11.0 第 2 条那次疏漏的形状。
**若项目负责人认为它应当成为判据,那是一个正当的决定,但必须显式作出并披露**,不得混入技术论据。

**【裁决 2026-08-06】不作为判据。** 项目负责人明确表态:家族一致性**考虑过,排除了**。
理由记明:本项目按 rigor 评价,"按预注册判据选型、并记录家族偏好被排除"比"选了赞助方的模型"
更经得起审阅;且 Granite 在本项目的叙事不缺这一块 —— G 模块的生产链路本就用 Granite。
**本条不因裁决而删除:它记录的是这个好处真实存在、且被有意识地放弃了。**

**(b) 不为选型增开零训练臂,且本决定对本项目有利,故必须连同利益一起声明。**

§3.1 预注册了**三臂**,三臂已跑完,族级断言("任何零训练模型都不够")据此成立,§3.8 因此解锁。
**在看到该比较的结果之后再往里加臂,正是 §9.0 存在所要防的那件事** —— 一直测到出现不同答案为止。

**但必须说明这个决定的受益方是谁:** 若新增的第四臂零训练就过了 0B-2,
**族级断言当场被证伪,§3.8 根本不该解锁,R013–R015 应当撤回**。项目在"不去看"上有直接利益。

**故预先写死两条:**

1. §11.2a 的测量若被启用,**只在 VitaminC official dev 上做**,不碰 0B-2 探针,
   因此在结构上不可能顺带产出一个新的 Gate 0B 臂。
2. **若日后出于任何其他原因取得了某个候选在 0B-2 上的零训练读数,且它两项同时过阈,
   该事实必须立即如实报告并触发对族级断言的复核** —— 不得因"它不是预注册臂"而搁置。

### 11.4 候选池与实测事实(2026-08-06)

**候选池的来源必须说明,否则"选型"仍是攒出来的:** 本池由 HF API 三次检索合并去重得到
(`pipeline_tag=zero-shot-classification` 按下载量、`search=nli` + `filter=text-classification`、
`search=deberta-v3-base-nli`),**83 个去重候选**,再按 Tier 0 逐条过滤。
v1 草稿的候选池是从项目文档里已提到的模型攒的,**不是一次检索**;这是本节相对 v1 的实质改动。

**所有字段为元数据,非任务测量。本节没有任何一项是本修订作者跑出来的。**

| 候选 | H1 权重 | H2 架构 | H3 许可 | H6 标签 | 层×hidden / ctx | 已有头 | 训练语料 | `base_model` |
|---|---|---|---|---|---|---|---|---|
| `microsoft/deberta-v3-base`(预注册) | **✗ 仅 .bin** | deberta-v2 | MIT | — | 12×768 / 512 | 无 | — | — |
| **`cross-encoder/nli-deberta-v3-base`** | ✓ | deberta-v2 | **Apache-2.0** | ✓ 具名 | 12×768 / 512 | **三类 contradiction/entailment/neutral** | **MNLI + SNLI** | **`microsoft/deberta-v3-base`** |
| `cross-encoder/nli-deberta-v3-large` | ✓ | deberta-v2 | Apache-2.0 | ✓ 具名 | **24×1024** / 512 | 三类,同语义 | **MNLI + SNLI** | `microsoft/deberta-v3-large` |
| `cross-encoder/nli-deberta-v3-small` | ✓ | deberta-v2 | Apache-2.0 | ✓ 具名 | 6×768 / 512 | 三类,同语义 | MNLI + SNLI | `microsoft/deberta-v3-small` |
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | ✓ | deberta-v2 | MIT | ✓ | 12×768 / 512 | 三类 | MNLI + ANLI + **FEVER** | — |
| `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | ✓ | deberta-v2 | MIT | ✓ | 24×1024 / 512 | 三类 | MNLI + ANLI + **FEVER** + LingNLI + WANLI | — |
| `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` | ✓ | deberta-v2 | MIT | ✓ | 12×768 / 512 | **二类** entailment/not_entailment | 未公开完整清单 | `microsoft/deberta-v3-base` |
| `sileod/deberta-v3-base-tasksource-nli` | ✓ | deberta-v2 | Apache-2.0 | ✓ | 12×768 / 512 | 三类 | **600 任务,含 VitaminC** | — |
| `tasksource/ModernBERT-base-nli` | ✓ | **modernbert** | Apache-2.0 | ✓ | 22×768 / 2048 | 三类 | **同上,含 VitaminC** | `answerdotai/ModernBERT-base` |
| `tasksource/deberta-small-long-nli` | ✓ | deberta-v2 | Apache-2.0 | ✓ | 6×768 / 1680 | 三类 | **同上,含 VitaminC** | `microsoft/deberta-v3-small` |
| `dleemiller/EttinX-nli-xs` | ✓ | **modernbert** | MIT | **✗ `LABEL_0/1/2`** | 10×384 / 7999 | 三类,**语义不可核验** | all-nli-distill | `jhu-clsp/ettin-encoder-32m` |
| `answerdotai/ModernBERT-base` | ✓ | **modernbert** | Apache-2.0 | — | 22×768 / 8192 | 无 | MLM 预训练 | — |
| `ibm-granite/granite-embedding-reranker-english-r2` | ✓ | **modernbert** | Apache-2.0 | **✗ `{0: LABEL_0}`** | 22×768 / 8192 | **单输出排序头** | 相关性排序(pListMLE) | `ibm-granite/granite-embedding-english-r2` |
| `tals/albert-xlarge-vitaminc-mnli` | ✓ | albert | **未声明** | ✓ | 24×2048 / 512 | 三类 | **VitaminC** + MNLI | — |

### 11.5 规则套用

**Tier 0 淘汰:**

- `microsoft/deberta-v3-base` —— **H1 不过**(无 safetensors)。这就是本修订存在的原因。
- **`sileod/deberta-v3-base-tasksource-nli`、`tasksource/ModernBERT-base-nli`、
  `tasksource/deberta-small-long-nli` —— H5 不过。** tasksource 的训练任务清单含
  **`bigbench/vitaminc_fact_verification`**。**这一族本来是最有吸引力的替代品**
  (deberta-v3-base 同规模、Apache-2.0、语义正确的三类头、600 任务的多任务 NLI 先验),
  **而它的 cardData 只列 `glue` 与 `anli`,不列 VitaminC** —— 不查上游清单就会选中它,
  并**静默污染 0B-1**。H5 的核验方式因此被改写(见 §11.2)。
  *(限定:BIG-bench 版可能只是 VitaminC 的子集。但 H5 是硬约束不是加权项,
  举证责任在想用它的人 —— 须证明该子集与 VitaminC official test 不相交。)*
- `tals/albert-xlarge-vitaminc-mnli` —— **H5 不过**(已训 VitaminC);**H3 亦不过**(许可证未声明)。
- `dleemiller/EttinX-nli-xs` —— **H6 不过**(`LABEL_0/1/2`,顺序不可从 config 核验);H2 亦未决。
- `granite-embedding-reranker-english-r2` —— **H6 不过**(单输出 `{0: LABEL_0}`);H2 未决。
- `answerdotai/ModernBERT-base` —— **H2 未决**:ModernBERT 需 transformers **≥ 4.48**,
  而 pyproject 只 pin `>=4.45,<5`。**允许不等于满足**,须由 §11.9 实测;通过前不得被选中。
- `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` —— **H5 未决**:未公开完整训练清单,
  按 §11.2 的新规则**计不通过**。另:其头是**二类**,与 A2 恢复的三类输出空间不符。

**Tier 1 逐条(存活候选):**

| | **`cross-encoder/nli-deberta-v3-base`** | `cross-encoder/nli-deberta-v3-large` | `cross-encoder/nli-deberta-v3-small` | MoritzLaurer base | MoritzLaurer large |
|---|---|---|---|---|---|
| **C1** 血缘距离 | **`base_model` 即预注册基座 —— 距离最小** | 同族但换规模(-large) | 同族但换规模(-small) | 不同血缘 | 不同血缘且换规模 |
| **C2** 先验方向 | **三类 NLI 头,语义正确** | 同左 | 同左 | 三类 NLI 头 | 三类 NLI 头 |
| **C3** 污染距离 | **MNLI+SNLI,无 FEVER** | 同左 | 同左 | 含 **FEVER** | 含 **FEVER** |

**C1 即分出胜负,且 C2、C3 无一反向。故 §11.2a 的测量不启用。**

**结论:`cross-encoder/nli-deberta-v3-base`。**

**必须同时记下的一个未被采纳的考量:** `cross-encoder/nli-deberta-v3-large` 与所选候选
**同配方、同语义、同许可、同污染面,只差规模**(24×1024 vs 12×768),仅在 C1 上让位。
而 R012 的实测里,唯一接近过阈的臂(`gold_supports_recall` .7942)恰是一个 **large 规模的三类 NLI 模型**,
远高于两个 base/xlarge 规模的臂。**这构成"规模可能才是关键变量"的提示。**

**本修订不采纳该提示,理由必须写清楚:** 用它就是**拿 Gate 0B 的读数来选基座** ——
与 §11.3(b) 拒绝增开零训练臂是同一条纪律,只是方向相反。
**但如实声明:该拒绝对本项目未必有利**(它可能让我们训了一个规模不足的基座)。
若项目负责人认为规模应当成为判据,那是正当决定,**但须显式作出、写明它使用了 Gate 0B 的结果,
并按 A1/A2 同规格披露** —— 不得因为"large 显然更好"就默默换掉。

**【裁决 2026-08-06】规模不作为选型判据,改为预注册的应急臂,条文落在 §3.8。**
即:R013–R015 按 base 规模跑;**若三 seed 训完 `gold_supports_recall` 仍未过 `.85`,
允许且仅允许升到同族同配方的 `cross-encoder/nli-deberta-v3-large` 重跑三 seed,两者并列报告。**

**这样处理的理由,以及它没有解决的部分:**

- **好处:** 现在换 large 等于拿 Gate 0B 的零训练读数选基座,要背一整轮披露;
  而应急臂的触发条件是**本项目自己的训练结果**,不是别人的零训练读数 —— 依据来源干净得多。
- **没解决的部分,如实写:** 应急臂被触发时,那**仍然是"看到结果之后换了模型"**。
  区别只在于**换法与触发条件是预先写死的**,且旧臂结果必须并列保留。
  这不是把问题消掉,是把它从"事后解释"降级为"预先描述过的路径"—— 与 §10.11 同一手法,同一局限。

**这个结果值得单独说一句:它正是 `selector.md` 的原始选择** —— §11.0 第 1 条那次未记录理由的替换,
把它换掉了。**规则独立套用之后又选回了它**,这既是对规则的一点旁证,也说明那次静默替换是有代价的。

**同时如实说明本次选型的性质:** 上述判据全部是**文档性**的 —— 血缘、头、语料清单。
**没有任何一条是能力测量。** 本修订不主张所选基座在本任务上最强,只主张
**在预先定死的规则下,它是唯一一个三条判据全部占优、且不依赖任何未决前置的候选**。

### 11.6 修订内容

| | g2-proto-3 现行 | A3 之后(g2-proto-4) |
|---|---|---|
| §3.8 训练基座 | `microsoft/deberta-v3-base` | **`cross-encoder/nli-deberta-v3-base`** |
| 头 | CrossEncoder 三类范式 | **不变** —— 且基座**已带**语义正确的三类头,无须重新初始化 |
| §3.8 其余全部条款 | VitaminC 主训 + revision-family 去污染 / NIAH 域适配 / parent page + synthetic family 的 5-fold OOF / 三 seed 13,42,73 / official test 只跑一次 / sealed-600 零重叠 | **逐字不变** |

### 11.7 代价与风险

1. **基座已在 MNLI + SNLI 上训过,§3.8"零训练路线下模型从未见过 NIAH 语料 ⇒ passage-hash 泄漏轴天然为空"
   那句话不再成立。** 该副作用条款须随之改写:**泄漏轴不再天然为空,必须实测**。
   *(此条对任何非裸基座都成立,不是本候选独有。)*
2. **C3 的污染排序未经实测,且其证据来源已被实证为不可靠。** VitaminC 与 FEVER 的实际重叠
   本修订**没有查**,仅按上游训练清单作保守排序。**该排序不构成"无污染"的证明。**
   更要紧的是:tasksource 那一族的 `cardData` **漏列了 VitaminC**(§11.5),
   即**模型卡的语料声明可以是不完整的**。所选候选的清单(MNLI + SNLI)来自
   sentence-transformers 的单一用途训练,比 600 任务的合集更可能完整 ——
   **但"更可能"不是"已验证",§11.9 未就此设检查项,这是本修订已知的薄弱处。**
3. **本次选型没有任何能力测量。** 若日后有证据表明另一候选在本任务上明显更强,
   本节的规则允许重开选型 —— 但须另开修订,且**不得在 R013 已产生结果之后重开**。
4. **Granite 家族的叙事好处被放弃了**(§11.3(a))。这是真实代价,记在此处而不是省略。
5. **这是第三份事后修订,协议一周内四个版本号**,其中本份还经历了一次自我推翻(v1 → v2)。
   过程质量减分项,如实记录。

### 11.8 是否使任何已 FAIL 的项变 PASS(A1/A2 同规格必答项)

**不。** 本修订不触及任何指标、任何阈值、任何已产生的读数;**Gate 0B 仍 FAIL**。
它改变的只是一个**尚未开始**的训练运行的基座 —— R013–R015 至今零结果,
故不存在"由 FAIL 变 PASS"的对象。这一条在三份修订里最易核验。

### 11.9 硬性前置检查(任一不过即停)

**执行方式:`python scripts/a3_preflight.py`。** 前六项由该脚本自动执行,并以退出码把关 ——
按 §10.8 的纪律,写成"必须先做 X"的义务要么有工具产出 X,要么迟早被跳过。

**(A) R013 开跑前必须过 —— 六项已于 2026-08-06 在 bp1 实测通过:**

- [x] **实测集群 torch 版本** —— 若 ≥ 2.6,§11.1 的必要性消失,本修订应被撤回。
      **实测 `torch 2.5.1+cu121` < 2.6,A3 的前提在集群上成立。**
- [x] **在集群上复核 `microsoft/deberta-v3-base` 确无 safetensors**,不以本地 API 查询为准。
      **实测权重清单 = `['pytorch_model.bin']`。**
- [x] `AutoModelForSequenceClassification.from_pretrained(..., use_safetensors=True)` 实测可加载。
      **实测加载成功,184,424,451 参数**(与 deberta-v3-base 的血缘一致)。
- [x] **实测 `id2label`,并按 `LABEL_ORDER` 的规矩登记。**
      **实测 `{0: contradiction, 1: entailment, 2: neutral}` ⇒ `("REFUTES", "SUPPORTS", "UNKNOWN")`,已登记。**
      **该条目是表中第一个 position 0 不是 SUPPORTS 的** —— 现有两条均以 SUPPORTS 打头,
      按位抄邻居会在每条边上把 SUPPORTS 与 REFUTES 对调,且不报错。测试已直接钉住该形状。
- [x] 记录权重指纹并写入 `model_version`。
      **基座指纹 = `cross-encoder/nli-deberta-v3-base@c95d83f857fd4fcd`**;训练产出的模型将有自己的指纹。
- [x] sentence-transformers CrossEncoder 脚手架实测能吃下该 checkpoint。
      **`CrossEncoder(num_labels=3)` 构造成功。**

**(B) sealed-600 评测前必须过,不阻塞 R013 [排序更正,2026-08-06]:**

- [ ] **§11.7 第 1 条:passage-hash 泄漏轴改为实测,不得再按"天然为空"处理**

**本项由 (A) 移入 (B)。理由与"移动"这件事本身一并记录,不得直接勾掉:**

1. **它现在跑不了,且原因与 R013 无关。** 泄漏轴要拿 **sealed-600** 去查,而 sealed-600 **尚未构建**
   (§8 第 2 项)。留在 (A) 会让 R013 被一个它自己不产生的前置无限期卡住。
2. **它本来就不是训练侧前置。** R013 训的是 VitaminC + NIAH **train** split 的 mutation-log 对;
   泄漏轴查的是**评测集**。泄漏若存在,污染的是"拿该模型去评 sealed-600"那一步,而非训练本身。
   **故它是"用模型"的前置,不是"训模型"的前置。**
3. **一条先验判断,但不构成豁免。** SNLI 取自 Flickr30k 图像描述、MNLI 取自 OANC 的十个体裁,
   **两者均不含 Wikipedia**;而 NIAH 语料是 dpr-w100(NQ 的 Wikipedia 切分),先验重叠面很小。
   **但 §11.7 刚把"构造保证"改成"需要证据",这段推理只能作背景,不得记作已验证。**

**为什么把移动理由写这么长:** §9.10 那份批准清单曾在实现已落地之后仍写着"未完成",
直到被别人发现才更正。**一个前置被移动而不留理由,与一份清单过时,是同一种缺陷。**

**[2026-08-06 补 —— 上面第 2 条成立,但单独读会得出错误结论,必须补全。]**
第 2 条说的是"泄漏轴查的是评测集,故不阻塞训练",这句话对。
**但它容易被读成"sealed-600 完全不阻塞 R013",而那是错的。**
sealed-600 通过**另一条完全独立的路径**留在 R013 的关键路径上 —— **§3.8 的硬约束**:
"NIAH 域适配对的 parent page 必须与 sealed-600 零重叠"。sealed-600 不存在则该重叠**无法核验**,
**R013 的域适配那一半因此不能合规执行**(VitaminC 主训那一半不受影响)。

**即:泄漏轴不阻塞训练,sealed-600 阻塞训练 —— 两者是不同的依赖,不可互相替代。**
本条与 §11.9(B) 的移动**都成立且互不矛盾**,但只看其中一条会算错 R013 的可开跑性。
tracker 的 R013–R015 已据此标 BLOCKED,阻塞原因写在行内。

### 11.9a 延后核验:两项可能重开选型的检查(**不阻塞 R013**)[2026-08-06]

**这两项与 §11.9 性质不同,必须分开放,否则会误把 R013 卡在一个被淘汰候选的问题上。**
所选基座 `cross-encoder/nli-deberta-v3-base` **不依赖这两项的结果**,R013 可照常开跑。
它们回答的是另一个问题:**tasksource 那一族是否本可以进候选池。**

**V1 — BIG-bench 的 `vitaminc_fact_verification` 是否与 VitaminC official test 相交。**

- 做法:下载 `tasksource/bigbench` 的该子集与 `tals/vitaminc` 的 **test** split,按 claim 文本做集合比对,
  同时对 train / validation 各做一次作对照。**必须先跑正样本对照**
  (拿一条确定来自 test 的 claim 去查,查得到才算方法可用)。
- 判读:**零相交** ⇒ tasksource 满足 H5 的举证责任(§11.5),可作为**另开修订**的候选;
  **有相交** ⇒ H5 确认不过,该族永久出局,本项记结案。
- **纪律(本项存在的理由):** 2026-08-06 本修订作者在本地跑过一次该比对,得到"三个 split 全零命中",
  **该结果作废** —— `tals/vitaminc` 的 `/rows` 端点返回 HTTP 错误,取数函数吞掉异常返回 `None`,
  而 `None` 被记成了"未找到"。**一个没有正样本对照的比对,其空结果没有信息量。**
  这正是本项目反复付学费的那一类缺陷:**产出合理数字而不是崩溃。**

**V2 — 即使 V1 判定零相交,基座见过 VitaminC train 是否破坏 §3.8 的去污染要求。**

- §3.8 要求 VitaminC 主训做 **revision-family 去污染**,该切分**严于 official split**。
  一个已在 VitaminC 上训过、且用的是更弱切分的基座,**破坏的是训练侧纪律,与评测集无关**。
- **故 V1 只是半个解。** V1 通过**不**足以让 tasksource 合格;V2 须单独裁决并写入该修订。

**强制机制(不靠"记得做"):** 本两项以 tracker 行 **R012e** 落账,状态与结论回填该行。
**在 R013–R015 的结果被写进任何报告之前,R012e 必须为 DONE 或被显式标注为"决定不做"并附理由。**
按 §10.8 的纪律 —— 只写在文档里、没有强制机制的义务,迟早被跳过;本节已经是第四次写下这句话。

### 11.10 本修订不改变什么,以及 §10.8 检查表的适用两项

1. **输出空间不变**(三类,A2 §10.2)。§10.8 检查表第 1–5 项**不适用**;第 6–7 项照做:
   - **第 6 项(平行文档):** `selector.md` 的基座表述与本修订一致,无须改;
     `gate0b-review-prep.md` 须补记 §11.0 第 2 条的比较对象错误;设计文档 §3.8 同源段落须同步。
   - **第 7 项(代码与测试):** 本修订**当前不落地任何代码**,训练脚本尚不存在。
     基座 id 与其 `id2label` 进入代码时**必须走注册表**(见 §11.9 第 4 项),不得硬编码在调用处。
2. Gate 0B 的两层、五项阈值、联合门、§9.5a 的阈值禁令,**全部不动**。
3. §3.8 除基座与 §11.7 第 1 条那句副作用外,**逐字不动**。
4. §10.11 的预先裁定继续有效。

### 11.11 利益冲突声明

**核心依据(§11.1 的可加载性)是环境事实,两条命令可独立复核,与 Gate 0B 的任何数字无关。**
选型判据(§11.2)在看候选之前写死,套用过程在 §11.5 全程可见。
这两点使本修订在事后偏误一轴上强于 A1、A2 —— 但**减轻不等于免除**:

- 本修订提出于项目**正想开跑 R013** 的时刻。
- **§11.3(b) 那个"不增开零训练臂"的决定,受益方是本项目自己。** 已连同其可被推翻的条件一并写明。
- **本修订的 v1 草稿曾给出一个不同的结论,并用一句错误的事实主张支持它。** v2 由一次候选调查推翻 v1。
  **审阅者应当据此加重对本节的审查,而不是因为 v2 更严谨就减轻。**

### 11.12 批准所需

- [x] 项目负责人批准,并将本文件版本号改为 `g2-proto-4` —— **2026-08-06 完成**
- [ ] **§11.9 的七项硬性前置全部实测通过 —— 在 R013 开跑前完成,不得与训练并行**
- [x] §3.8 正文的基座改写,并改写"泄漏轴天然为空"那句副作用(§11.7 第 1 条)
      —— **2026-08-06 完成**,同时落入 §11.5 裁决的 large 应急臂条文
- [x] `EXPERIMENT_TRACKER.md` R013–R015 的注记同步 —— **2026-08-06 完成**
- [x] `gate0b-review-prep.md` 补记 §11.0 第 2 条的比较对象更正 —— **2026-08-06 完成**
- [x] **项目负责人就 §11.3(a) 表态:"Granite 家族"是否应当成为选型判据**
      —— **2026-08-06 裁决:不作为判据**,记录见 §11.3(a)
- [x] **项目负责人就 §11.5 表态:"规模"是否应当成为选型判据**
      —— **2026-08-06 裁决:不作为判据,改为 §3.8 的预注册应急臂**,记录见 §11.5
- [x] §10.11 对"A3"的前向指称改为编号顺延 —— **2026-08-06 完成**

**批准后唯一未清项:§11.9 的七项前置**(集群上执行)。§11.9a 的 V1/V2 **不阻塞 R013**,由 tracker R012e 承接。

