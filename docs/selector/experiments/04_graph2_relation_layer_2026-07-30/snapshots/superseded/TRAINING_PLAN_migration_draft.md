# Graph-Assisted Evidence Selector 2.0 — 实验执行计划

## [2026-08-07 阅读须知] 本文件是迁移草稿,多处已被 M0 取代

**本文件是 2026-07-11 的迁移草稿,自那以后从未按 M0 逐条修订过。**
**凡 M0 冻结的条目,一律以 [M0_PROTOCOL_FREEZE.md](M0_PROTOCOL_FREEZE.md) 为准。**
这条优先级规则本就写在 M0 开篇;此处重述一遍,是因为**读本文件的人不会先去读 M0**。

**为什么光有优先级规则不够 —— 用已经发生的那一处说明,而不是讲道理:**
优先级规则**防不住伤害,因为什么都不会报错**。本文件 §4.2 原写"每题保留固定 q2d_granite Top-20",
而 M0 §4 冻结的检索器是 **bm25**。若有人照本文件把候选池建出来:
`candidate_sets.jsonl` **不携带检索器 id**,Gate 0A 验的是**窗口形状**(每题 20 条、rank 连续 1..N)
而**不是来源** —— 于是这批用错检索器的数据**在任何一环都不会报错**。
真正被无声打断的是**预注册阈值**:M0 §3.5 的 G-FC 基线 **0.4355** 与 §5.2 的 gate-off 参照
(Graph 1.0-lenient **−3.6pp**)**都是在 bm25 池上量出来的**,池一换,门就变成**跨池比较**,
并且照样打印出一组看着完全合理的数字。
**照本文件走的人,不会从系统那里得到任何警告。这就是本须知存在的理由。**

### 已被取代的条款一览

**A 组 —— 未曾声明的分歧。** M0 开篇只声明了四处实质修改(见 B 组),下列各条**从未被任何地方提醒过**,故列在前。

| # | 本文件怎么写 | M0 改成什么 | 治理条款 |
|---|---|---|---|
| **A1** | §1.2:Required Evidence Recall@10 **只对一个基准**报单侧非劣下界 | recall 非劣**必须对两个基准同时报**;**头条判据是 (b) gate-off**,而 Graph 1.0-lenient 在该口径为 **−3.6pp,FAIL** | **M0 §5.2 [冻结]** |
| **A2** | §2 模块图与 §3.6:选择器 = **LightGBM LambdaRank**,读 utility labels 训练,v1/v2 同训练预算、同 tie-break | **D1=A ⇒ 选择器不含任何学习参数**;本仓库**没有任何东西**产出 `utility_grade` | **M0 §1 D1** |
| **A3** | §1.2:C1 = Graph 2.0 **vs 重训的 ML Selector v1**,三条件 | C1 = Graph 2.0 (gate-on) **vs Graph 1.0-lenient (gate-on)**,gate-off 作绝对参照,并**增 G-FC 为第四条件** | **M0 §5.1**(G-FC 见 §3.5) |
| **A4** | `q2d_granite` 仍活在**三处**:开篇"适用范围"、§1.1 核心研究问题、§2 模块图(行号见表下) | 检索器**冻结 = bm25**(与 E2 同臂) | **M0 §4** |
| **A5** | §3.2:三类核心关系平铺(SUPPORTS / REFUTES / SAME_SOURCE),不区分产出口径与判读口径 | A1→A2 之后:**模型产三类,门与 0B-2 折叠为二类**读出 | **M0 §9、§10** |
| **A6** | §4 正文:开始训练前**必须物化 2000 个 NIAH train query**,否则须在 M0 预注册改训练规模 | 该问题**已消解** | **M0 §1 D2** |

**B 组 —— M0 开篇已声明的四处实质修改。** M0 声明了,但**本文件正文同样从未标注**,照读仍会出错。

| # | 本文件怎么写 | M0 改成什么 | 治理条款 |
|---|---|---|---|
| **B1** | §4 表首行、§3.4、§4.4:ContractNLI 用于 claim–passage NLI,并进关系训练、调参与一次测试 | ContractNLI **移出关系监督集** | **M0 §3.0 [实质修改]** |
| **B2** | §5:Gate 0B 为**单层**(ContractNLI/VitaminC dev 调参,各自 official test 跑一次) | Gate 0B **拆两层**:0B-1 外部效度 / 0B-2 任务效度,阈值各自独立 | **M0 §3.2 [实质修改]** |
| **B3** | §3.4:SAME_SOURCE **只**由 source_parent_id 确定性生成 | SAME_SOURCE **无条件修复** | **M0 §3.4 [实质修改]** |
| **B4** | §3.6:种子 **13/42/73**,三 seed 标准化预测分数取平均作唯一 primary ensemble | **废除 3-seed ensemble**(全链路无随机源);仅当 M0 §3.8 训练路径启动时,对关系模型恢复 | **M0 §5.4 [实质修改]** |

**A4 的三处行号(按本须知插入后的行号):** 第 78 行(适用范围)、第 103 行(§1.1)、第 140 行(§2 模块图)。
§4.2 那处(第 254 行)已于 2026-08-07 就地更正并存证。**A6 在第 234 行。**

**A1 是本表最危险的一条,它与其余各条性质不同。** 其余是**过时指称**,照读会得到错误的做法;
A1 是**本文件授权了一个被明令禁止的动作** —— §1.2 照字面执行,就是只报口径 (a) 通过即宣布成功,
而 M0 §5.2 用几乎同样的字面把这件事禁掉了:"只报 (a) 通过就宣布成功,等于把未闭合的守卫藏起来。禁止。"
**一份文件放行另一份文件明文禁止的动作,不是措辞过时,是可以直接产出错误结论的授权。**

**A2 是架构性的,不是措辞问题。** 它同时是 §4.2 所记第 4 条待裁项(`utility range` / `derived flags`
在 D1=A 下没有生产者)的**根因**:那两项之所以无人生产,正是因为选择器已经没有学习参数,
而本文件仍旧按"有学习参数"来写。

**本表是一次排查的结果,不是穷尽审计。**
**不得据此认为"除表列各条以外,本文件可以照读"。** M0 §9.12 已把同一类教训记过两次:
A1 的覆盖面当初也是靠一次关键词扫描摸底,其自述"扫描不是审计"事后被证明是准确的 ——
重做的审计发现那份清单**既有多余项、也有遗漏项**。本表的处境完全相同。
**凡本文件与 M0 有任何出入,一律以 M0 为准,而不以本表是否列出为准。**

**本须知只记录何者已被取代,不裁决任何事。** §4.2 末尾所记的六条待裁项**仍然全部待裁**,裁决人:项目负责人。

---

> **清理迁移状态：待重新设计，不得直接开始训练。**
>
> 这份文件从旧项目迁移，用于保留已经思考过的 Graph Selector 2.0 方案。
> 旧项目的 NIAH、人工假针、counterfactual、标签和评估假设不自动成为新项目方案。
> 清理完成后，Selector 小组要先重新确认任务、数据、标签、指标和共享存储。
> 当前唯一确定的跨模块接口是：候选证据集合 → 排序后的 selected evidence IDs。
> FinanceBench 由团队约定保留给完整 RAG 最终对比，不用于 Selector V2。

**项目：** IBM Granite Needle-in-a-Haystack RAG

**日期：** 2026-07-11

**修订：** zero-new-human-annotation protocol

**状态：** 迁移后的设计草稿；尚未批准执行

**适用范围：** 模块二 q2d_granite Top-20 → Top-10；不修改模块一 Retriever，不替代模块三 grounded generation

**数据边界：** FinanceBench 不属于 Selector V2 的训练、开发或测试数据；本规则由团队文档约定。

> **V1/V2 边界：** V1 是历史探索，V2 从新的任务和数据设计重新确认。FinanceBench 只属于完整 RAG 的最终系统对比，不参与本 Selector 计划。

## 0. 旧计划审核记录

旧版本曾检查过这份方案的内部一致性，但该检查不等于新项目已经批准执行。
清理后必须按照新的模块定义重新审核任务、数据构造、标签和实验条件。

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

> **A4 指针(2026-08-09,g2-proto-5):** 本文件中一切"NIAH train"字样受 M0 §12 约束——实测该
> split 从未真正划分(三个运行 manifest 的 `split` 均为构建默认值 `'dev'`,适配池与 dev 评测运行
> 相交 202/2000 query)。域适配对**剔除**与 dev 评测重合的 family,守卫按 query-ID 集合代码强制、
> 永不读 `split` 标签;凡与此冲突的旧文以 M0 §12 为准。

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

FinanceBench 不属于 Selector V2 的训练、开发、调参、测试或失败分析数据。团队将它保留给完整 RAG 的最终系统对比。

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

  **[2026-08-07 更正 —— 本条与 M0 §4 冲突,以 M0 为准;原文逐字保留存证,不改写。]**
  检索器**冻结为 bm25**(与 E2 同臂),**不是 `q2d_granite`**。M0 开篇已定"凡本文件冻结的条目,以本文件为准",
  故这不是一次新决定,只是本条作为 2026-07-11 迁移稿的原文,在 M0 §4(2026-07-30)冻结 bm25 之后没有回头改。

  **M0 §4 的理由须原样保留,不得转述掉:固定候选池正是冻结的模块接口**;Graph 2.0 的主张
  ("给定固定候选池,选择器更可靠")本身就是**条件于池固定**的;换检索器会把**池的组成变成混淆变量**;
  且 **E2 exact 臂的 dump 只有在池相同时才可复用**。
  队友的 StrongBM25/hybrid 按 M0 §4 作**独立泛化臂单列一张表**,不与主表合并、不进 C1 判定。

  **为什么这处旧字眼是致命的 —— 它不会以崩溃的形式暴露:**
  `candidate_sets.jsonl` **不携带检索器 id**(`CandidateSet` 只有 `query_id` 与 `candidates`,
  `EvidenceCandidate` 只有 retrieval_score / retrieval_rank / source_uri),
  而 Gate 0A(本文件 §5、M0 §6 第 3 项)验的是**窗口形状** —— 每题 20 条、rank 连续 1..N、
  doc 落在本语料内 —— **不验来源**。于是"用错检索器建出来的一批数据"**在任何一环都不会报错**。
  真正被无声打断的是**预注册阈值**:M0 §3.5 的 G-FC 基线 **0.4355**(429/985,R001)与
  §5.2 的 gate-off recall 参照(Graph 1.0-lenient **−3.6pp**)**都是在 bm25 池上量出来的**。
  池一换,门就变成**跨池比较**,而它照样打印出一组看着完全合理的数字。

  **本次更正的作用域仅限本节。** 同一处旧字眼在本文件 §"适用范围"(开篇)、§1.1、§2 的模块图中仍在,
  未一并改动 —— 一并列出以免读者误以为全文已对齐;如何处置由项目负责人裁,不在本文件落定。

- 只接受具有单一 normalized gold value、且某个 gold surface alias 在 needle 中恰好出现一次的 query；
- replacement 只从 seed=42 冻结的 NIAH-train answer bank 中选择，并与当前 gold value 不同；
- replacement 必须与 gold 属于同一机械字符串类别：integer、decimal、year/date、proper-name token-length bucket 或 common-noun token-length bucket；无法归类或没有同类 replacement 时丢弃；
- 替换后必须满足：所有 gold alias 均不残留、replacement 不属于 gold aliases、只有目标字符 span 发生变化、文本可由 mutation log 反向恢复；
- mutation log 必须记录 answer-bank hash、原答案、替代值、字符串类别、seed、变换位置和变换前后文本 hash；
- 多个互异 gold value、替代值可能仍被官方 aliases 接受、实体类别不匹配或任何规则无法机械确认时，样本自动丢弃；
- 在训练和 dev 实验前只冻结 manifest/hash，不查看模型结果。

**[2026-08-07 记录 —— 2026-08-06 构建 sealed-600 时暴露的六个问题,一律待裁决,本节不作决定。]**

本项目把"agent 私自把一个预注册问题决掉"视为严重缺陷,故以下每条只写**问题、可选项、该谁裁**,不给结论。
**裁决人:项目负责人**;协议条目的改动一律走 M0,不在本文件落定。
第 1–3 条影响 sealed-600 的**数据本身**,须在 manifest 冻结、Gate 0A 转 PASS 之前有裁决。

1. **父页面轴与 passage hash 轴的作用域从未定义。**
   本节只写"与 train/dev/旧 test ... 零重叠",没写**在哪些文档上**取零重叠。
   而 M0 §4 同时要求 distractor **从同一个 21M dump 重新蓄水池采样** —— 两侧的 distractor 页面按构造就会重合,
   **"全部文档零重叠"因此是不可能达成的**,不是难达成。
   构建器实际取的是**只算 gold passage**(`materializer/sealed600.py` 的 `fingerprint_bundle`),
   理由记在该函数的 docstring:本文件 §4.4 按父页面、answer entity、synthetic family 分组,而这三者都是 **gold 的属性**。
   - 选项 (a) 追认"仅 gold passage",把作用域写进 M0 §4 的五轴表;
   - 选项 (b) 改判"gold + 全部 distractor",则须同时废掉 M0 §4 的语料重建条款,否则两条自相矛盾;
   - 选项 (c) 另立一轴单独约束 distractor 重合度 —— 须给阈值,不能只写"尽量少"。
   **未裁决前,2026-08-06 那组零重叠数字(见 M0 §8 第 2 条)只在 (a) 的读法下成立。**

2. **来源池未指定。** 本节写"从现有 split 未使用的 DPR/NQ query 中抽取",没说是 **dev 的剩余**还是 **train**。
   两者的题分布不同,也会改变第 6 条那条拒绝率。
   构建器**不替裁决人设默认值**:`build_sealed600 --split` 为 **required、无 default**
   (其 CLI docstring 写明 which pool the fresh queries come from is a protocol decision)。
   2026-08-06 那次以 `--split dev` 跑,该取值已记入 manifest 的 `source_dataset_id`。
   - 选项 (a) 追认 dev 剩余;(b) 改判 train,则 2026-08-06 的 600 题**必须重建**。

3. **轴 5 的定义两份文件不一致。** 本节称"模板族"(§4.4 同);
   M0 §4 的五轴表定义为 `(gold_value, replacement_value, 机械类别)` 三元组。
   构建器取的是 **M0 的定义**,且**从 `relations.task_probe.synthetic_family` 导入**(`sealed600.py:59`),
   与训练侧 OOF 分组(`relations/niah_adaptation.py`)**共用同一个函数** ——
   这样审计口径与训练分组不可能各自漂移。
   - 待裁:确认以 M0 的三元组为准并把本节与 §4.4 的"模板族"一并对齐,还是另有第三种定义。

4. **M0 §6 第 3 项的 `utility range` / `derived flags` 在 D1=A 下没有生产者。**
   这两项是从本文件 §5 的 Gate 0A 继承下来的,而 §5 那份写于 **LightGBM 选择器**时代;
   M0 §1 D1=A 之后**选择器不含任何学习参数**,本仓库**没有任何东西**产出 `utility_grade`。
   审计的处置是记 `not_applicable` **并附上这条理由**,且 CLI 要求显式传 `--utility-labels-absent`
   (不传即报错)—— 让"这项没验"**不可能以沉默的形式通过**。
   - 选项 (a) 从 M0 §6 删除该项并注明 D1=A 是原因;(b) 保留但永久标 `not_applicable`;
     (c) 若将来 M0 §3.8 训练路径引入学习参数,再决定是否复活。
   **本项不是 PASS。** M0 §8 第 2 条已记:本实现不产生"带脚注的 PASS"。

5. **本节"旧 test"(即 §4 表中的"旧 sealed NIAH 300")在本仓库没有对应产物。**
   本文件 §1.4 与 §4 的表格都引用它(用途:代码回归与昨日结果复现),但仓库里查不到该 split 的任何 artifact。
   - 选项 (a) 它确实存在于别处,则**必须作为另一个已有 split 提供**,并进 `--existing-split` 一起审五轴;
   - 选项 (b) 它不存在,则**这项对比根本没有在做**,§1.4 与本节的相关措辞须撤下,
     且**报告必须明写"没有做这项对比"**,而不是沿用旧文让读者以为做了。

6. **由本节算法反推出的 skip-rate,量的是一个过滤条件不同的池。数字以 M0 §4 为准,本处只指路。**
   M0 §4 的规模反推声明"构建算法沿用本节全文",其 `.261` 是在 **2000 条未经泄漏过滤**的 dev query 上量的,
   而 900 是在**泄漏过滤之后**取的 —— 两者不是同一个总体。
   2026-08-06 实测:`attempted 816`(非 812)、realised skip rate **.2647**;
   且**泄漏过滤自身的拒绝率此前从未量过**,现已量到 `considered 6515` 中拒绝 `4407`(≈68%)。
   **谁若按 812 取值将不足。** 下次重建须按 816 与 .2647 重算,**不在本节另存一份数字**。

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
