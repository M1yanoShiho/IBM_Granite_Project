# Results summary — findings

已认证的实验发现(数字来自 `docs/hpc-run-log.md` 记录的 HPC 运行,raw 在 `results/`)。
统计单位=query,配对随机化 p + bootstrap CI(与 harm 同协议)。现有 Selector(S1–S7)与
Retriever(R1–R2)两节;Generator findings 由该模块补入。

## Selector

### S1 — corroboration 门显著降 harmful-in-context,但破召回守卫(E2,2026-07-23)

同一 bm25 池、同 α,唯一变量=门(gated-corroboration vs corroboration)。配对,n=1479。

- **Harmful-in-context −11.2pp**(0.680 → 0.569),p≈0,CI[−0.130,−0.093]。注入的反事实证据被挡出上下文。
- **Required recall −4.8pp**(0.868 → 0.820),**破预注册非劣下界 −0.01**。
- 门是**真实 Pareto 权衡**,非免费:收益显著,代价真实且超预算。诊断出来、摆明,不藏。

### S2 — 召回代价是**答案等价(聚类)**失败,非模型/检索失败(E1 + 低成本探针,2026-07-23)

E1 pool 级组件评估(exact-string 簇,标签零人工=provenance + official gold):

- `needle_gold_recovery` **.509**、`missed_conflict` **.386**(label-clean,精确 doc-id)、`false_conflict` .594(文本代理 noisy);selection-bias `multi_key_rate` .106(injector `len(normalized)!=1` 已滤掉别名假冲突,故为下界)。
- **根因链(CPU 探针,无 GPU):** needle-visibility 审计 → 答案在抽取器读到的段落里 **94.9%**(截断仅 5.1%、chunk-absent 0)→ 非截断/检索。needle probe → visible 失败中 recovered 634 / wrong 439 / none 127;wrong-reclassify → 439 里 **36%(158)是 exact-string 匹配 artifact**(模型对、字符串判错:"Apostle Paul"⊇"paul"、"2009"≡"in 2009"),lenient recovery **.509 → .634**(visible)。8B extractor 三次否掉(visibility / failure-mode / matching)。
- **定论:瓶颈是 exact-string 答案等价,不是模型能力,也不是可见性。** E2 的 −11.2pp harm 有相当部分非来自有原则的冲突检测;−4.8pp recall 大部分是簇碎裂误杀 gold。

### S3 — 门内 lenient 聚类**显著回收召回、零 harm 代价**(Phase 1 修复,2026-07-23)

门内换 lenient 答案等价(representative-anchored 贪心,containment/number/prep;`canonicalize`/injector/数据集全不动)。lenient vs exact gate-on,**同池**,配对。

- **Required recall +1.2pp**(0.820 → 0.832),**p≈0,CI[0.006,0.018]**(n=1848)——**显著**。
- **Harmful-in-context +0.3pp**(0.569 → 0.572),p=0.55,CI[−0.005,+0.012]——**不显著(无代价)**。
- 修复**显著且安全**,但幅度**适中**:回收 −4.8pp 门召回代价的 ~**25%**(0.012/0.048),仍未闭合 −0.01 守卫(−4.8pp→−3.6pp vs gate-off)。适中因:(a) injector 在本数据集设计掉了大部分 canonicalization 假冲突(10.6% 多别名题被跳);(b) 确定性匹配只吃 containment/number/prep,吃不到同义词/缩写。
- **未回收的 ~75%(含 39% 孪生 missed-conflict)= 语义等价(Graph 2.0)的残差,已量化。** 全建 NLI 前先跑 Phase 3 低成本探针:针对性"抽被查询属性的值"prompt 能否分开孪生 → 能则拿 Graph 2.0 主收益而不建 NLI,不能则 Graph 2.0 有据。

### S4 — 孪生 missed-conflict 是**抽取受限**,prompt 修不了;Graph 2.0 是经验赌注(Phase 3,2026-07-25)

对每注入题从 needle/cf 源文档直抽(隔离 prompt),baseline 现行 prompt vs 两个 targeted。n=1479。

| prompt | missed_conflict | needle_gold | cf_replacement |
|---|---|---|---|
| baseline | .281 | .486 | .211 |
| verbatim | .238 | **.363** | .183 |
| attribute | .254 | .479 | **.269** |

- **attribute 真但小(3B):** missed −2.7pp、**cf_replacement +5.8pp**(更常抽出注入 replacement)、gold 持平——正确方向,幅度小。
- ⚠️ **本表用 exact-string 计分,已被 S5 证明对"更啰嗦但正确"的输出有系统性偏见**(见 S5:同一模型 exact→lenient 差可达 +56pp)。**verbatim 那行"假胜"的判断因此不可靠**——verbatim 明确要求更长的 copied span,正是 exact 计分惩罚的形态;v2 未复跑 verbatim,该结论**存疑、不作引用**。
- **⚠️ 本节原结论"prompt 修不了孪生崩塌"已被 S5 推翻**:该结论只在 3B 上成立,且受 exact 计分混淆。8B 上 targeted prompt 近乎腰斩 missed。以 S5 为准。

### S5 — 孪生崩塌可被**容量 × 结构的交互**大幅修复(2×2,2026-07-25)

修掉 S4 探针的两个缺陷后重跑:(1) decoupled 的 Stage B 原先**没收到 question**(只给目标类型,无法判别passage里哪个实体)——已修;(2) 计分只用 exact 等价,系统性惩罚啰嗦但正确的输出——现**同时报 exact 与 lenient**。n=500/格,**下表为 lenient(可信列)**。

| 模型 × 策略 | missed_conflict | needle_gold | cf_replacement |
|---|---|---|---|
| 3B baseline | .318 | .576 | .248 |
| 3B attribute | .286 | .562 | .300 |
| 3B decoupled | .190 | **.442** | .240 |
| 8B baseline | **.444** | .664 | .296 |
| 8B attribute | .232 | .650 | .386 |
| **8B decoupled** | **.224** | **.690** | .372 |

- **计分伪影极大:** 8B 的 exact→lenient gold 差 **+56.6pp(attribute)/+62.6pp(decoupled)**。此前"8B 崩溃"(gold .084)**纯属 exact 计分伪影**,非能力问题;lenient 下 8B 反而**优于** 3B(baseline .664 vs .576)。
- **容量单独无效:** 8B baseline 的 missed **最差(.444)**——大模型配旧 prompt 更容易对孪生给同一答案。
- **结构单独无效:** 3B decoupled missed 降到 .190,但 gold 从 .576 塌到 .442 = 典型**假胜**形态。
- **容量 × 结构有效(真胜):** **8B + decoupled 把 missed 从 .444 砍到 .224(−22pp),同时 gold 持平/微升(.664→.690)、cf 抽取升(.296→.372)**——三项同向,不是假胜。8B + attribute 几乎同样好(.232)。n=500(SE≈.02),22pp 远超抽样噪声;gold 的 +2.6pp 在噪声内,只能称"持平"。
- **对 Graph 2.0:** 孪生 missed **可被抽取层大幅修复**(近腰斩),而非只能靠 NLI。真实杠杆确为**抽取质量**(S4 的方向对、结论错)。Graph 2.0 若上,应建在**更强抽取**之上;其增量价值须对照"8B+decoupled 抽取"这一新基线,而非对照旧的 .39 baseline。
- **限制:** 直抽源文档(隔离检索),非全管道;未做配对显著性(仅比对 SE);8B 成本更高;未验证收益能否级联到 E1/E2 的 in-pool 指标。

### S6 — S5 的收益**不级联**到真实检索池:解耦抽取在池内制造大量假冲突(2026-07-26)

把 S5 的 8B+decoupled 接到 **E2 真实 top-20 池**(cluster_eval,n=300,lenient 列)对照 3B+single:

| 指标 | 3B + single | 8B + decoupled | Δ |
|---|---|---|---|
| missed_conflict | .427 [.358,.499] | .308 [.253,.369] | −11.9pp |
| needle_gold_recovery | .609 [.547,.668] | .673 [.613,.729] | +6.4pp |
| **false_conflict** | **.494 [.421,.568]** | **.877 [.825,.916]** | **+38.3pp** |

- **三项里唯一统计确凿的是坏的那项:** missed 与 recovery 的 CI 互相重叠(提示性,未确立);false_conflict 的 CI **完全不重叠**,退化无疑。
- **机制:** decoupled 的 Stage A 每题只命名**一个**目标类型,再套到全部 20 段。孤立探针(S5)只有 needle+cf 两段 → 无害;真实池里其余 18 条干扰段被**逼着**吐出该类型的某个值 → 含 gold 的段与 needle 抽出不同答案 → **gold 碎裂 88%**。而 false_conflict 正是 E2 −4.8pp recall 的机制通道 → 接进门**大概率让 recall 更差**。
- **结论:S5 为真但不外推到两段以上。** 孤立探针高估了它,**池结构**才是破点。**"更强抽取"这条新基线因此不成立**——S5 结论(§S5 最后一条)据此下调:8B+decoupled 不是可用的系统级修复。
- **对 Graph 2.0(反而变强):** 失败根因是"每段必须吐一个答案"的单答案抽取**无法表达"本段与被问属性无关"**。而 Graph 2.0 的关系模型**设计上就有 UNKNOWN / 不建边**(TRAINING_PLAN §3.2、Gate 0B 的 abstention/coverage 指标)——正对这个失效模式。这是**用实测负结果**支撑 NLI 关系层的具体论据,而非泛泛而谈。
- **限制:** 两臂 denominator 不同(185 vs 237,"both clustered"随抽取策略变)→ 未做配对检验,仅各臂 Wilson CI;n=300 子采样。

### S7 — 零训练验证器的评估结论,**对 hypothesis 形式与标签记账的敏感度高于对模型选择的敏感度**(Gate 0B,2026-08-01/03)

Graph 2.0 的零训练关系模型验收门(R012,job 18235972)判 **FAIL**。但拆开来看,"模型不行"这个读数
**三次里有两次是测量方式造出来的**。这是本项目第三次自我推翻(前两次见 S4、S5)。

**起点是一个不自洽的读数。** 域对口 checkpoint `albert-xlarge-vitaminc-mnli`(59M 独立参数)在
VitaminC official test(0B-1,n=55197)上 macro-F1 **.922**、五项阈值全过;换到任务形状的探针
(0B-2,n=5888)上 gold-supports recall 只有 **.192**,被一个 0B-1 挂三项的通用臂
(`DeBERTa-v3-large-mnli-...`,.794)以 **4.1 倍**超过。能力不足应当是两臂同向退,不是这个形状。

**六个候选机制,全部由实验排除,不是由论证排除:**

| 机制 | 实测 | 判定 |
|---|---|---|
| premise 被截断 | `frac_over_cap` 两臂均 **0.000**(中位 150 token / cap 512) | 排除 |
| 标签序错配 | 同一映射下 0B-1 macro-F1 .922 —— 排列错了不可能有 .92 | 排除 |
| `canonicalize_answer` 改坏 gold claim | claim 中 answer 串 **97.6%** 逐字出现在 premise 中(干净对照组 100%) | 排除 |
| 大小写 | albert `do_lower_case=True`,对塌陷臂**可证无影响** | 排除 |
| premise 文本质量(CSV 转义残留) | 带残留 .369(n=899)vs 干净 .354(n=573),**< 1 SE**,且带残留反而略高 | 排除 |
| premise 长度 | 长 .362 vs 短 .365,**< 1 SE** | 排除 |

**两个是真的,而且量级都很大:**

1. **hypothesis 句式。** 冻结模板 `The answer to the question "{q}" is {a}.` 是一句**关于问题的元陈述**。
   改为 `{q}? {a}.` 后 albert:`twin_refutes_accuracy` **.674 → .760(越过 .70 阈值)**、
   gold-supports **.192 → .363**、弃权率 **.482 → .286**。而这个孪生判别正是 S4 记下的
   "三轮抽取层工作没能修好"的顽疾 —— **一个 59M 现成 checkpoint 在换掉句式后就做到了 .76。**
   反向证据同样重要:DeBERTa 在同一改动上**两项各退 12.9pp / 8.3pp**,因为 `{q}? {a}.` 去掉元指称的
   同时也不再是陈述句,而它训练分布是规范陈述句。**该阶梯每级并非只动一个变量,已如实记录。**
2. **三类 argmax 的记账方式。** 扫 `P(SUPPORTS)` 阈值,albert 的 gold-supports recall
   从 argmax 的 **.1916** 升至 θ=.05 的 **.7018(3.66×)**,rung 2 由 .3635 升至 .7799。
   **该 checkpoint 不是"判不出",是"不肯承诺"。**

**外加一个设计层面的发现:** 验收门要求关系模型输出三类,但 `independent_support` 只数 SUPPORTS 边,
主口径 `conflict_mode=distinct_cluster` 也不读 REFUTES ——**门在主口径下一次都没消费过第三类**。
按二分类(SUPPORTED / NOT-SUPPORTED)重读同一批预测,twin 项四种组合**全部通过**
(albert .998 / .987,DeBERTa .869 / .901),而 `gold_supports_recall` 不变、**四种组合仍全部未过**。
协议修订案(A1)据此起草,**其时间线、利益冲突与阈值防护条款一并写在案内**;在批准合入前,
**三类口径仍是唯一有效的 Gate 0B 判定**。

**结论(可独立于门的成败成立):** 零训练验证器的评估结论,对 **hypothesis 的句法形式**与
**标签空间的记账口径**的敏感度,**高于对模型选择的敏感度**。0B-1 式的外部效度**不能**外推到任务效度——
一个在官方 test 上 macro-F1 .922 的 checkpoint,可以在任务形状的探针上被测成 .19,而其中大部分是测量方式。
两层验收缺一不可,且第二层必须与机制实际消费的量对齐。

**范围限制(必须同时声明):** 预注册三臂中的 **MiniCheck-FT5**(唯一为 document-grounded verification
专训的一臂)因需二分类双向打分通路而未上场,故本轮**不支撑"任何零训练模型都不够"的族级断言**,
只支撑"这两类通用/域内 NLI checkpoint 在当前口径下不够"。rung 3(QA2D)亦未跑,是最后一个未测的测量变量。

**限制:** 0B-2 是**隔离对探针**,按 S6 纪律不许外推到池;阈值扫描曲线在全量上做,**已污染,不得用于选定阈值**。

## Retriever

### R1 — decompose 在多跳上的崩塌是**RRF 融合的排序失败**,不是检索失败(R-2wiki-decompose,2026-08-04)

2Wiki 上 decompose MRR .5702 vs strong-bm25 .9580(Δ **−0.3878**),是 3 数据集 × 8 变体矩阵里
唯一的灾难级退化(SciFact −.052、NQ −.047)。本轮在 jp25459 下重跑两臂取 per-case:
**5 指标逐位复现** MengW7(886cc8f)的数字,n=2000/臂。

- **伤害随检索深度单调收缩:** R@5 **−20.5pp**、R@10 −17.3pp、R@20 −9.6pp、
  Recall(top-50)**−0.7pp**。候选池几乎无损。
- **per-case:** strong-bm25 完美命中(MRR=1.0)的 1854/2000 条里,decompose **仅 8 条(0.4%)
  彻底丢 gold**,**1053 条(56.8%)gold 仍在池中、只是被排低**。降级子集平均 MRR ≈ .27 →
  **gold 典型地从 rank 1 滑到 rank 3–4**。
- **预注册的假设被推翻。** 原假设"子查询丢跨跳依赖 → 召回局部像/全局错的段落"会预测 gold
  掉出池子;实测 recall 几乎不动。预注册的两分支判定(集中 vs 均匀)命中**均匀**那支:
  worse 1138 / tied 827 / better 35,worst 20% 只占 48.5% 净损失(受影响的 1138 条上均匀
  应 ≈35%)——无承载崩塌的少数灾难 case,故根因在**融合环节**,非多跳难例。
- **机制:** RRF 加总 `1/(k+rank)`。多跳 gold 只回答**一跳**,在一个子查询列表排高、其余缺席;
  "各跳都沾一点、都不精准"的文档在**所有**列表拿中等分,累加后反超。**RRF 结构性奖励广谱平庸、
  惩罚单点精准,而多跳需要的正是单点精准。** 加重此效应的实现细节:`DecomposingRetriever`
  只融合子查询,**原始 query 仅在 LLM 空输出时作 fallback** → BM25 在 92.7% case 上把 gold
  排第 1 的最强信号从未进入融合。
- **结论:不应对多跳语料整体 gate off decompose**(池子是好的),而应先改融合:把原始 query
  作为一个融合臂加入或加权。预期 MRR 大幅回升、recall 基本不动。**该修法尚未测,是下一步。**

### R2 — 把原始 query 加回融合臂,**显著但只补回 37%**;多跳上 decompose 仍不划算(R-2wiki-decompose-orig,2026-08-04)

按 R1 的机制推断做的直接修法:`DecomposingRetriever` 新增 `include_original`(默认 False,
不影响任何既有结果),把未改写的原始 query 作为**一个额外融合臂**。2Wiki,n=2000,配对检验。

| 臂 | MRR | R@5 | R@10 | R@20 | Recall |
|---|---|---|---|---|---|
| decompose(基线) | .5702 | .4716 | .5491 | .6506 | .7610 |
| **decompose-orig(修法)** | **.7155** | .5727 | .6639 | .7371 | **.7675** |
| strong-bm25(上界参照) | .9580 | .6766 | .7222 | .7468 | .7678 |

- **修法真实且显著:** MRR **+0.1453 p=0.0000**、R@10 **+0.1148 p=0.0000**;recall 基本不动
  (+0.0065)。R1 的机制推断由此得到实验支持,不只是读码推断。
- **但只回收 37% 的 MRR 差距,且回收比例随深度递增:** MRR 37% < R@5 49% < R@10 66% <
  **R@20 90%**。即原始臂能可靠把 gold 拉回**前 20**,却抢不回**rank 1**——"1 票 vs N 票"
  稀释的指纹。**等权加入方向对、幅度不够;下一步是给原始臂加权**(未测)。
- **反向加强 R1:** 修法后 Recall .7675 vs strong-bm25 .7678,仅差 .0003——池子质量已等同,
  差的纯粹是排序,与"排序失败非检索失败"一致。
- **⚠️ 实用结论:修完仍明显不如直接用 strong-bm25**(.7155 vs .9580)。本修法补回的是
  decompose 的**自伤**,没有让它在多跳上变得有竞争力。R1 那句"不必对多跳整体 gate off"
  只就"候选池没坏"成立;**就该不该用而言,当前证据支持多跳上仍优先 strong-bm25。**

---

## 路线总结(3-phase route + 2×2 + 级联验证,2026-07)

诊断链(E1 + CPU 探针)把 E2 的 −4.8pp recall 代价定位到 **exact-string 答案等价**——这一诊断在 S5 又一次自我印证:**连我们自己的探针计分都栽在同一个坑上**。

**最终计分:**
- **唯一确立的系统级修复 = lenient 聚类**(S3,in-pool recall **+1.2pp p≈0**,harm 不变)。确定性、门内、零推理成本。
- **8B × decoupled 抽取(S5)在孤立探针上很漂亮(孪生 missed −22pp),但 S6 证明它不级联**——池内 false_conflict 从 .49 飙到 .88,而这正是伤 recall 的通道。**不采用。**
- 8B 本身的价值被平反了一半:前三次否决(recovery/matching 轴)是 exact 计分伪影;但**在真实池里它也没能变成可用修复**。
- 顽疾 **twin missed-conflict 仍在**(池内 .31–.43),且已证明**抽取层修不动**。

**对 Graph 2.0:** S6 的失败根因——单答案抽取**无法表达"本段与被问属性无关"**——正是 Graph 2.0 关系层 **UNKNOWN / 不建边**设计所针对的。所以这轮负结果**加强**了 NLI 关系层的论据,并给它一个具体的、可检验的必达目标:**在不推高 false_conflict 的前提下压低 missed_conflict**。Graph 2.0 的 Gate 0B(abstention rate、edge coverage、CLAIM_REFUTES precision ≥ .85)就是这个目标的验收口径。

rigor 底线已达:两个自查出的探针缺陷、两条被推翻的自家结论(S4、S5)、全部以实测更正。
