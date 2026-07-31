# Results summary — findings

已认证的实验发现(数字来自 `docs/hpc-run-log.md` 记录的 HPC 运行,raw 在 `results/`)。
统计单位=query,配对随机化 p + bootstrap CI(与 harm 同协议)。本文件先落 Selector 模块;
Retriever/Generator findings 由各模块补入。

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

## 路线总结(3-phase route + 2×2 + 级联验证,2026-07)

诊断链(E1 + CPU 探针)把 E2 的 −4.8pp recall 代价定位到 **exact-string 答案等价**——这一诊断在 S5 又一次自我印证:**连我们自己的探针计分都栽在同一个坑上**。

**最终计分:**
- **唯一确立的系统级修复 = lenient 聚类**(S3,in-pool recall **+1.2pp p≈0**,harm 不变)。确定性、门内、零推理成本。
- **8B × decoupled 抽取(S5)在孤立探针上很漂亮(孪生 missed −22pp),但 S6 证明它不级联**——池内 false_conflict 从 .49 飙到 .88,而这正是伤 recall 的通道。**不采用。**
- 8B 本身的价值被平反了一半:前三次否决(recovery/matching 轴)是 exact 计分伪影;但**在真实池里它也没能变成可用修复**。
- 顽疾 **twin missed-conflict 仍在**(池内 .31–.43),且已证明**抽取层修不动**。

**对 Graph 2.0:** S6 的失败根因——单答案抽取**无法表达"本段与被问属性无关"**——正是 Graph 2.0 关系层 **UNKNOWN / 不建边**设计所针对的。所以这轮负结果**加强**了 NLI 关系层的论据,并给它一个具体的、可检验的必达目标:**在不推高 false_conflict 的前提下压低 missed_conflict**。Graph 2.0 的 Gate 0B(abstention rate、edge coverage、CLAIM_REFUTES precision ≥ .85)就是这个目标的验收口径。

rigor 底线已达:两个自查出的探针缺陷、两条被推翻的自家结论(S4、S5)、全部以实测更正。
