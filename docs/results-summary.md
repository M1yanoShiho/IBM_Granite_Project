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
