# Lenient-clustering selector fix + 3-phase route — design

状态:APPROVED(brainstorm 2026-07-23,Weikai;两处决定经 AskUserQuestion 拍板)。承接 E1 诊断
([[2026-07-23-e1-cluster-eval-design]]):exact-string 聚类把"同意"的答案劈碎(recovery 51%→63% lenient),
门把碎成少数派的 gold 误踢(E2 recall −4.8pp)。修法:门内按 lenient 等价聚类,修自一致性的计票。

## 1. 决定(已拍板)

- **Blast radius:** lenient 等价**只进门的聚类**。`canonicalize_answer`、injector、metrics、数据集**全不动** →
  数据集稳定、E2/E1 可比、门是干净单变量。
- **算法:** representative-anchored 贪心。候选按 `retrieval_rank` 顺序;每个 valid answer 加入**第一个**
  "代表(首成员 raw answer)与之 `lenient_equivalent`"的簇,否则开新簇。只比代表 → 无传递链、确定性、有界合并。

**关键不变量:** 贪心下各簇代表**两两非等价**(新簇的founding member当时与所有已存代表非等价;lenient_equivalent
对称)。又因 `lenient_equivalent` 第一支是 canonicalize 相等,非等价 ⇒ 不同簇的 `canonicalize(rep)` 也不同 ⇒
**不同簇仍有不同 `cluster.answer`**。故门 `_decide` 的 competitor 判定(`cluster.answer != own.answer`)在
lenient 下依旧正确 —— **无需改 `_decide`**,唯一改动是 `_gate` 换聚类函数。

## 2. 接口与改动(最小)

- `src/evidence_rag/selector/clusters.py` 新增
  `build_clusters_lenient(window, answers, equivalence) -> tuple[AnswerCluster, ...]`(同 dataclass 输出:
  `answer`=代表 canonical、`member_ids`、`independent_support`=去重 document_id 数)。exact `build_clusters` 不动。
- `GatedCorroborationSelector.__init__` 加 `equivalence: Literal["exact","lenient"] = "exact"`
  (默认 exact = **今日行为逐字不变**)。`_gate`([gated.py:156](../../../src/evidence_rag/selector/gated.py))
  按 equivalence 分派 `build_clusters` vs `build_clusters_lenient(..., lenient_equivalent)`。
- `composition.build_selector` 为 `gated-corroboration` 增 `equivalence` 参数(白名单校验,默认 exact)。
- 新 config 臂 `configs/experiments/niah_e2_gate_on_lenient.toml`(gate-on,equivalence=lenient)供 E2 度量。

**为什么有效:** gold 碎片合并 → gold `independent_support` 升 → 不再是被踢的少数派(recall↑);且更易以 margin
压过孤立 cf(harm 也可能↑)。cf 与 gold canonically 可分,lenient 第一支不合并异值 → **植入冲突被保留**(missed 不劣化)。

## 3. 度量(Phase 2,实现后)

E2 第三臂:`gate-on-lenient` vs `gate-on-exact` vs `gate-off`,配对 harm + required-recall。
主假设:lenient 抬 recall(少踢 gold),harm 不劣。复用 `run_selector_gate.slurm` 三件套模式。可选 E1 lenient 复评。

## 4. Phase 3(全面 Graph 2.0 前的低成本探查)

针对顽疾 `missed_conflict` 0.39:为什么孪生崩塌?抽取器是否抓了**未替换实体**?一个针对性
"抽取被查询属性的值"的 prompt 能否把孪生分开?**能修 → 拿到 Graph 2.0 主要收益而不建 NLI;不能 → Graph 2.0 有据可依。**
先跑这个再决定是否上 NLI 全栈。

## 5. 测试(TDD)

- `test_clusters.py`:containment 合并("apostle paul"+"paul");canonically 异值不合并(gold+cf);无传递链
  (A≡B、B≡C、A≢C,C 只比代表 A → 不经 B 串入);rank 顺序确定性;同 document_id 去重一票;invalid 排除。
- `test_gated.py`:equivalence="exact" 默认下 §7.2 全表**保持绿**;equivalence="lenient" 下 gold 碎片合并 →
  阻止对 gold 的误踢(构造 winner support 3 vs 碎 gold 各 1:exact 踢 gold、lenient 合并后 support 2 不踢)。
- `composition`:gated-corroboration 接受 `equivalence`;非法值报错;默认 exact。

## 6. 非目标

不改 `canonicalize_answer` / injector / metrics / `_decide`;不引 NLI(属 Phase 3 结论后的 Graph 2.0);
lenient 只用于门的聚类,E1/E2 的 metric canonicalize 保持严格(度量一致性)。
