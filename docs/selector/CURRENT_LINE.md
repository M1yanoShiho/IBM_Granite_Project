# Selector — 实现了什么、按什么思路、为什么这么设计、主线是什么

**日期:** 2026-07-29 | **分支:** `refactor/three-module-baseline` | **范围:** 模块二 Selector(不改 Retriever / Generator 契约)

数字全部来自 `docs/hpc-run-log.md` 记录的 HPC 运行,raw 在 `results/`;findings 正文在 `docs/results-summary.md` S1–S6。Graph 2.0 的协议细节见 `docs/selector/TRAINING_PLAN.md`(迁移草稿)与 `EXPERIMENT_TRACKER.md`(Run 表,全 TODO)。

---

## 0. 主线(一句话)

**在单答案事实题上,用池内相对整数票做一道对抗门,把被压倒的矛盾证据挡出生成上下文(harmful-in-context −11.2pp);再把这道门唯一的真实代价(召回)逐层归因到底层的"答案等价 / 抽取质量",而不是归因到门的规则本身。**

可检验推论:修门的杠杆在**答案等价 / 抽取**这一层,不在往门里加更多判断信号。这条推论被验证了一半,又被我们自己推翻一半:确定性的 lenient 等价修复成立(S3),而"更强抽取"(8B + decoupled)在孤立探针上很漂亮却**不级联**到真实检索池(S6)。抽取层这条杠杆到此见底,顽疾(孪生 missed-conflict)因此移交给关系层 —— 这就是 Graph 2.0 的立项理由(见 §5)。

---

## 1. 我们实现了什么

### 1.1 生产选择器路径(`src/evidence_rag/selector/`)

| 文件 | 内容 |
|---|---|
| `top_k.py` | 可替换基线(按检索序截断) |
| `extraction.py` | 共享抽取底座:每个查询窗口一次抽取(`passage_chars=600`),重排段与门段共用 → **门零新增 LLM 调用**;含 `EXTRACT_PROMPT` 与 parametric 自答 |
| `corroboration.py` | 容错重排:`blended = α·relevance + (1−α)·corroboration`(α 默认 0.6) |
| `answer_norm.py` | 确定性归一化(千分位、货币、量级词、百分号、有限日期白名单)+ 精确相等 |
| `answer_equivalence.py` | lenient 等价:强归一化(前置虚词 + 数字词)后相等或一方为另一方的连续 token 子序列;**不含同义词/缩写**(那是 NLI 的活) |
| `clusters.py` | `build_clusters`(exact)与 `build_clusters_lenient`(representative-anchored 贪心);`independent_support` = 簇内去重 **distinct document_id** 数 |
| `gated.py` | `GatedCorroborationSelector`(重排 + 门 + 截断)、`GatedCoverageSelector`(门 + 集合覆盖打包);门决策走 `on_gate_decision` 回调,不进冻结契约 |
| `coverage.py` | A2 互补覆盖层:对幸存者按确定性 query+answer 特征做贪心集合覆盖,替代"逐点打分后截断" |

注册在 [composition.py:332](../../src/evidence_rag/composition.py#L332),参数白名单 `alpha / margin / support_cap / top_n / equivalence`;实验臂 `configs/experiments/niah_e2_gate_{on,off,on_lenient}.toml`。

### 1.2 评测数据构造(Materializer B 反事实注入器,纯 CPU、确定性、零人工标注)

对每道合格题复制 gold passage、把答案 span 机械替换成同类别的不同值,生成"投毒孪生"(`cf::<qid>::needle`)。孪生与真 needle 文本近重复 → bm25 必然共召回,门天然有活干。产 `provenance.jsonl`(gold_value / gold_alias_used / replacement_value)= 可反演的 harmful 标签。

资格过滤(重要,与下面的诚实项直接相关):gold 值必须归一化到**唯一**值、别名在某条 gold passage 里**恰好出现一次**、可归机械类别、存在合法替换值。任一不满足则跳过,不产脏样本。

基座:dpr-w100 NQ dev,子采样 100k passage / 2000 query,seed 42。

### 1.3 评测与诊断 harness(`src/evidence_rag/evaluation/`)

- `harm.py` / `harm_cli`:`selector.core.harmful_in_context`(孪生 doc 是否进选中集,方向 lower)+ pool-hit + 配对随机化 p + bootstrap CI。**离线**算,不碰共享 `ExperimentWorkflow`。
- `paired_metric_cli`:任意 stage metric 的配对随机化检验(用于 recall 显著性)。
- `cluster_eval`:E1 组件评估 —— missed-conflict / false-conflict / needle-gold-recovery + 注入 selection-bias(multi-key rate .106 / skip rate .261);支持 `--extraction single|decoupled` 与 exact/lenient 双计分。`--limit` 下 skip_rate 的分母误报已修(c5f813f)。
- `needle_visibility` → `needle_probe` → `wrong_reclassify`:纯 CPU/单模型的根因链(截断?抽不出?还是匹配判错?)。
- `missed_conflict_probe`:prompt × 模型 × 抽取策略探针(baseline / verbatim / attribute / decoupled 两段式),exact 与 lenient 双计分 + 原始回答 dump。

工程纪律:全程 TDD;`ruff check src tests` + `mypy src tests/typecheck.py` 全树干净;每个 HPC 实验 = 代码 + slurm 脚本 + 台账条目三件套,提交前填 BEFORE(目的/假设/预期指标+方向/精确命令/commit hash),拉回填 AFTER。

---

## 2. 按什么思路实现(数据流 + 门规则)

```
CandidateSet(来自 Retriever,bm25 池)
  ↓ 取窗口 top_n=20(窗外候选不抽取,原序尾随)
  ↓ 一次答案抽取(重排段与门段共用)
  ↓ 答案归一化 → 聚簇(exact 或 lenient)
  ↓ 独立票:同 document_id 只算一票;parametric 自答不计入门内票
  ├─ 重排段(容错):blended = α·relevance + (1−α)·corroboration
  ├─ 门段(苛刻):四条件判 drop
  ↓ 幸存者按 blended 降序 + 窗外尾随,截断至 max_selected(不足不硬凑)
SelectionResult
```

门规则(冻结,[gated.py:113](../../src/evidence_rag/selector/gated.py#L113)):设候选 c 属答案簇 K,竞争簇 K′ = 池内答案不同的簇中 `independent_support` 最大者。**drop c 当且仅当四条同时成立:**

1. c 抽出了有效答案(抽不出的永不被踢);
2. 存在竞争簇 K′(单答案假设下"答案不同"即矛盾,无需 NLI);
3. `support(K′) − support(K) ≥ margin`(默认 2);
4. `support(K) ≤ support_cap`(默认 1)——只剔"孤立且被压倒"的签名。

门在重排之后、截断之前。drop 腾出的槽位由更低排位的幸存者补进 —— 这是 drop 相对 demote 的核心收益:毒出窗口的同时替补进窗口。

---

## 3. 为什么这么设计(每条承诺锚在一条已验证结论上)

1. **踢人路径上不许出现绝对判断** → 门内只有同池整数票差,免校准;打分系统整体偏移不改变票差。
   依据:V1 学习型选择器整轮 —— 排序质量与正确证据保留率都升,但 harmful-in-context 未降,72% 特征重要性塌回 relevance/rank;ContractNLI 迁移 −0.134 演示了绝对刻度跨域错位。
2. **零票孤证不许被踢** → 无竞争簇 ⇒ 条件 2 假 ⇒ **结构性**安全,不是调参结果。
   依据:孤立假针与孤源真针观测上不可区分;误杀唯一真证据比放过一条毒更严重(毒进上下文生成器仍可能答对,真证据被删则物理性无米下锅)。守卫:Required Recall 非劣下界 −0.01。
3. **失效方向 = 闭嘴** → 投票碎裂域所有簇票数同降 → 票差缩小 → 门不 fire → 自动退化为纯重排。信号坏掉时静默,不乱杀。
4. **标签只来自 official / deterministic provenance,LLM 只当抽取器不当裁判**。
   依据:LLM-judge kappa 0.005–0.08 ≈ 噪声。故门的主信号取跨来源互证(唯一已认证携带增量的真伪信号,nested-CV +0.037,p=0.036),评测标签只用 qrels + counterfactual mutation log,零新增人工标注。这一点与 ArbGraph(Table 4 用 200 对人工标注做组件评估)形成方法学对照。
5. **信号毕业制** → 未验证信号(来源权威性、时效、语料一致性、模型先验)只能进重排影响排序(容错,故可试),经逐域验证后才能"毕业"进门影响踢留(苛刻,故须先证)。现有 parametric vote 属"模型先验",留在重排、禁入门内。

诚实边界(写进报告,不藏):**独立转抄的错误信息**(多个真独立的错源 vs 单一真源)会反转互证信号,本门会踢错。`document_id` 去重只能拦共享出处的转抄。这是 corroboration 一族的原理性边界,出路只有"权威/时效信号完成逐域验证后毕业进门"或"维持保守 margin 让门闭嘴"。

---

## 4. 研究主线与证据链(S1 → S6)

统计单位 = query,配对随机化 p + bootstrap CI。判定用**双 Gate**:harmful-in-context 显著下降 **且** Required Recall 非劣下界 −0.01。

- **S1 — 门有效,但代价真实且超预算(E2)。** 同池、同 α,唯一变量 = 门。harmful-in-context 0.680 → 0.569(**−11.2pp**,p≈0,CI[−0.130,−0.093],n=1479);Required Recall 0.868 → 0.820(**−4.8pp**),破 −0.01 守卫。结论:真实 Pareto 权衡,不是免费的午餐。
- **S2 — 代价的根因是"答案等价",不是模型能力也不是可见性(E1 + CPU 探针)。** E1 池级组件评估:`needle_gold_recovery` .509、`missed_conflict` .386、`false_conflict` .594(文本代理,noisy)。可见性审计:答案在抽取器读到的段落里 **94.9%**(截断仅 5.1%,chunk-absent 0)→ 非截断非检索。needle probe:visible 失败里 recovered 634 / wrong 439 / none 127;wrong 重分类:439 里 **36%(158 条)是 exact-string 匹配伪影**("Apostle Paul" ⊇ "paul"、"2009" ≡ "in 2009"),lenient 下 recovery .509 → .634。
- **S3 — 门内换 lenient 聚类:显著回收召回、零 harm 代价(Phase 1 修复)。** Required Recall 0.820 → **0.832**(+1.2pp,p≈0,CI[.006,.018],n=1848);harm 0.569 → 0.572(+0.3pp,p=0.55,不显著)。回收了 −4.8pp 代价的 ~25%,守卫仍未闭合(−3.6pp)。幅度适中的两个原因:(a) 注入器的资格过滤把大部分别名假冲突**设计掉**了(multi-key skip 10.6%),(b) 确定性匹配吃不到同义词/缩写。
- **S4 — 残差主体是孪生 missed-conflict,当时判"prompt 修不了"(已被 S5 部分推翻)。** 源文档直抽探针(3B,exact 计分):baseline missed .281,verbatim .238(靠抽取变噪的假胜,gold −12.3pp),attribute .254(真但小,cf_replacement +5.8pp)。
- **S5 — 在孤立探针里,孪生崩塌可被"容量 × 结构"的交互大幅修复(2×2,源文档直抽,lenient 计分,n=500/格;级联性由 S6 否决)。**

  | 模型 × 策略 | missed_conflict | needle_gold | cf_replacement |
  |---|---|---|---|
  | 3B baseline | .318 | .576 | .248 |
  | 3B decoupled | .190 | .442(塌) | .240 |
  | 8B baseline | .444(最差) | .664 | .296 |
  | **8B decoupled** | **.224** | **.690** | .372 |

  容量单独无效(8B + 旧 prompt missed 最差);结构单独无效(3B decoupled 是 gold 塌陷的假胜);**容量 × 结构才是真胜**:missed .444 → .224(−22pp)且 gold 持平微升、cf 抽取上升,三项同向。同时抓出我们**自己探针的计分缺陷**:8B 的 exact→lenient gold 差 +56~63pp,此前"8B 崩溃"纯属 exact 计分伪影。据此推翻两条旧结论:"prompt 修不了孪生崩塌"(只在 3B 且 exact 计分下成立)与"8B 无用"(前三次否决只在 recovery/matching 轴)。

- **S6 — S5 的收益不级联到真实池:干净的负结果(E1-cascade,n=300,lenient 列)。** 把 8B + decoupled 接到 E2 真实 top-20 池,对照 3B + single:

  | 指标 | 3B + single | 8B + decoupled | Δ |
  |---|---|---|---|
  | missed_conflict | .427 [.358,.499] | .308 [.253,.369] | −11.9pp(CI 重叠,仅提示性) |
  | needle_gold_recovery | .609 [.547,.668] | .673 [.613,.729] | +6.4pp(CI 重叠,仅提示性) |
  | **false_conflict** | **.494 [.421,.568]** | **.877 [.825,.916]** | **+38.3pp(CI 完全不重叠)** |

  三项里唯一统计确凿的是坏的那项。机制 = 预注册的担忧成立且远超预期:decoupled 的 Stage A 每题只命名**一个**目标类型,再套到全部 20 段;孤立探针里只有 needle + cf 两段所以无害,真实池里其余 18 条干扰段被**逼着**吐出该类型的某个值 → 含 gold 的段与 needle 抽出不同答案 → **gold 碎裂率 88%**。而 false_conflict 正是 E2 −4.8pp recall 的机制通道,所以接进门大概率让 recall 更差。**不采用。** S5 为真但不外推到两段以上;孤立探针高估了它,**池结构**才是破点。限制:两臂 denominator 不同(185 vs 237)→ 只用各臂 Wilson CI,未做配对检验。

**这条链就是主线本身:**门给出机制收益 → 代价被归因到底层而非规则 → 归因指出的两条杠杆各自被实测裁定(lenient 等价成立,更强抽取不成立)。诊断本身还自我印证了两次:连我们自己的探针计分都栽在同一个 exact-string 坑里(S5),以及自己预注册的担忧被自己的数据坐实(S6)。两条被推翻的自家结论(S4、S5)全部以实测更正 —— rigor 是这条主线的底线,不是修辞。

**最终计分:**唯一确立的系统级修复 = lenient 聚类(S3);8B × decoupled 在孤立探针漂亮但不级联(S6),不采用;8B 本身平反一半(前三次否决是 exact 计分伪影,但它在真实池里也没变成可用修复);顽疾 twin missed-conflict 仍在(池内 .31–.43),且**已证明抽取层修不动**。

---

## 5. Graph 2.0:为什么要做、1.0 做了什么、什么没解决

### 5.1 先分清两个"1.0",别混

- **ML Evidence Selector v1**(队友的学习型选择器,LightGBM LambdaRank + 特征):排序质量与正确证据保留率显著提升,但 **harmful-in-context 未降**;72% 特征重要性塌回 relevance/rank;ContractNLI 迁移 −0.134。它是 Graph 2.0 原计划里主张 C1 的对照基线,也是我们门的承诺 1(踢人路径不许有绝对判断、不做学习型门)的直接来源。
- **我们的门 v1 = Graph 1.0 的确定性最小实现**,S1–S6 跑的就是它。用图的词汇重述一遍就能看出 2.0 要换的是哪一层:

| 图元素 | Graph 1.0(现在,全确定性) | 缺口 |
|---|---|---|
| claim 节点 | 答案簇(exact 或 lenient 字符串等价聚类) | 每段只能吐一个答案值,**无法表达"本段与被问属性无关"** |
| conflict 边 | 跨簇答案不同即矛盾(单答案假设下的确定性代理) | 无语义判断;字符串不等 ≠ 语义矛盾 → false_conflict ≈ .49 |
| support 边 | 无显式 support 边,只用簇内成员数当代理 | CLAIM_SUPPORTS 缺席;E1-support 边准确率至今无法测(缺标注) |
| duplicate 边 | 同 `document_id` 合一票 | = SAME_SOURCE 的最小实现;不做跨文档同父来源(`source_parent_id`) |
| 聚合 / 决策 | `independent_support` 整数票 + 冻结的四条件 | 边没有 confidence,系统没有"弃权"这个动作 |

也就是说:**Graph 1.0 已经是一张图了,只是每条边都由字符串规则确定性生成,而且没有"不建边"这个选项。**

### 5.2 1.0 解决了什么(已确立,别自我贬低)

- 机制成立:harm −11.2pp,p≈0(S1)。免校准、孤源结构性安全、失效方向是闭嘴。
- 代价可归因:不是模型能力、不是检索、不是截断,是答案等价(S2)。
- 一个确立的修复:lenient 等价,in-pool recall +1.2pp p≈0、零 harm 代价、零推理成本,已在生产路径(S3)。

### 5.3 1.0 没解决的三件事(= 2.0 的立项理由)

1. **false_conflict ≈ .49,而且是结构性的。** 字符串不等就判矛盾,于是"同意但措辞不同"被判成分歧 → gold 碎裂 → 门误踢 → E2 的 −4.8pp recall 就是从这条通道流走的。lenient 只回收了 25%,剩下的是同义词、缩写、句法变体 —— 确定性规则**原理上**吃不到。
2. **missed_conflict .31–.43(孪生崩塌),且已证明抽取层修不动。** S4 判"prompt 修不了"(后被部分推翻),S5 显示 8B + decoupled 在两段隔离下能腰斩,S6 证明那个收益在 20 段真实池里是靠制造假冲突换来的(false .49 → .88)。**这条路走到头了。**
3. **单答案抽取无法弃权 —— 这是 1 和 2 的共同根因。** 每段都被逼着吐一个值,没有 UNKNOWN,任何"这段跟被问属性无关"的段都会被强行变成一个竞争答案。S6 的机制就是这一条被放大 18 倍。

### 5.4 为什么"关系层"正对这个失效模式

Graph 2.0 把"边"从字符串规则换成关系预测,而且 **UNKNOWN / 不建边是一等输出**(TRAINING_PLAN §3.2):无法确定时必须弃权,不许强制建边;每条预测边保存 confidence、模型版本、source ID 与输入文本 hash。主实验只冻结三类有官方或确定性监督来源的关系:`CLAIM_SUPPORTS`、`CLAIM_REFUTES`、`SAME_SOURCE`(后者由 `source_parent_id` 确定性生成)。关系模型基座固定 `cross-encoder/nli-deberta-v3-base`,监督来自 ContractNLI official 与去污染后的 VitaminC。

对应到 §5.3 的三个缺口:

- 缺口 3(不能弃权)→ 直接由 UNKNOWN 承接;Gate 0B 把 abstention rate 与 edge coverage 列为验收项。
- 缺口 1(false_conflict)→ 语义关系替代"字符串不等";"同意但措辞不同"应判 SUPPORTS 而非 REFUTES。
- 缺口 2(missed_conflict)→ 孪生的 gold 与注入 replacement 在语义上是真矛盾,REFUTES 边应当抓到;Gate 0B 要求 CLAIM_REFUTES precision ≥ 0.85。

**S6 的价值在于:它把 Graph 2.0 从"NLI 应该更聪明"这种愿望,变成一个可证伪的必达目标 —— 在不推高 false_conflict 的前提下压低 missed_conflict。** 靶子具体,而且衡量它的 harness 已经现成(`cluster_eval` 的三项 + Wilson CI)。这是用实测负结果支撑关系层,不是用希望支撑。

### 5.5 2.0 靠什么防止重犯 1.0 的错(协议已冻结的部分)

- **零新增人工标注:** primary label 只能是 official label / 可复现 deterministic rule / 带 mutation log + seed + hash 的 synthetic provenance。LLM teacher 或 judge 只能做特征、故障诊断或 secondary proxy,不得当 primary label 或过 Gate 的唯一依据(依据:judge-kappa 0.005–0.08)。
- **关系模型必须先过 Gate 0B 才能靠近选择器:** ContractNLI 与 VitaminC 各自 official test 各跑一次且**不许 pooled**;CLAIM_REFUTES precision ≥ .85、support/refute macro-F1 ≥ .80、non-UNKNOWN coverage ≥ .80(SUPPORT / REFUTES 各自 ≥ .70)、SAME_SOURCE 规则单测 100%;外加 metamorphic 测试(同一来源重复不得增加 independent support;打乱边后图收益必须消失)。
- **负控必跑:** NLI-without-Graph(排除"收益只是多了个 NLI 模型/更多参数")、graph-only、shuffled-graph。
- **全链路 OOF:** claim 抽取、聚类、关系预测、graph 特征都按父页面 + synthetic family 做 5-fold,不是只对最后一层关系分做 OOF。
- **主结论一次性:** 只在 fresh sealed 600 上跑,一次运行全部冻结系统;三 seed(13/42/73)预先固定成 primary ensemble,fresh test 不许挑 seed。
- **联合判定(C1):** ΔHarmful@10 ≤ −0.02 且双侧 95% CI 上界 < 0,Required Recall@10 与 NDCG@10 的单侧非劣下界 ≥ −0.01。三项必须同时满足,不能用次级指标的改善替代失败的主指标。
- **七条立即停止条件**(TRAINING_PLAN §9),包括"必须读 dev/test gold 才能建图"与"shuffled graph 复现全部收益"。

### 5.6 一处必须先拍板的内部张力

`TRAINING_PLAN.md` 是从旧项目迁移的草稿(文件头写明"待重新设计,不得直接开始训练")。它的 C1 是 **Graph + LightGBM LambdaRank v2 vs 学习型 v1** —— 也就是把踢留决策交回学习器,这与我们门的承诺 1(踢人路径上不许有绝对判断、不做学习型门)**直接冲突**。两种收法:

- **A(与现有承诺一致,倾向这条):** Graph 2.0 只升级**边与票的构造**(NLI 关系 + UNKNOWN + `source_parent_id`),门的四条件结构与整数票判据**不动**。NLI 只进"边怎么建",不进"踢留的刻度" —— 这正是承诺 5 信号毕业制的形式:关系边先过 Gate 0B 才允许进门。主对照回到同池配对(Graph 1.0 vs 2.0,gate-on/off),沿用现成的 E1/E2 harness。
- **B(按 TRAINING_PLAN 原样):** 回到学习型 ranker + graph 特征,需要重训 v1 作对照、重批数据协议,并重新论证为什么 V1 的 harm 失败这次不会重演。成本高,且与 S1–S6 这条叙事线断开。

这个选择是 M0 的第一件事,必须写进协议冻结文档。

### 5.7 M0 是什么(下一步的唯一交付)

M0 是**预注册里程碑,不写 NLI 代码**。冻结四样东西:协议与三关系 schema;训练规模 2000 vs 500 的决定(现有 artifact 只有 500 个 train query,必须在看 dev 结果**之前**定);fresh sealed 600 的 manifest 与泄漏审计(query / 父页面 / answer entity / passage hash / synthetic family 五轴零重叠);power 与 MDE 计算(公式 + 输入参数 + 可复现脚本,并把 S6 的 false_conflict 护栏折进去作为新增 gate)。退出判据:provenance / hash 审计零违规。

---

## 6. 未闭合项(必须讲)

1. **双 Gate 仍 FAIL。** lenient 之后 Required Recall 相对 gate-off 仍 −3.6pp,守卫是 −0.01。这是 selector 能否给出正面系统结论的唯一卡点,而抽取层已经证明修不动它 → 只能指望关系层。
2. **顽疾 twin missed-conflict 仍在**(池内 .31–.43),已排除截断、检索、prompt、模型容量、抽取结构五种解释。
3. **false_conflict 从未被直接优化过。** 它是 E2 recall 代价的机制通道,S6 又证明它极易被推高;Graph 2.0 必须把它当一等护栏,而不是只盯 missed。
4. **8B + decoupled 不进生产。** 它只活在评测 harness(`cluster_eval_cli` / `missed_conflict_probe_cli`),`selector/extraction.py` 保持单段 `EXTRACT_PROMPT` —— 这是 S6 之后的**决定**,不是待办。
5. **E3(coverage on/off)与 E1-support(support 边准确率)仍 BLOCKED**,分别缺"互补压力数据"与"official supporting-fact 标注"。后者恰好是 Graph 2.0 的 CLAIM_SUPPORTS 边要补的那一块。
6. **下一步 = Graph 2.0 M0**(预注册,无 NLI 代码),先解决 §5.6 的收法选择。
