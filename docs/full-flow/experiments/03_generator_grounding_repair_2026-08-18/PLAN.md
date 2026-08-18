# Generator 证据使用修复与 Selector 分阶段协同执行计划

**路线：** `03_generator_grounding_repair_2026-08-18`
**日期：** 2026-08-18
**修订状态：** `G212 FAIL / LENGTH / CONTROLLED LENGTH REPAIR NEXT / NO TRAINING STARTED`
**修订原因：** 明确旧 Selector 的数据与外推边界；把模块资格、强统计结论和完整系统资格分开；将 Generator-aware Selector 纳入同一条交替冻结路线
**上一阶段：** G230 `COMPLETE / NO CANDIDATE`
**主生成模型：** `ibm-granite/granite-4.1-3b@c0650403...`
**执行 tracker：** [TRACKER.md](TRACKER.md)
**原始方案快照：** [snapshots/PLAN_v1_generator_only_2026-08-18.md](snapshots/PLAN_v1_generator_only_2026-08-18.md)

---

## 1. 一句话目标

不同时自由修改 Selector 和 Generator，而是按“Generator 修复并冻结 -> 用冻结 Generator 产生证据效用标签 -> 训练并冻结 Utility Selector -> 完整三模块验证”的顺序，让 Retriever 找得全、Selector 对当前 Generator 筛得有效、Generator 对筛选后的证据用得可靠，最后只依据冻结系统在未见数据上的结果决定能够支持哪些研究结论。

---

## 2. 当前实际情况

### 2.1 正式系统结构没有改变

正式运行仍然只生成一次答案：

```text
Retriever -> Selector -> Generator -> one answer
```

实验中的 TopK、Legacy Selector、Utility Selector 和不同 Generator 是平行对照臂，不是部署时对同一问题先后回答多次。

### 2.2 Retriever 当前状态

- 当前 Hybrid RRF/default Retriever 和索引保持冻结；
- 本路线不重新训练 Retriever，也不把扩大候选池误写成 Generator 可见证据增加；
- 联合实验仍报告 support visibility；支持证据不在 Generator 可见上下文时，错误归 Retriever；
- Retriever 的最终跨数据表现仍由完整系统 held-out 单独报告。

### 2.3 旧 Lean Selector 的真实训练范围

旧 Lean Seed-13 Selector 不是只用一个数据集训练：

| 数据 | train-fit 问题 | active evidence rows | 实际监督 |
|---|---:|---:|---|
| NIAH | 920 | 5,312 | protect + synthetic-counterfactual harm |
| 2Wiki | 2,700 | 5,311 | protect only |

模型为 `cross-encoder/nli-deberta-v3-base` 的 NLI-aware 双头：

- `protect`：预测证据是否应该保留；
- `harm`：预测证据是否具有反事实/冲突风险；
- runtime 只在高 harm、低 protect 时删除 0、1 或最多 2 条。

必须保留的限制：

1. `harm` 的有效正监督来自 NIAH synthetic counterfactual，不来自 2Wiki；
2. 2Wiki 主要训练和验证“不要破坏 supporting evidence / multi-hop chain”；
3. L003 在 1,000 个 2Wiki 问题上实际删除 0 条，因此没有测到 2Wiki 上的有害过滤效用；
4. F005 sealed600 也是 NIAH synthetic-counterfactual 体系内的独立集合；
5. F005 的 74/82 和 90.24% deletion precision 只证明该 NIAH 风险代理上的高精度删除，不能外推成 HotpotQA、MuSiQue-Full、RGB 上的通用有害证据识别。

因此旧 Lean Selector 的正式角色是：

```text
Legacy safety/risk baseline
不是最终 Selector
不是跨数据通用 harmful filter
```

事实来源：[L002 训练与开发报告](../../../selector/experiments/08_r005ab_repair_current_2026-08-12/L002_DEVELOPMENT_REPORT.md)、[L003 最终盲测报告](../../../selector/experiments/08_r005ab_repair_current_2026-08-12/L003_FINAL_REPORT.md)、[F005 独立结果](../01_selector_generator_bridge_2026-08-12/F005_RESULTS.md)。

### 2.4 Generator 当前状态

G230 已经证明：

- GC/GM 的 draft empty 为 0；
- final empty 从 G0 的 12.04% 降到约 3.79%–6.09%；
- GM answer match 相对 G0 的点估计提高约 3.52–6.36pp；
- GM 对 support+harmful 的改善在三个 seed 上方向一致。

但同时：

- mixed-context 原冻结门没有全部通过；
- GC/GM 的 MiniCheck citation precision/recall 相对 G0 下降约 6–8pp；
- 因此当前没有通过可靠性要求的新 Generator；
- G0 仍是可运行、引用相对可靠的基线。

事实来源：[G230 Generator 开发门结果](../02_generator_selector_alignment_2026-08-15/G230_RESULTS.md)。

### 2.5 当前组件判定

| 组件 | 可运行基线 | 已确定的新方法 | 当前用途 |
|---|---|---|---|
| Retriever | frozen Hybrid RRF/default | 不在本轮改变 | 真实候选证据入口 |
| Selector | TopK keep-all、Legacy Lean | 没有最终 Selector | Lean 只作 safety/risk baseline |
| Generator | G0 | 没有新 `G*` | G0 是可靠性基线和必要时的 fallback teacher |
| 完整系统 | TopK + G0 | 没有 `SystemF` | 所有新方法的共同基线 |

---

## 3. 核心方法：有联系地分阶段训练

Selector 和 Generator 不能完全独立，因为一条证据是否“有用”取决于当前 Generator 怎样使用它；但也不能同时自由更新，因为那会产生移动训练目标。

本路线只允许一个受控协同循环：

```text
阶段 G：修复 Generator，使其适应 TopK、删减、噪声和多证据上下文
    -> 冻结一个教师 Generator GQ

阶段 S：固定 GQ，测量删除每条证据对答案与引用的真实影响
    -> 训练 Utility 候选 SU，并冻结一个职责合格 Selector SQ

阶段 I：固定 Retriever + GQ + SQ
    -> 完整系统开发验证并冻结 SystemF

阶段 H：SystemF 一次性运行 HotpotQA / MuSiQue-Full / RGB
    -> 只决定主张强度，不再修改方法
```

重要纪律：

- Generator 训练可以看到 Legacy Selector 形成的上下文，但不把 Legacy Selector 当最终方法；
- Selector utility labels 必须由已经冻结的 GQ 生成；
- utility labels 生成后不得修改 GQ；
- 如果以后更换 Generator，旧 utility labels 自动失效，必须重新生成；
- 本路线不允许反复 G -> S -> G -> S 循环调参。

术语固定为：

- `SL`：旧 Legacy Lean safety/risk Selector；
- `SU`：legacy safety + Generator utility 的新训练候选；
- `SQ`：最终通过模块职责门的 selected qualified Selector，主目标是 `SQ=SU`；
- 只有 SU 失败而 SL 在同一 GQ 下确有正的端到端作用时，才允许 `SQ=SL` 作为系统 fallback；此时 C2 的 Utility Selector 主张明确失败。

---

## 4. 三层资格与结论

### 4.1 模块职责资格

回答的问题是：

> 这个组件是否足够稳定，可以进入下一阶段协同实验？

它使用主要指标点估计、预冻结的实际容忍范围、跨 seed 方向和严重退化保护。它不要求每个指标和每个 slice 的 95% CI 都显著优于基线。

### 4.2 强统计结论

回答的问题是：

> 我们是否已有足够统计证据声称这个组件本身优于基线？

只有预注册主要指标的 CI 支持时才写“统计可信提升”。如果 CI 跨 0，只能写“模块职责通过、观察到正向点估计，但独立优越性证据不足”。

### 4.3 完整系统资格

回答的问题是：

> 冻结的 Retriever + Selector + Generator 是否值得进入一次性 system held-out？

这里要求完整系统主要联合结果正向、Selector 对同一个 GQ 有净作用、答案与引用没有实际不可接受的退化。最终 held-out 负责检验跨数据泛化，而不是继续选方法。

### 4.4 数据冻结门与路线继续条件

数据冻结门回答的问题不是：

> 这个研究方向有没有意义？

而是：

> 当前这批训练/验证 target 是否足够干净、足够大，可以作为正式 Generator 训练和配方选择数据？

G210 的 `2Wiki model-val >=100` 是预先冻结的验证稳定性保护：2Wiki 负责多证据链、跨 relation citation 和 Generator 配方 maximin 选择；model-val 太小会让配方选择、错误分层和 chain/citation 退化判断过不稳定。它不是自然科学常数，也不是说 99 和 100 有本质差异；它是为了防止在小样本上把偶然点估计当成方法成功。

因此：

- 已观察到 `76/100` 后，不能把原门改成 `>=76` 并宣布 G210 通过；这会变成按结果改门；
- `76/100` 也不证明路线失败，只证明当前 target construction/audit 不能直接冻结；
- 如果失败同时满足积极信号条件，路线应进入受控数据修订，而不是结束或直接训练。

积极信号条件是：

1. structural audit 全部通过；
2. train/model-val split group overlap 和 component overlap 仍为 0；
3. train answerable groups 仍足够支撑训练；
4. unsupported update ratio 仍在预算内；
5. 失败集中在可解释、可修订的 target construction/audit mismatch，而不是随机大面积污染；
6. 没有读取 sealed600、system held-out、official dev 或任何禁止数据；
7. 没有启动训练或 utility labels。

满足这些条件时，允许新增一次数据修订阶段；不满足时才停止路线或降级结论。继续推进的含义是重做 G200/G210，而不是在失败数据上启动 G300。

---

## 5. 研究主张

### C1：Generator 职责主张

经过 claim-grounded context training 的 Granite draft Generator，可以在不重现 G230 明显 citation 退化的前提下，提高“答案正确且事实引用受支持”的联合结果，并保持可接受的 coverage 与 unsupported behavior。

### C2：Selector 职责主张

在冻结 GQ 下，由实际答案、coverage 和 citation 变化构造的 utility labels，比只学习 NIAH harmful/protect 的 Legacy Selector 更接近“哪些证据对当前 Generator 真正有用”。

### C3：完整系统主张

Utility Selector 与其冻结教师 Generator 组合后，相对 TopK + G0 提高完整系统的 `correct_and_cited`，并且 Selector 相对同一个 GQ 提供正的净作用。

### 主张降级规则

- GQ 通过职责门但主要 CI 跨 0：可进入 Selector 阶段，不声称 Generator 已统计显著优于 G0；
- SQ 证据标签很好但同一 GQ 下端到端不提高：不声称 Selector 有系统效用；
- 完整系统开发点估计通过但 held-out 不稳定：方法和结果照常报告，跨数据主张降级；
- 每个数据集单独报告，不用一个数据集的提升抵消另一个数据集的明显退化。

---

## 6. 不破坏既有工作的边界

### 6.1 保持冻结

- Retriever 当前配置、index、corpus identity 和模块结论；
- Granite 4.1-3B base revision、tokenizer 和基础 draft prompt contract；
- TRUE checkpoint及其 runtime verifier 角色；
- MiniCheck 只作生成后的独立 citation judge；
- G0、GC、GM checkpoint、manifest 和 G230 历史结论；
- Legacy Lean checkpoint、F005 threshold `0.9212157130241394`、cap=2 和历史结论；
- sealed600 retired 状态；
- HotpotQA、MuSiQue-Full、RGB 预抽样 IDs 与 hash。

### 6.2 runtime 与 gold 边界

- Retriever 和 Selector runtime 不读取 gold/reference；
- Generator runtime 不读取 gold/reference；
- gold、provenance 和 official supporting facts只用于离线训练 target、utility label 构造和生成完成后的评分；
- 每个 manifest 必须记录 `gold_loaded_at_runtime=false` 与 `reference_answers_loaded_at_runtime=false`。

### 6.3 组件修改边界

- 阶段 G 默认只训练 draft LoRA；
- claim splitter 或 routing 只有在 G100/G110 归因证明其为主要断点时才做一次共享的条件修复；
- TRUE 本轮不微调；
- 阶段 G 不训练 Selector；
- 阶段 S 不修改 GQ；
- 阶段 I 不修改任何组件。

---

## 7. 数据集角色

| 数据 | 阶段 G | 阶段 S | 阶段 I / H | 禁止 |
|---|---|---|---|---|
| NIAH train / G200 train-fit | answer/citation、噪声、位置、unsupported 训练 | leave-one-out utility、legacy safety | 不作最终结论 | 当作跨数据证明 |
| NIAH train 新 model-val | GR-F/GR-C 选择 | utility 配方选择 | 不作最终结论 | 训练或最终结论 |
| G200 已揭示 model-val 62 | 历史回归报告 | 不选方法 | 不作资格 | tie-break |
| NIAH decision-dev 739 | GQ locked qualification | SQ locked qualification | 完整系统开发 | 反复调参、最终 superiority |
| 2Wiki train | 多证据链 Generator 训练/model-val | multi-hop utility 与 MUST_KEEP | 不作最终结论 | 估计 NIAH harmful 主效应 |
| 2Wiki dev 2,000 | GQ 跨数据资格 | SQ 跨数据资格 | 完整系统开发 | 训练、反复调参 |
| ASQA/QAMPARI 已揭示数据 | citation catastrophe guard | 不作主训练 | regression report | 新的独立结论 |
| F005 sealed600 | 不使用 | 不使用 | 只读历史 | 任何训练、筛选、再测试 |
| HotpotQA 400 | 不读取逐题内容/结果 | 不读取 | SystemF 一次性最终测试 | 方法选择 |
| MuSiQue-Full 400 对 | 不读取逐题内容/结果 | 不读取 | SystemF 一次性最终测试 | 方法选择 |
| RGB 300 + cf 100 | 不读取逐题内容/结果 | 不读取 | SystemF 一次性最终测试 | 方法选择 |
| SciFact/NQ | 无 | 无 | Retriever 历史基准 | 本轮 G/S 训练 |

### 7.1 split 和去泄漏

1. 切分单位是 question group、provenance component 和 relevant parent page；
2. 同一问题的所有 context variants 和 leave-one-out variants 位于同一大 split；
3. 新 NIAH model-val 与旧 G200 全部 1,023 个 role-assigned queries/components 不重合；
4. 2Wiki official train 内固定 train/model-val；official dev 不参与；
5. decision-dev、sealed600 和 system held-out IDs/components/parents 进入 denylist；
6. 先冻结 source manifest、ordered IDs 和 SHA256，再生成 target 或运行模型；
7. 不按模型效果删题、补题或重抽。

### 7.2 为什么仍使用 NIAH 和 2Wiki

- NIAH 提供可控 support、benign、synthetic harmful 和位置变化，是学习抗干扰和观察 Selector 动作的受控数据；
- 2Wiki 提供多文档、多跳支持链，使 Generator 和新 Selector 不只适应 NIAH 的单 carrier 模板；
- 2Wiki 的 distractor 是否值得删除由冻结 GQ 的 utility 结果决定，不伪造 `harm` gold label；
- 最终三个数据集不参与开发，它们检验这些能力是否真正泛化。

---

## 8. 阶段 G：Generator 证据使用修复

### G000：协议和统计范围冻结

冻结：

- Git commit、G230 archive、模型/prompt/TRUE/MiniCheck revision；
- data IDs、denylist、held-out hash；
- GR-F/GR-C 配方、计算预算和选择规则；
- 模块职责门、强结论门和严重退化 tripwire；
- cluster unit、bootstrap seed 和报告模板。

G010 使用 G230 paired discordance 做 simulation-based MDE/sensitivity，不使用 observed power。它只说明样本能识别多大效应，不把无法识别的小效应写成等价或无效。

### G100/G110：citation 断点归因

对 G0/GC/GM 共同 answered rows 重建：

```text
draft claim
-> declared citation
-> split claim
-> TRUE route
-> final attached evidence
-> MiniCheck judgment
```

每个 regression row 归入最早失败阶段：

1. `UNSUPPORTED_DRAFT_CLAIM`
2. `DRAFT_CITATION_MISSING_OR_WRONG`
3. `SPLITTER_BOUNDARY_OR_REWRITE`
4. `TRUE_ROUTING_OR_ATTACHMENT`
5. `EVALUATOR_DISAGREEMENT`
6. `NO_REGRESSION`

若 regression rows <=120 则全部人工复核；否则固定分层抽取 120 条，每个非空关键 stratum 至少 20 条。至少 20% 由两名复核者盲于实验臂独立判断并裁决分歧。

默认进入 draft repair。只有 downstream 两类数量明确更多并经人工复核支持，才激活一次 G120 splitter 或 G130 routing 确定性修复；该修复必须由 G0、GR-F、GR-C 共享。

G110 已完成独立审计：120 条主审、24 条双审，双审一致率 91.67%。最终标签分布中 `TRUE_ROUTING_OR_ATTACHMENT` 为 44/120，高于 `EVALUATOR_DISAGREEMENT` 37/120、`SPLITTER_BOUNDARY_OR_REWRITE` 22/120 和 draft/unsupported 合计 17/120，因此条件激活 G130 routing/attachment repair。该判定只允许一次共享 runtime 修复实现，不允许启动训练、改训练门或读取 held-out。

G130 已完成确定性 runtime 修复：TRUE routing hypothesis 改为最终展示给用户的 citation-stripped sentence；trace 与 G230 routing export 显式记录 `routing_hypothesis`、`declared_verified`、`rescued_by_scan`、`review_flagged` 和 `attachment_verified`；observe-only entity gate warning 保持非破坏性，不再被解释为 final attachment failure。TRUE 模型/阈值、Retriever、Selector、gold/reference runtime 边界、held-out 和训练均未改变。

### G200/G210：训练数据

监督文本不得由 G0/GC/GM/GR 候选自己生成后再当正确标签。

- NIAH：优先复用 G200 已审计 515 train-fit targets；新 model-val 在旧 1,023 题之外用同一 frozen QA2D revision 和审计规则构造；
- 2Wiki：只使用 official answer、`evidences`、`supporting_facts` 和 context；evidence triples 用固定模板形成按 chain order 排列的原子事实；
- unsupported：从 train split 内移除完成答案必需的 support，target 固定为 `I don't know.`；
- 每个原子事实最终只绑定一条 verified evidence，符合当前 runtime `ClaimRouting.citation`；
- 多跳答案写成多条原子事实组成的证据链，不强行让一个复合 claim 同时绑定多条最终 citation。

最低数据门：

- NIAH >=400 answerable train groups，另有 >=100 全新 model-val groups；
- 2Wiki >=400 answerable train groups，另有 >=100 model-val groups；
- unsupported 占 optimizer updates 的 10%–15%；
- 任一 answerable 数据源不超过 answerable updates 的 55%；
- 0 split leakage、0 invalid citation、0 truncation。

每个 target 通过 answer alias、citation remap、TRUE entailment、minimal support、unsupported support-absence 和分层人工审计。审计不通过就排除并记录，不能降低门凑数量。

G200 已完成数据预物化：NIAH train 515 groups、NIAH 新 model-val 307 groups、2Wiki train 1,075 groups、2Wiki model-val 136 groups、unsupported 1,075 updates，占 optimizer updates 10.1703%；split group/component overlap 均为 0。G200 只达到 `PRE_AUDIT PASS`，TRUE、minimal support、citation remap、unsupported support-absence、人工样本和长度/truncation 审计仍属于 G210/G300；G210 通过前不得训练。

G210 已执行 structural audit 和 TRUE audit。结构性检查 3,108/3,108 通过；TRUE worklist 2,758 rows 中 2,140 entailed、618 not entailed。剔除 TRUE 不通过 case 后，NIAH train/model-val 为 515/215，2Wiki train/model-val 为 683/76，unsupported update ratio 为 12.4855%，split overlap 仍为 0。由于 2Wiki model-val answerable groups 只有 76，低于最低门 100，G210 hard data gate failed；G300 不得启动，人工样本审计也未进入。

G215 已完成门槛解释与数据修订授权。G210 失败结论不变，但该失败不再被解释为路线无意义：结构、泄漏、训练规模、unsupported 比例和禁止数据边界均给出积极信号，失败主要集中在 2Wiki `official triple -> templated atomic fact -> TRUE entailment` 的 relation 模板/审计匹配。因此当前路线恢复为“受控数据修订继续”，下一步必须回到 G200R/G210R，不能从 G300 继续。

### G215/G200R/G210R：数据修订恢复路径

G215 只修改计划解释和恢复路径，不改任何 G210 结果、不启动训练、不降低 TRUE 阈值。

允许的修订范围：

- 只针对 G210 triage 指出的 2Wiki relation target construction/audit mismatch；
- 在重新运行 TRUE 前预注册 relation 模板、support-sentence 对齐规则、yes/no target 规则和 candidate ordering；
- 可以扩大 2Wiki official train 内的预审计候选池，但必须仍保持 train/model-val component isolation；
- 可以让 2Wiki target 更贴近 support sentence 表达，但每个可训练 atomic target 仍必须有可审计 citation 绑定；
- 重新执行 G200R materialization、G210R structural/TRUE/manual/length audit、manifest、ordered IDs 和 SHA256；
- 旧 G210 76 个通过 case 只作为历史诊断，不自动并入冻结数据。

禁止的修订范围：

- 不把 G210 的最低门从 100 降到 76；
- 不改变 TRUE checkpoint、TRUE threshold 或 judge 角色来凑通过；
- 不读取 sealed600、HotpotQA、MuSiQue-Full、RGB 或 2Wiki official dev 来选方法；
- 不按 TRUE 结果临时挑题补足数量；
- 不在 G210 failure 数据上训练 Generator、生成 utility labels 或推进 S/I/H。

G200R/G210R 通过后，才能进入 G300；如果 G210R 仍无法形成足够 2Wiki model-val，则记录数据修订失败，并由用户另行决定是否只做探索性 pilot 或降级为 NIAH-focused Generator study。

G215 报告见 [G215_GATE_AND_DATA_REVISION_AMENDMENT.md](G215_GATE_AND_DATA_REVISION_AMENDMENT.md)。

G200R 已完成 revised data materialization：保持 NIAH 515/307、2Wiki 1,075/136、unsupported 1,075 和 split overlap=0；2Wiki target construction 改为 `support_sentence_aligned_v1`，runtime train/validation case SHA256 分别为 `239f274b362ac8467c865a6f16eeb14b032a37ea539ea465e636ea7aa11d8db8` 与 `24e1a618a7a4c667bfcd1392d767033e107894285a09eb4e52d9efad021b9e9f`。G200R 只达到 `PRE-AUDIT PASS`，TRUE/manual/length 审计仍属于 G210R；报告见 [G200R_DATA_MATERIALIZATION_REPORT.md](G200R_DATA_MATERIALIZATION_REPORT.md)。

G210R-v1 structural audit 已失败：25 个 2Wiki answerable case 未通过 answer alias preservation，原因均为 `literal_answer_missing`；TRUE audit、finalize、manual audit 和训练均未启动。该失败说明 G200R2 必须在物化时过滤 answer alias 不保留的 support-sentence targets；报告见 [G210R_STRUCTURAL_FAILURE_REPORT.md](G210R_STRUCTURAL_FAILURE_REPORT.md)。

G200R2 已完成 answer-alias-preserving materialization：NIAH train/model-val 为 515/307，2Wiki train/model-val 为 1,053/133，unsupported groups 为 1,053，unsupported update ratio 为 10.0881%，split overlap=0。G200R2 只达到 `PRE-AUDIT PASS`，必须进入 G210R2 审计；报告见 [G200R2_DATA_MATERIALIZATION_REPORT.md](G200R2_DATA_MATERIALIZATION_REPORT.md)。

G210R2 已完成 structural audit、TRUE audit 和 pre-manual finalize：structural 3,061 cases 全部通过，TRUE worklist 2,708 rows 中 2,338 entailed、370 not entailed。剔除 TRUE 不通过 case 后，NIAH train/model-val 为 515/215，2Wiki train/model-val 为 828/106，unsupported groups 为 1,053，unsupported update ratio 为 11.3068%，split overlap=0；自动数据门均通过，包括 2Wiki model-val `106 >= 100`。这证明 G215 授权的数据修订有积极信号，但 G210R2 manifest 仍明确 `manual_audit_required_before_training=true`，所以数据尚未正式冻结，G300 仍 blocked。下一步为 G212 manual/length audit；报告见 [G210R2_TARGET_AUDIT_REPORT.md](G210R2_TARGET_AUDIT_REPORT.md)。

G212 已完成训练前 length/manual packet prepare，但长度审计失败：全量 11,348 个 examples 中有 4 个超过冻结 `max_length=2304`，全部来自同一个 2Wiki train answerable case `2wiki::b779ecdc08c411ebbd8eac1f6bf848b6` 的 4 个长 context variants；validation、NIAH 和 unsupported examples 均无超长。manual sample 已按 5 个 stratum 各抽 20 条，但因 length gate 失败未进入 reviewer 判定。G212 不能解锁 G300；下一步只能执行 G214 controlled length repair，成组排除该 overlength train group 及其 unsupported counterpart 后重跑 G212R。报告见 [G212_LENGTH_AUDIT_REPORT.md](G212_LENGTH_AUDIT_REPORT.md)。

G214 修订边界：

- 只允许排除 `group_id=b779ecdc08c411ebbd8eac1f6bf848b6` 的 2 个 train cases：answerable case 与对应 unsupported case；
- 不允许提高 `max_length=2304`；
- 不允许改变 TRUE checkpoint、TRUE threshold、manual sample 规则或 held-out/dev 边界；
- 修订后必须重新写 train/validation cases、manifest、ordered IDs、SHA256，并重跑 G212R length/manual audit；
- 如果 G212R 仍有超长或 manual review 不通过，G300 继续 blocked。

### G300：训练实现

固定：

- Base：IBM Granite 4.1-3B；
- LoRA：`r=8`、`alpha=16`、`dropout=0.05`；
- adapter 只在 draft call 启用；
- splitter 默认 adapters disabled；
- TRUE frozen；
- greedy decode；
- loss 只作用于 assistant target。

token 权重：

```text
answer/punctuation = 1
citation brackets/index = 4
prompt = 0
```

每个 question group 的多个 context variants 合计等权，不能把 8 个变体当成 8 个独立问题。

### G310/G320：两个便宜配方选一个

| 配方 | 初始化 | 学习率 | 含义 |
|---|---|---:|---|
| GR-F | frozen Granite base 新 LoRA | 1e-4 | 从头联合学习 answer+citation |
| GR-C | GM13 adapter continuation | 5e-5 | 保留 GM answer/empty 能力并修复 grounding |

两者使用相同数据、updates、batch、model-val、splitter、TRUE 和 MiniCheck。只运行 seed13 screen。

screen 硬排除：

- runtime error、missing trace、invalid citation 不为 0；
- citation point regression >3pp；
- answer 或 coverage point regression >2pp；
- unsupported ungrounded assertion 增加 >2pp；
- final empty 明显高于 G0。

存活配方按 NIAH/2Wiki model-val 的 `correct_and_cited` delta 做 maximin 选择；并列选 GR-F。只允许一个配方进入 G330。

这只是落地配方选择，不是纯初始化因果消融；GR-C 的历史 updates 和总计算量必须报告。

### G330：三个 seed 正式训练

唯一配方训练 seeds 13/42/73：

- 每个 seed 只保存 final-completed checkpoint；
- 不按 qualification 选 epoch；
- persisted adapter 必须 fresh-base strict reload；
- 记录 answer-token/citation-token loss、model-val、fingerprint、显存和 wall time；
- 效果失败不能换 seed 重训。

### G400/G410/G420：Generator 模块资格

先定义：

```text
correct_and_cited =
  answer_match
  AND every factual sentence has a citation
  AND MiniCheck supports every sentence-citation pair
```

#### A. 技术完整性硬门

- runtime error = 0；
- missing trace = 0；
- invalid citation index = 0；
- runtime gold/reference loaded = false；
- splitter/TRUE frozen fingerprint 不变。

#### B. Generator 职责硬门

相对同输入 G0，三 seed family point estimate 必须满足：

- NIAH `correct_and_cited` delta >0；
- answer delta >= -2pp；
- coverage delta >= -2pp；
- citation precision 和 recall delta 分别 >= -3pp；
- unsupported ungrounded assertion delta <= +2pp；
- 至少 2/3 seeds 的 `correct_and_cited` 方向非负；
- 任一 seed 的 citation precision/recall 不得下降超过 5pp；
- 2Wiki `correct_and_cited` delta >= -2pp，answer/citation 任一项不得下降超过 3pp。

这些是模块可用性的实际容忍范围，不是显著性门。G010 在新结果出现前复核其可分辨性和业务含义，但不得根据候选表现放宽。

#### C. 严重退化 tripwire

- 任一主要 context stress 的 family `correct_and_cited` 下降超过 5pp；
- ASQA 或 QAMPARI citation precision/recall 下降超过 5pp；
- unsupported 无依据回答增加超过 5pp；
- 两个 seeds 在同一关键安全指标上为负且超过职责 margin。

stress slices 必须完整报告，但 support-only、benign、harmful、support-last 不再要求每个 slice、每个 seed 都严格优于 G0。

#### D. 强 Generator 结论

只有 NIAH 主要 `correct_and_cited` family 95% CI 下界 >0，且 answer/citation 的预注册非劣 CI 通过时，才声称新 Generator 本身得到统计可信提升。

强结论失败不自动否决职责合格的候选。

### G430：冻结教师 Generator

- 唯一新候选通过 A/B/C：冻结为 `GQ`，进入阶段 S；
- 通过职责门但 D 失败：仍可成为 `GQ`，报告“module-qualified / strong claim not established”；
- 新候选未通过职责门：记录 Generator repair `NO NEW CANDIDATE`，使用 G0 作为冻结教师 `GQ=G0`，阶段 S 仍可继续；
- 无论哪种情况，GQ 一旦用于 utility labels 就不能再修改。

这个 fallback 避免把“没有新 Generator”错误等同于“不能研究 Selector 对可靠基线的真实效用”。

---

## 9. 阶段 S：Generator-aware Utility Selector

### S100：冻结教师和 100 题 utility pilot

冻结 GQ 的：

- adapter/base identity；
- prompt/decode；
- splitter/TRUE；
- answer/citation scorer；
- runtime environment。

对 NIAH train 和 2Wiki train 的固定 100 个问题先运行：

```text
GQ(q, full context)
GQ(q, context without evidence e_i)
```

full 与 leave-one-out 平衡执行顺序。gold 只在全部生成后评分。

标签：

| 标签 | 离线含义 |
|---|---|
| MUST_KEEP | 删除后正确变错误、答案变空、citation/coverage 退化或必要支持链断裂 |
| SAFE_DROP | 删除后错误变正确；或删除 distractor 后答案/coverage/citation 不退化且无必要 support 损失 |
| NEUTRAL | 删除前后主要结果无实质变化，也无独立 harmful/support 标签 |
| UNCERTAIN | 重复方向不一致、judge 无法判定或上下文本身不可回答 |

所有对错转换至少重复一次；只有方向一致时进入 SAFE_DROP/MUST_KEEP。UNCERTAIN 默认保留或不进入 utility loss。

pilot 只检查：

- 稳定标签产出率；
- 重复一致率；
- NIAH/2Wiki、rank、support/distractor 覆盖；
- 估算完整 leave-one-out 计算预算。

标签过少或重复不稳定时，停止 Utility Selector 训练并报告原因，不强造训练集。

### S110：完整 utility 数据

pilot 通过后，按预冻结抽样扩展：

- NIAH 保留 legacy protect/harm、support/benign/synthetic-harmful 身份；
- 2Wiki 不伪造 harmful 标签，只使用 official support 与 GQ utility；
- 同一 query 的 full/leave-one-out 和重复 rows 作为一个 group；
- train/model-val 按 component/parent 隔离；
- decision-dev、sealed600、system held-out denylist 生效；
- 保存每次 GQ 输出、评分、label reason 和 hash。

### S200：Utility Selector 实现

保持 Selector 独立模块身份：

- 基础 encoder 沿用 NLI-aware DeBERTa；
- Legacy protect/harm 作为 safety guard；
- 新增或训练 utility 目标预测 MUST_KEEP / SAFE_DROP / NEUTRAL；
- runtime 只读取 question、candidate evidence 和允许的候选列表特征；
- 只有 safety guard 允许且 utility 高置信 SAFE_DROP 时才能删除；
- MUST_KEEP、NEUTRAL、UNCERTAIN 或缺分数全部保留；
- 第一轮 cap 仍为 2，不同时搜索 cap、阈值和新架构。

比较：

| 组 | 含义 |
|---|---|
| S0 | keep-all TopK |
| SL | frozen Legacy harmful/protect Selector |
| SU | legacy safety + GQ utility 主候选 |
| SU-no-safety | 仅作机制消融，不允许成为默认部署 |

### S210/S220：选择和三 seed

- 只用 train/model-val 做 seed13 utility threshold/recipe screen；
- 选择规则优先 MUST_KEEP safety，再看 SAFE_DROP precision 和同一 GQ 下的 `correct_and_cited`；
- 只冻结一个 SU 配方；
- 正式训练/复现 seeds 13/42/73；
- qualification 数据不用于选择 threshold、cap 或 architecture。

### S300：Selector 模块资格

在同一个冻结 GQ 下比较：

```text
TopK + GQ
SL + GQ
SU + GQ
```

#### A. 技术和证据安全硬门

- runtime gold=false、error=0、trace complete；
- SU 输出始终为 TopK 子集，删除 0–2；
- NIAH required evidence 和 2Wiki supporting/chain point loss 各 <=1pp；
- MUST_KEEP recall 不低于 G010/S100 冻结门；
- 任一 seed 的 supporting/chain loss 不得超过 3pp。

#### B. Selector 职责硬门

相对 `TopK + GQ`：

- `SU + GQ` 的 `correct_and_cited` family point delta >0；
- answer、coverage point delta 各 >= -2pp；
- citation precision/recall point delta 各 >= -3pp；
- wrong->right > right->wrong；
- 至少 2/3 seeds 的联合指标方向非负。

相对 `SL + GQ`：

- `correct_and_cited` 点值必须不低于 SL；
- 如果未超过 SL，不能声称 utility 对齐优于 legacy safety；
- 即使 evidence-level SAFE_DROP 很好，只要同一 GQ 下端到端不提高，就不能冻结 SU 为最终 Selector。

#### C. 强 Selector 结论

只有 `SU + GQ - TopK + GQ` 的主要 paired CI 下界 >0，才声称 Utility Selector 已统计可信地改善相同 Generator。

职责通过但 CI 跨 0 时，可进入完整系统开发验证，但结论保持“正向候选、证据不足”。

### S310：冻结 SQ 或停止

- SU 通过职责门：冻结为 `SQ`；
- SU 失败而 SL 在同一 GQ 下通过相同技术、evidence safety 和端到端职责门：允许冻结 `SQ=SL` 作为风险基线 fallback；SL 不适用 utility-head 的 MUST_KEEP/SAFE_DROP 指标，但其他 answer/coverage/citation/chain margin 不得放宽；
- `SQ=SL` 时不能声称 generator-aware utility 成功，C2 记为不支持；
- SU/SL 均不能相对 TopK 提供正的端到端作用：没有 Selector candidate，完整三模块新方法停止；保留 GQ 单模块结果。

---

## 10. 阶段 I：完整三模块联合验证

### I100：冻结真实 Retriever 输出

- 从当前真实 Retriever 入口运行冻结 Hybrid RRF；
- 缓存候选只为平行臂输入一致；
- manifest 绑定 index、corpus、config、TopK 和 hash；
- 报告 support visibility、oracle answerability 和 noise load；
- 不用旧候选池代替真实 full-flow 入口。

### I200：锁定开发比较

核心臂：

| 臂 | Selector | Generator | 用途 |
|---|---|---|---|
| A | TopK keep-all | G0 | 当前默认系统 |
| B | TopK keep-all | GQ | Generator 相对默认的作用 |
| C | Legacy SL | GQ | 旧风险过滤基线 |
| D | selected SQ（主目标 SU，条件 fallback SL） | GQ | 完整冻结候选 |

如果 `GQ=G0`，A 与 B 完全相同并复用，不制造重复运行。
如果 `SQ=SL`，C 与 D 完全相同并复用；此时不计算或声称 Utility 相对 Legacy 的改善。

关键比较：

```text
Generator 作用       = B - A
Legacy Selector 作用 = C - B
Utility Selector 作用 = D - B
Utility 相对 Legacy   = D - C（仅 SQ=SU 时）
完整系统作用         = D - A
```

这些都是平行实验。部署的 SystemF 仍只运行一次 D。

### I210：完整系统职责门

SystemF 必须满足：

- `D-A correct_and_cited` family point delta >0；
- `D-B correct_and_cited` point delta >0，证明 SQ 对同一个 GQ 有净作用；
- D-A answer/coverage point delta 各 >= -2pp；
- D-A citation precision/recall point delta各 >= -3pp；
- supporting evidence/chain safety 通过；
- unsupported 无依据回答不增加超过 2pp；
- 无 runtime 错误、无 gold boundary 违规；
- 两个以上 seeds 的完整系统联合指标方向非负。

stress、changed-only、support-visible、noise type 和每 seed 结果完整报告，但不要求每个 slice 单独显著。

通过后冻结：

```text
Retriever config + SQ + GQ + prompt + splitter + TRUE = SystemF
```

如果 D-A 正向但 D-B 不正向，只能说 Generator 提供了系统收益，不能声称 Selector 有净贡献，也不能把 D 作为三模块新方法进入最终主张。

### I220：强开发结论

- D-A 主要 paired CI 下界 >0：开发数据支持完整系统强结论；
- D-B 主要 paired CI 下界 >0：开发数据支持 Selector 净作用强结论；
- CI 跨 0 但 I210 职责门通过：仍可冻结 SystemF 进入一次性 held-out，但必须把强结论标记为未建立。

---

## 11. 阶段 H：一次性 system held-out

只有 SystemF 完成冻结、全部 hash 可复算，并获得用户单独授权后，才运行：

- HotpotQA 400；
- MuSiQue-Full 400 answerable + 400 paired unanswerable；
- RGB 300；
- RGB counterfactual 100 只作次级分析。

运行后不再修改方法。

每个数据集分别比较 A/B/D，不 pooling：

- `D-A`：完整系统相对默认；
- `D-B`：Selector 对同一 GQ 的净作用；
- `B-A`：Generator 作用。

### 11.1 最终主张规则

不设置“所有数据集、所有指标、所有 CI 必须同时通过”的单一总门。

按层报告：

1. 每个数据集分别给出 answer、coverage、citation、`correct_and_cited`、错误转换和 CI；
2. 某数据集主要 CI 下界 >0，才对该数据集写统计可信提升；
3. 跨数据完整系统趋势要求至少 2/3 主数据集的 D-A 联合点值为正，剩余数据集没有超过实际容忍范围的退化；
4. 跨数据 Selector 趋势要求至少 2/3 主数据集的 D-B 联合点值为正，剩余数据集无严重 safety/citation 退化；只有 `SQ=SU` 时才能称为 Utility Selector 趋势；
5. 未满足时按数据集报告 mixed result，不用平均 pooling 掩盖差异；
6. held-out 只决定结论强度，不反向选择 checkpoint、阈值或方法。

---

## 12. 统计方案

### 12.1 独立单位

- query provenance component 是主要独立单位；
- 同一 query 的 context/leave-one-out variants 是 repeated measurements；
- seeds 是方法重复，不是三倍样本量；
- sentence/claim 嵌套在 query 内，不单独扩大样本数。

### 12.2 Family estimate

每个 query 先对三个 seed 的 paired delta 求平均，再按 component cluster bootstrap 10,000 次，seed=13，报告 family 95% CI。逐 seed 点值、CI 和方向同时报告。

### 12.3 Answer

- exact McNemar；
- component-cluster paired bootstrap；
- wrong->right / right->wrong；
- answer text same/different。

### 12.4 Citation

每个系统先在 query 内计算 citation precision/recall。不同系统的句子数量和文本不同，不能把“第几句”强行配对。

主比较在共同 answered queries 上配对 query-level 指标，并按 component 整组 bootstrap。另报告完整 answered set 的 macro/micro、answered 数和 claims 数。

MiniCheck 是外部 judge；TRUE 是被测 runtime verifier，不评价自己。

### 12.5 多指标纪律

- 模块职责门只包含该模块必须承担的少量核心条件；
- stress 与机制指标用于解释和严重退化预警；
- 强结论只绑定预注册主要指标；
- 不用某个显著结果抵消核心安全失败；
- 不因为次级 CI 跨 0 就否决职责合格组件。

---

## 13. 预算和停止规则

### 13.1 Generator 预算

- 2 个 seed13 screen fits；
- 3 个 formal fits；
- 1 个仅在 G110 激活 downstream 修复时的 smoke/refit；
- 不做 rank/alpha/dropout/lr/prompt 网格。

### 13.2 Selector 预算

- 100-query leave-one-out pilot；
- 1 次冻结规则的完整 utility materialization；
- 1 个 seed13 recipe/threshold screen；
- 3 个 formal seeds；
- SU-no-safety 只允许一次机制消融，不参与默认选择。

### 13.3 联合与 held-out 预算

- 1 次 locked full-flow development bundle；
- 1 次 SystemF held-out bundle；
- 技术中断可用相同配置重试；效果失败不得换 seed、门或数据后冒充同一协议。

### 13.4 停止与 fallback

- 新 Generator 失败：使用 G0 作为 GQ，Selector 阶段可继续；
- utility pilot 标签不足：停止 Utility Selector，不强造；
- Selector 不提供同 GQ 下端到端正作用：没有三模块候选；
- 完整系统职责门失败：不运行 held-out；
- held-out mixed/negative：报告真实结果，不再调方法。

---

## 14. 产物、Git 和服务器纪律

### 14.1 每阶段保存

- human-readable protocol/report；
- machine-readable manifest；
- ordered IDs 和 SHA256；
- per-query generation/score/citation/utility rows；
- model/tokenizer/prompt/checkpoint identity；
- command、environment、GPU、wall time、error；
- runtime gold boundary；
- tests 和 verify-only 报告。

### 14.2 GitHub

每个可独立复核阶段完成后：

1. fetch 并核对 `origin/refactor/three-module-baseline`；
2. 团队有新提交时正常 pull/merge；
3. 不建立临时 worktree；
4. 不 reset/stash、删除或还原用户文件；
5. 只 stage 本阶段文件；
6. tests/hash/report 一致后 push；
7. tracker 记录 commit 和阶段判定。

### 14.3 服务器

- repo：`/home/fl25387/projects/IBM_Granite_Project_latest`；
- runtime：`/scratch/fl25387/IBM_Granite_Project_latest`；
- 不 reset/stash 服务器 dirty state；
- 服务器 SSH 不可用时不得伪称阶段已实体核验；
- 计划获得用户批准前不启动新 GPU run。

---

## 15. 主要风险与控制

| 风险 | 控制 |
|---|---|
| 把 NIAH harmful 结论外推到所有数据 | 明确 proxy 范围；2Wiki 用 utility 而非伪 harm；最终三数据单独确认 |
| Generator 与 Selector 同时变化 | 只允许 G -> freeze -> S -> freeze -> I |
| 没有新 G 就完全阻断 Selector | G0 fallback teacher；不把新 G 失败等同于 S 不可研究 |
| 资格门过严导致有用组件被淘汰 | 模块职责、强结论、完整系统三层分开 |
| 为了通过而放松可靠性 | 保留 answer/citation/coverage/unsupported 实际 margin 和严重退化 tripwire |
| utility labels 随 Generator 改变 | GQ hash 绑定 labels；换 G 必须重建 |
| 2Wiki 只保护不动作 | 用冻结 GQ 的 leave-one-out utility 学 distractor 作用 |
| MiniCheck 误判 | G110 人工分层审计；TRUE 不自评；保留 raw rows |
| 重复 rows/seed 扩大显著性 | query/component bootstrap；seed 先在 query 内平均 |
| 最终数据被用于调参 | held-out 只在 SystemF 冻结后一次运行，之后只降级主张 |

---

## 16. 用户批准前检查表

- [x] 用户确认旧 Selector 只作为有限范围的 safety/risk baseline。
- [x] 用户确认采用 G -> freeze -> S -> freeze -> I 的单循环协同。
- [x] 用户确认新 Generator 失败时允许 G0 作为 utility teacher。
- [x] 用户确认模块职责通过不等于统计显著优越。
- [x] 用户确认最终三个数据集只在 SystemF 冻结后运行一次。
- [x] G000 protocol 已从本修订快照为 frozen version。
- [x] G230、Legacy Selector 和全部输入 hash 可复算。
- [x] sealed600 与 system held-out denylist 生效。
- [ ] 新 NIAH/2Wiki split leakage 为 0。
- [x] 所有 gate、margin、预算和 fallback 在新结果前冻结。
- [x] GitHub/服务器实体已核对，用户无关文件未被 stage。

以上确认完成前，不启动训练、不生成新 utility labels、不运行 held-out。
