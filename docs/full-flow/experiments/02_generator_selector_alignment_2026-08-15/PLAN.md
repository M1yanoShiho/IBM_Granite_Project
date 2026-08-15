# Generator-first、Selector-aligned：下一阶段执行计划 v1

**日期：** 2026-08-15  
**状态：** `PLAN ONLY / NOT EXECUTED`  
**上一阶段结论：** F005 `FINAL-NO-WIN`；sealed600 已退休  
**研究主线：** Reliability across the full evidence flow  
**对应跟踪表：** [TRACKER.md](TRACKER.md)

---

## 0. 最终执行决定

本路线采用一个明确顺序：

```text
冻结 Retriever 和现有 Selector
        ↓
定位并修复 Generator 的 evidence utilization
        ↓
冻结通过开发门的 Generator
        ↓
用该 Generator 的离线反事实结果形成 evidence utility 标签
        ↓
训练与该 Generator 对齐、但运行时独立执行的 Selector
        ↓
使用真实 Retriever 做 2×2 联合验证
        ↓
在全新 held-out 数据上做一次最终确认
```

最终系统仍然只运行一次：

```text
Question → Retriever → Selector → Generator → Answer
```

多次删除证据、多次生成和读取 reference answer 只允许发生在离线训练与评测阶段，不能进入正式运行路径。

---

## 1. 当前问题锚点

F005 已经证明：

- Selector 删除 82 条证据，其中 74 条是已知有害证据，deletion precision 为 90.24%；
- 它只删除全部 524 条可见有害证据中的 74 条，harmful reduction 为 14.12%；
- `TopK + Base` 与 `Selector + Mixed` 都是 379/600，完整系统没有答案提升；
- coverage 从 90.00% 下降到 87.50%；
- 固定同一个 Mixed Generator 后，Selector 净作用 `C-B=-0.17 pp`；
- sealed600 已完成最终职责，禁止继续用于开发、阈值选择或方法筛选。

F002/F005 同时表明：正确证据仍在时，Generator 会出现事实漏取、空答案和上下文变化导致的答案翻转。F006 只训练了 key-fact extraction call，draft generation、claim splitting 和 TRUE 路径没有被该 adapter 训练。

因此，本路线不再把问题定义为“进一步提高 harmful deletion precision”，而定义为：

> 当 Retriever 已经提供支持证据时，如何让 Generator 稳定使用这些证据；随后如何让 Selector 按该 Generator 的真实需要构造充分、低干扰的 evidence context。

---

## 2. 模块职责与协同方式

### 2.1 Retriever

职责：提高支持证据可见性。下一阶段初期冻结当前候选方案，不调检索参数。只有受控诊断确认候选池缺少必要证据时，才单独开启 Retriever 路线。

### 2.2 Selector

职责：从候选列表构造“支持信息充分、干扰较低”的上下文。现有 harmful/protect 能力作为安全约束保留，但不再作为唯一优化目标。

### 2.3 Generator

职责：在支持证据存在时稳定提取、组合并生成答案；适应 Selector 造成的证据数量、位置和编号变化；最后通过现有 Verify-and-annotate/TRUE 路径核验和引用。

### 2.4 协同原则

```text
结构上分离：每个模块可单独替换、测试和审计
目标上协同：Selector 的训练标签来自 Generator 的离线答案效用
数据上适配：Generator 训练时见到 Selector 会产生的上下文分布
运行时单向：Retriever → Selector → Generator，不回环、不读 gold
```

不做不可解释的端到端联合更新。每次只冻结一个模块、改动另一个模块，然后用联合消融测主效应和交互。

### 2.5 冻结模型栈

本项目保持 **Granite-centered modular RAG**，不要求每个辅助模块都使用 Granite：

| 位置 | 冻结基座/默认模型 | 下一阶段允许的变化 |
|---|---|---|
| Retriever | Strong BM25 + `ibm-granite/granite-embedding-english-r2` | 第一阶段不调参；只在支持证据缺失被确认后另开 Retriever 路线 |
| Selector | `cross-encoder/nli-deberta-v3-base` 的 NLI-aware 双头 checkpoint | 保留 legacy protect/harm safety，增加 Generator utility 目标 |
| Generator | `ibm-granite/granite-4.1-3b`，冻结 snapshot `c0650403...b196c4e` | 只比较同一 Granite 基座的 Base、clean draft LoRA 和 mixed draft LoRA |
| Claim verifier | TRUE：`google/t5_xxl_true_nli_mixture` | 冻结，不与 Generator 训练同时改变 |
| Citation evaluator | MiniCheck | 只做独立评测，不进入生产决策 |

主答案生成模型不得在本路线中替换成其他模型家族，否则不能把结果解释为 Granite evidence-utilization 的提升。Selector 和独立 verifier 保留专门模型，以维持模块效率和避免 Granite 自生成、自评判。

---

## 3. 需要验证的两个主张

| 编号 | 主张 | 最低可信证据 |
|---|---|---|
| C1 | 训练实际 draft/answer 路径的 context-robust Generator，在支持证据仍存在时，比当前 Generator 更少漏答、空答和上下文翻转 | Support-only、TopK、Selected、noise、position 五类上下文上的配对开发结果；新 Generator 在完整分布上优于旧 Generator且 coverage 不劣 |
| C2 | 使用 Generator utility 标签训练的 Selector，比只有 harmful/protect 标签的 Selector 更能提高同一个 Generator 的最终答案质量 | 相同新 Generator 下，Utility Selector 优于 TopK 和 Legacy Selector；独立 held-out 上通过预注册门 |

必须排除的解释：

- 改善只来自更多训练步骤或更多参数；
- 改善只来自生成更多次再挑答案；
- Selector 或 Generator 在 runtime 读取 gold/reference；
- 改善只出现在 Selector-changed 子集，不适用于完整分布；
- 改善来自跨作业 GPU 漂移或固定实验臂顺序；
- 改善只来自删掉更多证据，同时牺牲支持证据或 coverage。

---

## 4. 数据边界与冻结规则

### 4.1 数据角色

| 角色 | 允许用途 | 禁止用途 |
|---|---|---|
| NIAH train（候选池 2,000 题；实际合格数由新规则冻结） | Generator 训练、utility label 构造、Selector 训练 | 最终结论 |
| NIAH decision-dev（full-flow 739 题，当前 changed 109 题） | 失败诊断、方法选择、消融 | 最终 superiority 声明 |
| F005 sealed600 | 只读历史结果 | 任何新方法训练、选择、调参或再次最终测试 |
| 2Wiki train/dev（3,000/2,000 题） | Generator 多证据泛化、Retriever 辅助验证；第一轮不混入主训练 | 代替 NIAH 估计 Selector 主效应；当前 Selector 在该任务几乎不动作 |
| ASQA / QAMPARI（均已揭示） | Generator 与 citation regression | 新的独立最终结论 |
| 新 held-out | 冻结方法后的唯一最终确认 | 在方法冻结前查看逐题标签或结果 |

主路线使用 NIAH，因为它同时提供支持证据、普通干扰和已知有害证据，能够直接测量 Selector–Generator 协同。2Wiki 只承担多文档泛化；SciFact/NQ 保留为 Retriever 模块基准，不进入第一轮 Generator–Selector 训练。

### 4.2 新 split 要求

训练、验证和新 held-out 必须按 `source component / relevant parent page / provenance group` 分组切分，而不只是排除相同 query ID。同一来源组件生成的问题不能跨 split。

执行 A000 时必须：

1. 建立所有旧 train、dev、sealed query ID 和 provenance group 的 denylist；
2. 为新 held-out 确定此前未用于方法选择的数据来源；
3. 先写入 query/candidate/provenance hash，再开始方法开发；
4. 将 held-out gold 放在运行脚本不可读取的独立评分路径；
5. 如果找不到足够的新 held-out，允许继续开发，但禁止形成最终提升结论。

### 4.3 Gold 边界

离线训练允许在 Generator 完成生成后用 reference answer 评分，也允许用 gold provenance 检查是否误删。正式 runtime payload 只允许包含：

```text
query
retrieved candidates
retrieval metadata
Selector 自身预测与动作
selected evidence
```

不允许包含 reference answer、official supporting IDs、gold chain 或 leave-one-out 评分。

---

## 5. 阶段 A：冻结协议与补齐可观测性

### A000：协议、数据和统计门冻结

**目的：** 在任何新训练前冻结数据边界、主比较和统计判定。

**任务：**

- 写出 train/dev/new-held-out manifest 和所有输入 SHA256；
- 冻结上表中的 Granite Generator snapshot、Granite embedding、Selector、TRUE 和 MiniCheck 版本；
- 定义完整分布为主分析，Selector-changed 为次级诊断；
- 用 F005 的 paired discordance 和新数据规模完成 power analysis；
- 冻结 answer superiority、coverage/citation non-inferiority margin；
- 冻结运行环境、prompt 和解码配置；
- 明确新 held-out 只运行一次。

**必须产物：** `A000_PROTOCOL.md`、`A000_DATA_MANIFEST.json`、`A000_POWER.json`。

**通过门：** 数据来源、split、hash、主要比较和判定门全部可复核。否则不得开始正式训练。

### A001：Generator 全链路 trace

**目的：** 确定错误发生在 draft、claim splitting、faithfulness 还是 routing。

**新增 trace 字段：**

- `raw_draft_text`；
- `draft_empty_reason`；
- `splitter_raw_output` 与 parse/fallback 状态；
- 拆分后的每个 claim；
- `faithful_to_answer` 结果与跳过原因；
- TRUE entailment/routing 结果；
- 最终保留、`[unverified]` 或丢弃原因；
- `final_empty_reason`。

**实现边界：** trace 默认只写实验产物，不改变生成行为，不改变生产接口。

**测试：**

- trace 开/关时，同一进程内的最终输出一致；
- 每个空答案都能归入一个明确阶段；
- gold 字段无法进入 Generator 输入对象；
- JSON/claim parser 失败与模型主动输出空字符串可区分。

**通过门：** 100% 空答案具有可审计原因；非空答案能够从 draft 追踪到最终 claim。

### A002：基线复现与运行方差

**比较：** 当前 `TopK + Base Verify-and-annotate` 与 `Legacy Selector + Base`。

**执行：**

- 在已揭示 dev 上同进程配对复现；
- 对至少 10% 分层样本做相同输入重复生成；
- 按 query 轮换实验臂执行顺序；
- 记录 GPU、CUDA、PyTorch、Transformers、kernel 和模型 hash。

**产物：** 基线答案、trace、重复一致率和方差报告。

**停止条件：** 如果同输入运行差异大到足以改变后续 utility 标签方向，先解决执行稳定性或增加重复判定，不能直接进入 utility label 训练。

---

## 6. 阶段 B：Generator evidence-utilization 诊断

### B100：受控上下文矩阵

对每个诊断问题构造以下上下文。所有 oracle 变体只用于离线诊断，不是可部署方法。

| 变体 | 内容 | 回答的问题 |
|---|---|---|
| K | 原 TopK10 | 当前真实基线 |
| S | 当前 Selector 输出 | 删减后发生什么 |
| O | 仅 gold-supported evidence | 没有干扰时 Generator 能否使用正确证据 |
| O+B | 支持证据 + benign distractors | 普通噪声影响 |
| O+H | 支持证据 + harmful distractors | 有害噪声影响 |
| O-P | 相同证据的位置/编号排列变体 | 位置和重编号影响 |

**样本：**

- 全部已揭示 dev 的 Selector-changed 问题；
- 等量、按问题类型和候选结构匹配的 unchanged 问题；
- 如预算允许，再扩展到完整已揭示 dev。

**指标：**

- answer match 和 coverage；
- 空 draft、零 claims、全 claims 被跳过、routing empty 的比例；
- 同一问题不同上下文的答案一致率；
- 支持证据存在但答案错误的比例；
- citation precision/recall；
- 单证据和多证据问题分层结果。

### B110：瓶颈路由

按以下规则决定后续工作：

```text
候选池缺少支持证据
    → 记录为 Retriever bottleneck，单独开 Retriever 路线

O 仍经常错误或为空
    → Generator 基础读取、draft 或 splitter bottleneck

O 正确，但 O+B / O+H / O-P 明显下降
    → Generator context robustness bottleneck

O、噪声和位置均稳定，但 S 失败
    → Selector 删除充分性或选择目标问题
```

**阶段产物：** `B100_DIAGNOSTIC_REPORT.md` 和逐题 `B100_cases.jsonl`。

**通过门：** 每个主要失败类别都有数量、比例和逐题证据；不得只用“Generator 不稳定”作为总括解释。

---

## 7. 阶段 G：训练真正负责答案的 Generator 路径

### G200：训练目标与样本构造

本阶段不继续扩充五类 key-fact slots。训练对象改为实际 `evidence → draft/answer` 路径。

**每个训练问题的目标：**

- 完整回答 reference answer 中受证据支持的内容；
- 每个事实句自包含；
- 引用当前上下文中的支持 evidence 编号；
- 相同问题的不同上下文变体使用同一个语义答案；
- 上下文重排时只重映射 citation index，不改变答案事实。

只有当目标答案能够由当前上下文中的标注证据完整支持时，样本才可进入训练。无法可靠映射来源的样本必须排除并记录原因。

**上下文变体：**

- support-only；
- TopK；
- Legacy Selector context；
- support + benign distractors；
- support + harmful distractors；
- 支持证据处于开头、中间和末尾的排列。

训练/验证按 provenance group 隔离。相同 query 的多个上下文只能位于同一 split。

### G210：工程 fallback 条件分支

如果 B100 显示主要失败是“draft 非空但 splitter 得到零 claims”或 parser 失败，先实现共享的确定性 fallback：

- JSON parse 失败时保留现有 fallback；
- 零 claims 或全部无 anchor 时，按原 draft 的句子边界构造降级 claims；
- trace 中标记 `degraded_splitter_fallback=true`；
- TRUE 仍逐条验证，不把 fallback claim 自动视为可信。

所有 Generator 训练组必须共享相同 fallback，避免把工程修复误当成训练收益。如果 B100 不支持该归因，则不实现此分支。

### G220：最小训练消融

| 组 | 可训练部分 | 训练上下文 | 作用 |
|---|---|---|---|
| G0 | 无 | 无 | 当前 Base Verify-and-annotate |
| GN | 历史 Mixed key-fact adapter | 历史设置 | 负结果/历史控制，不重新调参 |
| GC | draft LoRA | clean/support-only，样本数与 GM 相同 | 排除收益只是普通答案微调 |
| GM | draft LoRA | clean + TopK/Selected/noise/position 混合 | 主候选：上下文鲁棒训练 |

**Adapter scope：**

- LoRA 只在实际 draft generation call 启用；
- key-fact extraction 不作为主训练路径；
- claim splitter 第一轮保持 frozen base；
- TRUE 保持冻结；
- 如果 B100 证明 claim splitter 是独立主瓶颈，另设后续实验，不与第一轮 draft LoRA 同时改动。

**初始训练配置：**

- 以 F006 的 `r=8 / alpha=16 / dropout=0.05 / lr=1e-4` 作为无搜索起点；
- clean 与 mixed 保持相同 query、example 数、optimizer steps 和 target；
- 先用 seed 13 做 smoke，再用冻结配置训练 3 个 seeds；
- max length 根据训练语料 token 分布冻结，并报告截断比例；
- loss 只作用于 assistant target tokens；
- 训练中不读取 dev 或 held-out gold。

这不是授权大规模超参数搜索。只有 smoke 暴露训练失败时，才允许修改配置，并必须在 tracker 记录理由。

### G230：Generator 开发验证

**主比较：** `GM vs G0`；机制比较：`GM vs GC`；历史比较：`GM vs GN`。

**主数据：** 完整已揭示 dev。B100 诊断子集只用于解释机制，不能代替完整分布。

**候选进入下一阶段的开发门：**

1. 完整分布 answer match 点估计高于 G0；
2. coverage 满足 A000 预注册的 non-inferiority margin；
3. 空 draft、零 claims 或 final empty 至少一个主要失败率实质下降；
4. 在 O+B/O+H/O-P stress slices 上，GM 优于 GC；
5. citation 指标不超过预注册退化边界；
6. 3 seeds 的方向一致，不允许只挑最好 seed；
7. GM 必须表现出相对 GC 的鲁棒性证据，否则只保留更简单的 GC，不声称 mixed-context 方法成立。

开发门只决定是否值得训练 Selector，不形成最终结论。

**停止条件：** 如果 GM/GC 都不能让 support-only 可靠性和空答案优于 G0，停止 Selector utility 训练，继续定位 Generator，不把不可靠 Generator 当教师。

---

## 8. 阶段 S：构造 Generator-aware Selector

### S300：冻结教师 Generator

从 G230 按预注册规则选择一个 Generator 配置 `G*`，冻结：

- adapter checkpoint 和 seed 处理方式；
- prompt、decode、claim splitter、TRUE 配置；
- runtime 环境和 hash。

形成 utility labels 后不得再修改 `G*`。如果以后更换 Generator，utility labels 必须重新生成。

### S310：离线 evidence utility 标签

对 NIAH train 中每个候选列表 `C`，先运行 `G*(q,C)`，再运行删除单条证据后的 `G*(q,C\{e_i})`。

定义标签：

| 标签 | 离线判定 |
|---|---|
| `MUST_KEEP` | 删除后由正确变错误、由有答案变空答案，或丢失完成问题所需的受支持事实 |
| `SAFE_DROP` | 删除后由错误变正确；或删除已知 harmful/unsupported distractor 后正确性与 coverage 保持、citation 不退化，且没有删除必要 supporting evidence |
| `NEUTRAL` | 删除前后正确性、coverage 和受支持内容均无实质变化，同时没有独立证据表明该候选有害或必须保留 |
| `UNCERTAIN` | 重复运行方向不一致、评价无法判定或上下文本身不充分 |

执行纪律：

- gold/reference 只在两次生成完成后评分；
- full 与 leave-one-out 在同一作业内配对，并平衡执行顺序；
- 所有出现对错转换的样本至少重复一次；
- 只有方向一致的转换进入 `MUST_KEEP/SAFE_DROP`；
- `UNCERTAIN` 默认训练为保留或从 utility loss 中排除，不能当作可删除；
- 先做 100 题全 leave-one-out pilot，统计标签产出率和重复一致率，再决定是否扩展；
- 正式扩展优先覆盖所有候选；若计算受限，抽样规则必须在看结果前冻结并覆盖不同 rank、相关性和 harm 类型。

**Pilot 停止门：** 如果稳定 `SAFE_DROP/MUST_KEEP` 标签过少，或重复一致率不足，停止训练 utility head；保留诊断结论，不强造 Selector 方法。

### S320：Utility Selector 训练

最小实现保持现有 Selector 在系统中的独立模块身份：

- 冻结或保留现有 protect/harm 评分作为 safety guard；
- 在 Selector 内增加 generator-utility 预测目标；
- runtime 只根据 query、候选证据和候选列表特征预测 utility；
- 只有 legacy safety rule 允许、且 utility 预测为稳定负效用时才删除；
- 预测为正效用或不确定时保留；
- 第一轮冻结 F005 的 threshold 和删除 cap，以隔离 utility 目标的作用；
- 只有 utility 版本通过开发门后，才能单独研究 cap、排序或 compact-set 输出。

**训练比较：**

| 组 | 训练目标 | 目的 |
|---|---|---|
| S0 | 不选择，保留 TopK | 系统基线 |
| SL | F005 legacy harmful/protect Selector | 证据标签基线 |
| SU | legacy safety + Generator utility | 主候选 |
| SU-no-safety | 只有 utility | 安全约束消融，只做开发/附录，不作为默认部署 |

**Selector 自身指标：**

- `MUST_KEEP` recall；
- `SAFE_DROP` precision/recall；
- legacy harmful deletion precision/reduction；
- gold/supporting evidence deletion；
- 每题删除数和 unchanged 比例；
- 在 G* 下的 answer utility transition。

### S330：Selector 开发门

SU 进入联合最终候选必须同时满足：

1. 相同 G* 下，`SU + G*` 的完整分布答案点估计高于 `TopK + G*`；
2. `SU + G*` 高于或不差于 `SL + G*`，证明收益不是 legacy Selector 已经具备；
3. coverage 和 citation 满足 A000 non-inferiority margin；
4. `MUST_KEEP` recall 和 supporting-evidence safety 不低于预注册门；
5. Generator utility 转换中 `wrong→right > right→wrong`；
6. 结果在 3 seeds 或预注册重复下方向一致。

如果 SU 只提高 evidence-level 指标而不提高相同 G* 的答案，停止，不进入 held-out。

---

## 9. 阶段 I：真实三模块联合验证

### I400：冻结真实 Retriever 输出

最终联合实验必须从当前真实 Retriever 入口生成候选，而不是只引用旧冻结 pool 的检索层结论。

执行方式：

- 使用冻结的 Hybrid RRF 配置对 dev/new-held-out 运行一次；
- 缓存 TopK candidate pool 只是为了让各实验臂输入完全一致；
- manifest 记录 Retriever 配置、index、corpus 和候选 hash；
- 报告 support recall@K、oracle answerability 和 harmful load；
- 如果支持证据不可见，错误归 Retriever，不让 Selector/Generator 承担不可解任务。

### I410：五臂开发联合实验

A/B/C/D 组成核心 `Selector off/on × Generator old/new` 2×2；H 是额外的 legacy Selector 对照，不属于新的系统组件。

| 臂 | Retriever 输出后 | Generator | 回答的问题 |
|---|---|---|---|
| A | TopK | G0 | 当前默认系统 |
| B | SU | G0 | Utility Selector 对旧 Generator 是否有效 |
| C | TopK | G* | 新 Generator 单独作用 |
| D | SU | G* | 完整新系统 |
| H | SL | G* | utility 对齐是否优于 legacy harmful Selector |

关键比较：

```text
Generator 主效应             = C - A
Utility Selector 对旧 G 的作用 = B - A
Utility Selector 对新 G 的作用 = D - C
Selector-Generator 交互       = (D - C) - (B - A)
Utility 对齐相对 legacy       = D - H
完整系统总作用               = D - A
```

所有臂是平行实验系统。正式部署仍然只有其中一个冻结系统生成一次答案。

### I500：新 held-out 最终确认

只有 G230、S330 和 I410 都通过开发门，才运行新 held-out。

**主要比较：**

1. `D-A`：完整系统是否优于当前默认系统；
2. `D-C`：同一个新 Generator 下，Selector 是否提供净收益；
3. `D-H`：generator-aware utility 是否优于 legacy harmful/protect 目标。

**共同主要结果：** answer correctness 与 coverage。Citation reliability、evidence safety 和稳定性作为关键次要结果。

**统计：**

- paired bootstrap 95% CI；
- binary correctness 的 exact McNemar 检验；
- 报告 `wrong→right/right→wrong`，不只报告均值；
- 完整分布为主，Selector-changed、support-visible、noise 类型为预注册分层；
- 多 seeds 报均值、范围和每个 seed 方向，不挑最好 seed；
- 同一作业中平衡或轮换实验臂顺序。

**最终通过条件：**

- `D-A` 的 paired 95% CI 通过 A000 冻结的 answer superiority 门；
- `D-C` 的 paired 95% CI 通过 A000 冻结的 Selector 净收益门；
- 如果要声称 utility 对齐优于 legacy 目标，`D-H` 也必须通过对应的预注册置信区间门；
- coverage/citation 不越过 non-inferiority margin；
- supporting evidence safety 不退化；
- 运行错误为 0，runtime gold boundary 审计通过。

任何一个条件未通过，都按对应主张降级，不在 held-out 上继续调参。

---

## 10. 结果报告模板

每一阶段必须同时保存机器产物和可读报告。

### 10.1 Generator 表

| System | Answer | Coverage | Empty draft | Zero claims | Routing empty | Citation P/R | Context stability |
|---|---:|---:|---:|---:|---:|---:|---:|

### 10.2 Selector 表

| Selector | Changed Q | Dropped | SAFE_DROP precision | MUST_KEEP recall | Harm reduction | Gold/support drops |
|---|---:|---:|---:|---:|---:|---:|

### 10.3 联合结果表

| Arm | Answer | Coverage | Citation P/R | W→R | R→W | 95% CI vs baseline |
|---|---:|---:|---:|---:|---:|---:|

### 10.4 必须提供的逐题字段

```text
query_id
retrieved evidence IDs and ranks
Selector scores/actions/reasons
selected evidence IDs and order
raw draft
split claims and discard reasons
TRUE routing
final answer and citations
post-generation answer/citation scores
runtime_gold_loaded=false
```

---

## 11. 运行顺序、资源和决策门

| 顺序 | Milestone | 主要成本 | Go 条件 | Stop 条件 |
|---:|---|---|---|---|
| 1 | A000 数据/协议冻结 | 本地 | manifest、power、门完整 | 无新 held-out 时禁止最终声明 |
| 2 | A001 trace | 本地开发 | 空答案 100% 可归因 | trace 改变输出 |
| 3 | A002 基线/方差 | 约 2 个基线臂 + 重复子集 | utility label 可稳定比较 | 同输入波动过大 |
| 4 | B100 诊断 | 约 6 个上下文变体 | 主瓶颈清楚 | 归因仍混杂则扩大 trace，不训练 |
| 5 | G200/G220 | 2 arms × 3 seeds，另有 smoke | GM/GC 至少一项通过 G230 | support-only 仍无改善 |
| 6 | S310 pilot | 100×约 11 个上下文，转换重复 | utility 标签产出和一致率可用 | 标签稀少或方向不稳 |
| 7 | S310 full/S320 | 训练集候选 leave-one-out + Selector 训练 | SU 通过 S330 | 只改善删除指标、不改善答案 |
| 8 | I400/I410 | Retriever 一次 + 五臂 dev | D-A、D-C、D-H 达开发门 | 任一核心机制失败 |
| 9 | I500 | 五臂 fresh held-out，一次 | 全部最终门通过 | 结果归档，不在该集继续开发 |

GPU 小时不在计划阶段伪造。A002 和 B100 完成后，以真实吞吐将“生成调用数、TRUE claims 数、训练 steps”换算为预算，并写入 tracker。申请资源前必须给出这个测量值。

---

## 12. 明确不做的事情

- 不再使用 sealed600 选择方法；
- 不继续围绕 F005 threshold/cap 做扫描；
- 不把五类 key facts 扩展当作主路线；
- 不同时训练 draft、splitter、Selector，避免无法归因；
- 不让 Selector runtime 调用 Generator 多次；
- 不让 runtime 使用 reference answer 或 official provenance；
- 不在主瓶颈未确认前增加 F003C、迭代检索或 agent loop；
- 不因为开发集点估计为正就宣布成功；
- 不只报告 Selector-changed 子集；
- 不把 greedy decoding 当作天然可复现。

---

## 13. 代码任务映射

预计涉及的现有入口：

- `src/evidence_rag/generator/draft.py`：draft trace、adapter scope、条件 fallback；
- `src/evidence_rag/generator/claim_splitter.py`：splitter trace/fallback；
- `src/evidence_rag/generator/verify_annotate.py`：faithfulness 与 routing trace；
- `src/evidence_rag/generator/granite.py`：冻结 base/adapter 启用边界；
- `src/evidence_rag/selector/risk_controlled.py`：legacy safety 与 utility 决策组合；
- `src/evidence_rag/pipeline/service.py`：保持正式单向三模块调用；
- `scripts/`：新增诊断、draft LoRA、utility label、联合 eval 入口；
- 本路线 `artifacts/`：每阶段 manifest、逐题 trace、报告和 hash。

每次代码提交只覆盖一个 milestone。用户工作区中的无关文件不得删除、还原或混入实验提交。

---

## 14. 最终完成定义

只有满足以下全部条件，才能说“下一阶段完整系统获得可靠提升”：

1. 新 Generator 在支持证据可见时更稳定，并在完整分布上优于旧 Generator；
2. Utility Selector 在相同新 Generator 下优于 TopK 和 Legacy Selector；
3. 实际 Retriever、Utility Selector、新 Generator 组成的一次性正式流程优于默认系统；
4. coverage、citation 和 supporting-evidence safety 没有用不可接受的退化换取答案提升；
5. 结果来自全新 held-out，置信区间和预注册门支持主张；
6. runtime 未读取 gold，且所有输入、模型、配置和逐题产物可复核。

如果只满足第 1 条，结论是 Generator 改进成立，Selector 协同尚未成立。  
如果第 1、2 条成立但第 3 条失败，结论是检索分布或模块交互仍未解决。  
如果 evidence 指标改善但答案不改善，不能把它写成端到端系统提升。
