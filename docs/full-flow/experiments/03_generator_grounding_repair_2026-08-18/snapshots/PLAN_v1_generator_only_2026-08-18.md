# Generator 证据使用与引用修复执行计划

**路线：** `03_generator_grounding_repair_2026-08-18`
**日期：** 2026-08-18
**状态：** `DRAFT FOR REVIEW / NO NEW RUN AUTHORIZED`
**上一阶段：** G230 `COMPLETE / NO CANDIDATE`
**主模型：** `ibm-granite/granite-4.1-3b@c0650403...`
**执行 tracker：** [TRACKER.md](TRACKER.md)

---

## 1. 一句话目标

在不改变已完成 Retriever 和 frozen Selector 结论的前提下，训练并资格审查一个新的 Granite draft Generator：它必须保留 G230 已观察到的 answer/coverage/empty 改善，同时恢复事实句与最终引用之间的可靠对应；只有在 NIAH、2Wiki 和已揭示 citation regression 数据上共同通过开发门，才冻结为 `G*` 并解锁后续 Generator-aware Selector。

---

## 2. 当前实际情况

### 2.1 已经成立的部分

1. 正式系统仍是单向运行：

   ```text
   Retriever -> Selector -> Generator -> one answer
   ```

2. F005 Selector 是保守的 harmful-evidence filter。它在 sealed600 删除 82 条，其中 74 条为已知 harmful，deletion precision 90.24%；这些是已完成历史结果，不在本路线重新调 threshold 或 cap。
3. G230 的 GC/GM draft LoRA 均把 draft empty 降到 0%，把 final empty 从 G0 的 12.04% 降到约 3.79%–6.09%。
4. GM 的完整 NIAH decision-dev answer match 点估计相对 G0 提高约 3.52–6.36pp；GM13/GM42 的单 seed CI 高于 0，GM73 的 CI 跨 0。
5. 所有 G230 运行错误和 trace 缺失为 0，runtime 没有加载 gold。

### 2.2 仍未解决的部分

1. GM 没有在 support+benign、support+harmful 和 support-last 三类 stress 上全部稳定优于 GC，因此 mixed-context robustness 门失败。
2. GC/GM 的独立 MiniCheck citation precision/recall 相对 G0 下降约 6–8pp，明显越过预注册的 -2pp 非劣边界。
3. 当前训练只直接更新 draft LoRA；claim splitter 使用 adapters-disabled frozen Granite base，TRUE 也冻结。最终 citation 经过 draft citation hint、claim splitting、TRUE scan/routing 和 final attachment 多个步骤，不能把全部退化直接归因于 LoRA。
4. G200 虽有 4,120 rows，但只来自 515 个 train query 的 8 个上下文变体；目标主要是一句 declarative answer 加一个 citation index，语义和引用结构仍然偏单一。
5. 因 G230 没有可靠 `G*`，第 02 路线的 S300 utility-label Selector 尚未解锁。

### 2.3 当前正式判定

```text
Retriever: 保持已完成配置；不是本路线变量
Selector:  保持 F005 frozen 配置；不是本路线训练对象
Generator: G0 仍是可靠性基线；GC/GM 是有价值但未通过的开发资产
G*:        不存在
Held-out:  HotpotQA / MuSiQue-Full / RGB 继续封存
```

---

## 3. 计划能保证什么，不能保证什么

实验计划不能诚实保证模型一定取得正增益。它能提高成功概率并保证即使结果为负，也能得到明确、可复核的下一结论。

本计划通过以下设计降低再次“完整训练后才发现目标错位”的风险：

1. 先用已有逐题产物定位 citation 断点，不先训练；
2. 先在全新 train/model-val 内比较 continuation 和 fresh training，不在 qualification 数据上挑方法；
3. 训练目标同时监督 answer text、atomic claim format 和 citation index，不再把 citation 当作少量普通 token；
4. 加入 2Wiki 多证据与 unsupported context，避免只适应 NIAH 单一格式；
5. 只允许一个配方进入三 seed 和跨数据集资格审查；
6. answer、coverage、citation、context robustness 是联合硬门，不能用一个高分掩盖另一个退化；
7. system held-out 仍只在完整系统冻结后运行一次。

如果唯一正式候选未通过 R420，本路线必须报告 `NO CANDIDATE`，不得在同一 qualification bundle 上继续改 prompt、loss、seed 或 gate。此时当前 G0 和原系统仍保持有效基线，不会破坏已完成工作。

---

## 4. 本路线的主张与反主张

### C1：主要主张

**Claim-grounded context training 能在 frozen Granite 4.1-3B 上保留 answer/coverage/empty 改善，同时使最终 citation precision/recall 对 G0 非劣。**

最低可信证据：

- 三 seed family-level 的 `correct_and_cited` 高于 G0；
- answer correctness 不下降且保持正方向；
- coverage 和 citation precision/recall 的配对 CI 不越过 -2pp；
- final empty 仍明显低于 G0；
- 结果不是只来自挑选 GM42 或只在 Selector-changed 小子集成立。

### C2：支持主张

**加入多证据、噪声、位置和 unsupported contexts 后，Generator 的证据使用不只适用于 NIAH 的单一格式。**

最低可信证据：

- harmful-noise stress 优于 G0；
- benign-noise 和 support-last 至少非劣；
- 2Wiki dev 的 answer/coverage/citation 不退化；
- ASQA/QAMPARI 已揭示 regression 上 citation 不出现新的明显退化。

### 必须排除的解释

- 提升只因为模型回答更多，而新回答没有可靠引用；
- citation 下降只是双方 answered set 不同造成；
- 收益只来自某一个 seed；
- 收益来自同时修改 splitter/TRUE，而不是训练设计；
- 方法只适用于 NIAH 的人工干扰格式；
- 重复的 context rows 被错误当作 4,120 个独立问题。

---

## 5. 不破坏既有工作的边界

### 5.1 保持冻结

- Retriever 当前 Hybrid RRF/default 路线、index 和模块结论；
- F005 Selector checkpoint、threshold `0.9212157130241394`、每题删除 0–2 条的动作规则；
- Granite 4.1-3B base revision、tokenizer、基础 prompt contract 和 greedy decode；
- TRUE checkpoint及其作为 runtime verifier 的角色；
- G0、GC、GM 历史 checkpoint、manifest 和 G230 结果；
- sealed600 retired 状态；
- HotpotQA、MuSiQue-Full、RGB 的预抽样 IDs 和 hash。

### 5.2 默认只训练 draft

R100/R110 没有证明 downstream 为主要 citation 断点时：

- 只训练 draft LoRA；
- claim splitter 保持 frozen base；
- TRUE 保持 frozen；
- MiniCheck 只在生成后评分，不进入 runtime，也不把自身输出作为训练标签。

### 5.3 条件组件修复

R100 将每个 citation regression 归到“最早发生错误的阶段”：

1. `UNSUPPORTED_DRAFT_CLAIM`：draft 事实本身不受 evidence 支持；
2. `DRAFT_CITATION_MISSING_OR_WRONG`：事实可支持，但 draft hint 缺失或指错；
3. `SPLITTER_BOUNDARY_OR_REWRITE`：draft hint 正确，拆分/faithfulness 后事实边界或语义改变；
4. `TRUE_ROUTING_OR_ATTACHMENT`：draft 与 split 正确，但 TRUE scan 或 final attachment 指错/遗漏；
5. `EVALUATOR_DISAGREEMENT`：自动指标与人工/规则审计不一致；
6. `NO_REGRESSION`：没有可确认的 citation 退化。

路由规则在看新训练结果前固定：

- `UNSUPPORTED_DRAFT_CLAIM + DRAFT_CITATION_MISSING_OR_WRONG` 数量不少于 downstream 两类时，进入默认 draft repair；
- downstream 两类更多时，先执行 R120 或 R130 的单组件修复，再使用所有训练臂共享的同一修复；
- 两类 downstream 中谁更多，就只激活谁；并列时先选择不训练新模型的 deterministic splitter/attachment 修复；
- 不允许在一个实验臂里同时改变 draft、splitter 和 TRUE 后声称知道收益来源。

TRUE 本轮不做参数微调。若 R130 激活，只允许修复 evidence scan、declared-index handling 或 final attachment 的确定性逻辑，并用独立 tests 和 revealed data 证明行为。

R110 的人工复核样本也在看新模型结果前固定：若 citation regression rows 不超过 120 条则全部复核；否则从自动退化、自动改善、`declared-correct/final-wrong`、`final-correct/MiniCheck-wrong` 四个 strata 固定抽取 120 条，每个非空 stratum 至少 20 条，其余按数量比例分配。至少 20% 由两名复核者在不知道 G0/GC/GM arm 的条件下独立判断，分歧经裁决后归档。人工结果用于判断错误发生在哪一层和解释 MiniCheck disagreement，不替换预冻结的主要自动指标。

---

## 6. 数据集角色

| 数据 | 本路线用途 | 明确禁止 |
|---|---|---|
| NIAH train / G200 train-fit | harmful/benign、position、selected context、claim-citation 训练；构造 unsupported pair | 最终结论 |
| NIAH train 内全新 model-val | 从旧 G200 全部 1,023 个 role-assigned queries 之外抽取；R310 选择 fresh 或 continuation | 训练；最终结论 |
| G200 已揭示 model-val 62 | 只作旧行为回归报告 | R310 tie-break 或正式 qualification |
| NIAH decision-dev 739 | R400 唯一 locked qualification；完整分布与 stress | R310/R320 调配方；最终 superiority 声明 |
| 2Wiki train | 多证据/多文档 claim-citation 训练和独立 model-val | 替代 NIAH 估计 harmful Selector 效应 |
| 2Wiki dev 2,000 | R410 跨数据集 qualification | 训练、反复调参 |
| ASQA/QAMPARI（已揭示） | citation regression 和多事实输出诊断 | 新的独立最终结论；主训练数据 |
| F005 sealed600 | 只读历史结果 | 任何训练、筛选、调参或再次测试 |
| HotpotQA 400 | 完整系统冻结后的 multi-hop/distractor 最终测试 | 本路线任何读取、生成或评分 |
| MuSiQue-Full 400 对 | 完整系统冻结后的 answerable/unanswerable 最终测试 | 本路线任何读取、生成或评分 |
| RGB 300 + cf 100 | 完整系统冻结后的 noise/counterfactual 最终测试 | 本路线任何读取、生成或评分 |
| SciFact/NQ | Retriever 模块历史基准 | 本轮 Generator 训练 |

### 6.1 split 规则

1. 切分单位是 provenance component、relevant parent page 和 question group，不只是 query ID。
2. 同一问题的所有 context variants 必须位于同一个 split。
3. 训练和 model-val 不允许与 NIAH decision-dev、sealed600 或 system held-out 的 query/component/parent 重合。
4. 新 NIAH model-val 还必须与旧 G200 的 1,023 个 role-assigned queries 及其 provenance components 不重合；旧 62 题只作已揭示 regression，不能参与新配方选择。
5. 如需为新 NIAH model-val 产生候选证据，只能运行当前 frozen Retriever/config；这是 R200 数据物化，不是 Retriever 方法实验。
6. 2Wiki train 内按 component/group 做一次固定 train/model-val 切分；official dev 不参与该切分。
7. 所有旧 train/dev/sealed IDs 和 provenance groups 进入 denylist。
8. 先写 manifest 和 SHA256，再产生 target 或训练；数据冻结后不得按效果删题。

### 6.2 为什么仍保留 NIAH

NIAH 是当前唯一同时提供支持证据、普通干扰、已知 harmful twin 和可控位置变化的数据，因此仍是 Selector–Generator 协同的主要受控训练源。问题不是继续使用 NIAH，而是只使用 NIAH 并把其开发结果误当作通用结论。

### 6.3 为什么加入 2Wiki

2Wiki 用于增加语义和结构多样性：一个答案可能依赖多个文档或证据链。它不能替代 NIAH 的 harmful 标签，但可以检查 Generator 是否只学会了单句、单 carrier、单 citation 的 NIAH 模板。

### 6.4 为什么不提前使用三个最终数据集

我们可以依据公开任务定义训练“多跳、抗噪、可靠引用、证据不足时不乱答”等能力，但不能依据预留样本的逐题错误和分数修改系统。最终数据一旦参与方法选择，就不再是最终独立测试。

---

## 7. 新训练数据定义

### 7.0 监督文本从哪里来

监督答案不能由 G0、GC、GM、GR-F 或 GR-C 自己生成后再作为“正确答案”。来源固定如下：

- **NIAH train：** 优先复用 G200 已通过审计的 515 个 train-fit targets。它们来自官方 single reference answer，经冻结的 `MarkS/bart-base-qa2d@94f286a...` 离线改写，并已通过 answer-preservation 和 TRUE entailment 审计；本轮只增加原子格式、context 和 citation 监督，不重新生成语义答案。
- **新 NIAH model-val：** 在旧 G200 1,023 题之外按固定 component 顺序抽样，使用同一冻结 QA2D revision 和同一审计规则，直到得到至少 100 个合格 groups。抽样停止只取决于预冻结的数据质量门，不取决于 GR-F/GR-C 结果。
- **2Wiki：** 只使用 official train 的 answer、`evidences`、`supporting_facts` 和 context。将官方 evidence triples 按固定模板改写为原子事实，按官方 chain order 组成多句 target；每条事实必须一对一映射到 supporting context 并通过 entailment 审计。不能使用 Granite 候选输出补写 chain。
- **Unsupported：** 由 train split 内确定性移除必要 support 后构造，target 固定为 `I don't know.`。

QA2D 和 evidence-triple 模板只是离线监督文本转换器，不是 runtime 模块，也不成为评估 ground truth。评估 correctness 仍使用数据集官方 answer/reference；转换后的文本只在训练数据审计中使用。

### 7.1 以 question group 为训练单位

统计和 optimizer schedule 都以不同问题数为准，不以展开后的 context row 数为准。每个 question group 的全部变体合计只占相同总权重，避免一个问题因为有 8 个排列就比另一个问题重要 8 倍。

R210 的最低数据门：

- NIAH 至少有 400 个合格 answerable train groups，另有至少 100 个不重叠的 model-val groups；
- 2Wiki 至少有 400 个合格 answerable train groups，另有至少 100 个不重叠的 model-val groups；
- unsupported groups 占正式 optimizer updates 的 10%–15%；
- 任一 answerable 数据源不得超过 answerable updates 的 55%；
- 最终数量和比例只根据 target audit 后的可用数据冻结，不根据模型效果改变。

如果无法达到最低 unique-question 数量，R210 失败，不用重复 context rows 填满配额。

### 7.2 Answerable target

每个 target 必须满足：

- 只包含当前 evidence context 完整支持的事实；
- 每句话只表达一个可独立核验的事实；
- 每个原子事实句以当前 context 中一条正确的 evidence index 结尾；这与现有 runtime 中“每个 claim 最终至多绑定一条 verified evidence”的接口一致；
- 多跳答案写成多条原子事实组成的证据链，每个事实各自绑定一条证据；如果一个复合句必须联合多条证据才成立、任一单条证据都不能支持，则不能作为本轮训练 target；
- 同一问题不同 context 中语义答案不变，只重映射 citation index；
- official answer/provenance 只用于离线构造和生成后评分，不进入 runtime payload。

NIAH 每个 group 至少包含四类上下文：

1. support-only；
2. original TopK 或 frozen Selector context；
3. matched benign/harmful noise（在 group 间平衡）；
4. support position reorder。

2Wiki 每个 group 至少包含：

1. official supporting context；
2. supporting context + official distractors；
3. reordered context；
4. 一个保持完整 evidence chain 的 compact context。

### 7.3 Unsupported target

从 train split 内构造配对样本：保持同一个问题和非支持 evidence，但移除完成答案必需的 supporting evidence。target 固定为 prompt contract 中的 `I don't know.`，不带 factual citation。

必须满足：

- 移除后 evidence 中不存在 reference answer 的支持 carrier；
- 不是简单通过 document ID 或特殊前缀泄漏标签；
- answerable/unsupported 两个版本只能位于同一个大 split；
- unsupported 不超过 15%，防止重新训练出大量空答案；
- coverage 和 ungrounded assertion 在 model-val 单独报告。

### 7.4 Target 审计

进入正式训练前，每个 target 必须通过：

1. citation index 范围与 evidence ID 可逆映射；
2. target answer token/alias 保真；
3. 每个事实与其 minimal support 的 TRUE entailment 审计；
4. citation 去掉后答案文本保持一致的重排检查；
5. context 不包含来自其他 split 的 source parent；
6. unsupported context 的支持缺失检查；
7. 随机分层人工审计，包括 NIAH、2Wiki、多证据链和 unsupported 四类。

自动审计失败的 row 必须排除并记录原因，不能降低门来凑数量。

人工审计在看模型结果前冻结样本：若合格 target groups 不超过 120 个则全部检查；否则固定抽取 120 个 groups，每个非空类别至少 20 个，其余按数据源和 target 类型比例分配。至少 20% 由两名审计者独立重叠复核，分歧必须裁决并记录；正式训练数据的人工审计通过率必须不低于 95%。

---

## 8. 训练方法

### 8.1 固定模型和作用域

- Base：IBM Granite 4.1-3B frozen revision；
- LoRA：`r=8`、`alpha=16`、`dropout=0.05`；
- Adapter 只在 draft generation call 启用；
- claim splitter 默认 adapters disabled；
- TRUE frozen；
- greedy decode；
- max input/target length 由 R210 token audit 冻结，正式训练截断率必须为 0；
- loss 只作用于 assistant target tokens。

### 8.2 新的 supervision

旧 G220 对全部 target token 使用相同权重。本路线改为：

```text
answer / punctuation token weight = 1
citation brackets and index token weight = 4
prompt token weight = 0
```

目的不是让模型只输出编号，而是避免一两个 citation token 在整句 token loss 中几乎没有影响。权重 4 在 R000 冻结，不做网格搜索。

每个 optimizer update 对一个 question group 的 context variants 做等权平均；seed×variant 不是额外独立样本。

### 8.3 两个 seed-13 screen 配方

| 配方 | 初始化 | 学习率 | 数据/步数 | 回答的问题 |
|---|---|---:|---|---|
| GR-F | frozen Granite base 上新建 LoRA | `1e-4` | 与 GR-C 完全相同 | 从头联合学习 answer+citation 是否更干净 |
| GR-C | G230 GM13 adapter 继续训练 | `5e-5` | 与 GR-F 完全相同 | 是否能保留 GM answer/empty 能力并修复 grounding |

GM13 只作为 warm-start 初始化，不是已接受的 `G*`。不使用 GM42 作为起点，避免根据最好开发分数挑 seed。

这是面向落地的“两个训练配方选择”，不是只研究初始化变量的因果消融：GR-C 已经承受过 G220 的旧训练 updates，且学习率不同。报告必须同时列出其历史与新增计算量，不能据此声称差异只由 fresh/continuation 初始化造成。

两个 screen 配方共享：

- 相同 frozen train/model-val questions；
- 相同 query-group schedule 和总 optimizer updates；
- 相同 target、loss weighting、batch/gradient accumulation；
- 相同 model-val 生成任务；
- 相同 splitter、TRUE 和 MiniCheck；
- 不读取 NIAH decision-dev、2Wiki official dev 或 system held-out。

### 8.4 正式训练

R320 只能选 GR-F 或 GR-C 中一个配方。随后以同一配方训练 seeds 13/42/73：

- 每个 seed 只保存 final-completed checkpoint；
- 不按 qualification score 选择 epoch；
- persisted adapter 必须 fresh-base reload；
- 记录 train loss、citation-token loss、answer-token loss、model-val metrics、参数 fingerprint、peak memory 和 wall time；
- formal 训练后不修改 lr、rank、数据比例或 steps。

### 8.5 计算预算上限

本路线在重新写协议前最多允许：

- 2 个 seed-13 screen fits；
- 3 个 formal fits；
- 1 个仅在 R110 激活 downstream 组件修复后、且未打开 qualification bundle 时的 smoke/refit；
- 不做 rank、alpha、dropout、learning-rate 或 prompt 网格搜索。

这不是为了省掉必要实验，而是避免把有限 dev 反复用于超参数选择。

---

## 9. 方法选择规则

### 9.1 R310 只用 train/model-val 选择训练配方

先对 GR-F 与 GR-C 做以下硬筛选：

1. runtime error、missing trace、invalid citation index 都为 0；
2. draft citation precision/recall 和 final MiniCheck citation precision/recall 相对同一 G0 model-val 不低于 -2pp 点值；
3. answer correctness 和 coverage 相对 G0 不低于 -2pp 点值；
4. unsupported ungrounded assertion 不高于 G0；
5. final empty 不回到 G0 以上。

被硬筛选排除的配方不能靠综合平均分复活。

如果两者都存活，使用预先固定的 `correct_and_cited` 选择：

```text
correct_and_cited = answer_match is true
                    AND every factual answer sentence has a citation
                    AND MiniCheck supports each sentence-citation pair
```

分别计算 NIAH model-val 和 2Wiki model-val 相对 G0 的 delta，选择“两个数据集中较小 delta 更高”的配方，即 maximin 规则。仍并列时选择 GR-F，因为它不继承 G230 的 citation 缺陷，方法也更简单。

如果两个配方都失败，R320 为 STOP；不能打开 qualification bundle。

### 9.2 R320 后不再选择方法

R320 冻结：

- 唯一训练配方；
- 三 seed；
- dataset/hash；
- prompt/decode；
- splitter/TRUE；
- answer/citation/coverage/stress gates；
- bootstrap 和 random seeds；
- qualification run order。

R400 与 R410 是同一个 locked qualification bundle。不能在先看到 NIAH 后，再决定是否改模型去跑 2Wiki。

---

## 10. 评估指标

### 10.1 共同主要指标

1. `correct_and_cited`：本路线最重要的联合结果；
2. answer correctness / `system.core.answer_match`；
3. coverage；
4. MiniCheck sentence-final citation precision；
5. MiniCheck sentence-final citation recall。

### 10.2 机制指标

- draft empty；
- zero claims；
- final empty；
- declared citation present/correct；
- declared citation verified；
- rescued-by-scan；
- splitter raw/located/retained claim counts；
- TRUE verified/unverified/skipped；
- unsupported ungrounded assertion；
- answer text stability across paired contexts；
- citation changes under evidence reorder。

### 10.3 Stress 指标

- support-only；
- support+benign；
- support+harmful；
- support-last/reordered；
- answerable vs unsupported；
- multi-evidence vs single-carrier。

每个数据集和 slice 分开报告，不把 NIAH、2Wiki、ASQA、QAMPARI pooling 成一个总分。

---

## 11. 统计方案

### 11.1 独立单位

- 主要独立单位是 query provenance component；
- 同一 query 的多个 contexts 是 repeated measurements；
- 三个 seeds 是同一方法的重复，不是三倍样本量；
- citation sentence/claim 嵌套在 query 内，bootstrap 必须整 query/component resample。

### 11.2 Family-level 估计

对每个 query 先计算三个 seed 的 paired delta 平均值，再按 component cluster bootstrap 10,000 次，seed=13，得到 family-level 95% CI。不能把 `seed × query` 展平成独立 rows。

同时逐 seed 报告点值、CI 和方向；不能只展示 family mean 隐藏某一 seed 的明显退化。

### 11.3 二元答案

- exact McNemar test；
- paired component-cluster bootstrap 95% CI；
- `wrong->right` 和 `right->wrong`；
- answer text same/different。

### 11.4 Citation

每个系统先在一个 query 内，根据自身实际生成的 sentence-citation pairs 计算 query-level precision/recall；不同系统的句子文本和数量可能不同，因此不能把“第几句”强行配对。主比较只在双方共同 answered queries 上配对 query-level 指标，并在 component 层整组 bootstrap，query 内的全部 sentence/claim rows 随 query 一起重采样。另报告各自完整 answered set 的 macro/micro aggregate、answered 数和 claims 数，避免回答数量变化被隐藏。

逐 sentence/claim 的 MiniCheck rows 保留为机制审计原始数据，不被当作独立样本扩大显著性。

MiniCheck 是外部 judge，TRUE 是被测 runtime verifier，TRUE 不评价自己。

### 11.5 多指标

主要门是 conjunctive gate：所有主要条件都必须满足，不用某个显著结果抵消另一个失败。Secondary slices 的 CI 作为解释性结果，不根据其显著性重新选择方法。

### 11.6 Power/MDE

R010 在训练前基于 G230 的 paired discordance 和 component 结构做 simulation-based sensitivity analysis，报告 NIAH 739、2Wiki 2,000 以及最终每个 300–400 题 held-out 能确认的最小效应。

当前预期边界是：最终 300–400 题只足以稳定确认约 3–6pp 的中等 paired effect；CI 跨 0 不能解释成等价或“已经证明没有退化”。如果预注册门小于样本可分辨范围，报告限制，不事后放宽门。

---

## 12. R400/R410 正式资格门

唯一候选必须同时通过以下条件，才可冻结为 `G*`。

### 12.1 NIAH 完整分布

相对 G0：

1. family-level `correct_and_cited` delta > 0，且 95% CI 下界 > 0；
2. answer correctness delta > 0，且 95% CI 下界不低于 0；
3. coverage 95% CI 下界不低于 -2pp；
4. citation precision 和 recall 的 95% CI 下界分别不低于 -2pp；
5. final empty 点值低于 G0，且没有 seed 比 G0 高超过 2pp；
6. runtime error、missing trace、gold-loaded-at-runtime 均为 0/false。

### 12.2 Context robustness

相对 G0：

- support+harmful 的 answer 或 `correct_and_cited` family delta 必须 > 0；
- support+benign 和 support-last 的 answer/coverage CI 下界不得低于 -2pp；
- support-only answer 不退化，final empty 仍低于 G0；
- unsupported ungrounded assertion 不高于 G0，answerable coverage 不得因此下降超过 2pp。

这里不再要求“每个 stress slice、每个 seed 都严格优于 GC”。那条 G230 门把“没有必要提高的 benign/position slice”也当成 superiority 目标。本路线改为：目标风险场景要求改善，普通场景要求非劣。

### 12.3 2Wiki 跨数据集资格

相对同输入 G0：

- answer、coverage、citation precision/recall 的 family CI 下界均不低于 -2pp；
- `correct_and_cited` 点值不低于 G0；
- multi-evidence 和 chain-complete strata 单独报告；
- 不用 2Wiki 的结果宣称 NIAH harmful Selector 有效。

### 12.4 已揭示 citation regression

ASQA 和 QAMPARI 分别报告：

- citation precision/recall；
- answered/coverage；
- claims per answer；
- empty/unsupported；
- 相对 G0 的 -2pp citation non-inferiority。

它们只承担 regression guard，不形成新的 held-out generalization 结论。

### 12.5 三 seed 规则

- family gate 通过；
- 三个 seed 的 answer delta 不得出现两个为负；
- 任一 seed 的 citation precision 或 recall 点值不得低于 G0 超过 4pp；
- 不允许只保留最好 seed 作为系统。

---

## 13. 阶段执行顺序

### R000：协议和输入冻结

**任务：**

- 绑定当前 Git commit、G230 archive、G0/GC/GM adapters、prompt/model/TRUE/MiniCheck revision；
- 写入所有 data IDs、source manifests、denylist 和 held-out hash；
- 冻结 R100 分类规则、R310 tie-break、R420 gate 和预算；
- 核对服务器 repo/runtime 实体，不运行新模型。

**通过门：** 所有输入存在且 hash 可复算；任何 held-out 逐题结果都没有被读取。

### R010：统计可分辨范围

**任务：** 使用现有 paired transitions 做 MDE sensitivity simulation，冻结主张强度和报告模板。

**通过门：** 代码、参数、cluster unit 和 simulation seed 可复算；不使用 observed power。

### R100/R110：citation 断点归因

**任务：**

- 对 G0/GC/GM 的共同 answered rows 重建 sentence -> declared citation -> split claim -> TRUE route -> final citation -> MiniCheck 全链；
- 对每个 regression 分配唯一“最早失败阶段”；
- 分层抽取自动退化、自动改善、declared-correct/final-wrong、final-correct/MiniCheck-wrong 样本做独立审计；
- 冻结默认 draft route 或条件 R120/R130。

**通过门：** 全部 regression rows 都进入 attribution table；无法可靠归因的 rows 比例 <5%；独立审计的抽样、双审和裁决均可复算。

### R120/R130：条件修复

只有 R110 路由到对应组件才执行。修复必须先通过 unit/integration tests，并让 G0、GR-F、GR-C 全部共享，不能成为某个训练臂的隐藏额外优势。

### R200/R210：数据构造与冻结

**任务：**

- 构造 NIAH、2Wiki answerable targets 和 train-only unsupported pairs；
- component/group split；
- target support、citation remap、token length、leakage、人工抽样审计；
- 生成 model-val G0 baseline；
- 冻结 train/model-val cases 和全部 hash。

**通过门：** 达到 unique-question 最低数；0 split leakage；0 invalid citation；0 truncation；target audit 通过。

### R300：实现和 smoke

**任务：**

- query-group batch/loss；
- citation token weighting；
- fresh 和 continuation load/save/reload；
- adapter 只在 draft call 生效；
- 训练前后 splitter/TRUE 参数 fingerprint 不变；
- 小样本 overfit 与 unsupported behavior smoke。

**通过门：** tests 通过；loss finite；citation tokens 和 answer tokens 均有非零 gradient；persisted adapter strict reload；无 OOM/truncation。

### R310/R320：便宜筛选并冻结唯一配方

只用 train/model-val 比较 GR-F/GR-C seed13，按第 9 节规则选唯一配方。结果和 tie-break trace 归档后冻结 qualification protocol。

### R330：三 seed 正式训练

按唯一配方训练 seeds 13/42/73。任何 seed 技术失败可按同配置重试一次；效果失败不能重训。

### R400/R410/R420：一次 locked qualification

三 seed 与 G0 在 NIAH qualification、2Wiki dev、ASQA/QAMPARI regression 上按预冻结任务运行。全部生成完成后才加载 gold 评分；MiniCheck 独立运行；最后统一计算 gates。

### R430：冻结或停止

- 全部门通过：安装唯一 `G*` manifest；
- 任一主要门失败：`NO CANDIDATE`，保留 G0，不在相同 qualification 上修补；
- 每个 claim 只按实际通过的数据和指标表述。

### R500：交回三模块路线

只有 R430 产生 `G*` 时：

- 写入第 02 路线 S300 所需的 adapter/prompt/splitter/TRUE/runtime hash；
- 解锁既有 S310 leave-one-out utility-label pilot；
- 不在本路线重新发明 Selector loss 或改变 F005 safety guard；
- 完整系统通过 S330/I410 后，才允许 I500 一次性 system held-out。

---

## 14. 最终 system held-out 边界

本路线不运行 system held-out。最终顺序固定为：

```text
R430 reliable G*
    -> existing S300/S310/S320/S330 Generator-aware Selector
    -> existing I400/I410 full-flow development
    -> freeze complete Retriever + Selector + Generator system
    -> I500 one-time HotpotQA / MuSiQue-Full / RGB
```

最终三个主数据集分别报告，不 pooling：

- HotpotQA：10 paragraphs、2 gold + 8 distractors，主要检查 multi-hop/distractor；
- MuSiQue-Full：answerable/unanswerable 配对，分别检查 multi-hop 和 honest abstention；
- RGB `en.json`：noise robustness/negative rejection；counterfactual 100 题只作次级分析。

最终结果无论通过或失败都报告；不得根据 held-out 结果再改变系统。

---

## 15. 运行、产物与 Git 纪律

### 15.1 每阶段必须保存

- human-readable protocol/report；
- machine-readable manifest；
- ordered input/output SHA256；
- per-query generation/score/citation rows；
- model/tokenizer/prompt/checkpoint identity；
- runtime command、environment、GPU、wall time、error count；
- `gold_loaded_at_runtime=false` 和 `reference_answers_loaded_at_runtime=false`；
- tests 和 independent verify-only 结果。

### 15.2 GitHub 同步

每个独立阶段完成后定向提交并推送。提交前：

1. fetch/核对 `origin/refactor/three-module-baseline`；
2. 如果团队有新提交，正常 pull/merge；
3. 不建立临时 worktree；
4. 不 reset/stash、删除或还原用户现有文件；
5. 只 stage 本阶段文件；
6. 测试/hash/报告一致后推送。

### 15.3 服务器

- repo：`/home/fl25387/projects/IBM_Granite_Project_latest`；
- runtime：`/scratch/fl25387/IBM_Granite_Project_latest`；
- 不 reset/stash 服务器已有 dirty state；
- 正式 run root 建议：`/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/GROUNDING-REPAIR-v1`；
- checkpoint 和大 generations 留在 runtime，Git 保存必要压缩产物、manifest、报告和 hash。

---

## 16. 主要风险与控制

| 风险 | 为什么重要 | 控制 |
|---|---|---|
| 继续模型继承错误 citation 习惯 | GM answer 高但 citation 已退化 | GR-F fresh 对照；model-val maximin；并列选 fresh |
| 只修 citation 后重新大量不答 | 可能牺牲已得到的 coverage | unsupported 限 10%–15%；answer/coverage/final-empty 联合门 |
| 多数据混合导致一个数据源压倒另一个 | 样本数不同会改变目标 | query-group equal weight；source 上限；比例预冻结 |
| 2Wiki target 不能逐事实可靠映射 | 错 target 会教坏 citation | minimal support、TRUE audit、人工抽样；失败排除而非放宽 |
| MiniCheck 误判 | 自动 judge 不完美 | R110 分层独立审计；TRUE 不自评；保留原始 rows |
| 重复使用 NIAH dev 造成研究过拟合 | decision-dev 已多次揭示 | 只在 R400 对唯一配方运行；方法在新 model-val 冻结 |
| 同时改多个组件无法归因 | 无法知道何者有效 | R110 单路径路由；所有臂共享条件修复 |
| 三 seed 被当成三倍样本 | 夸大置信度 | per-query seed average + component bootstrap |
| 最终 held-out 样本量不足 | 小效应 CI 可能跨 0 | R010 MDE；只声称可分辨效应；不以 observed power 辩护 |
| 为了“必须成功”放宽门 | 会制造不可靠结论 | gate 在训练前冻结；失败保留 G0；不在 qualification 上修补 |

---

## 17. 开始执行前检查表

- [ ] 用户确认本计划的目标和边界。
- [ ] R000 protocol 已从本文件快照为 frozen version。
- [ ] G230 所有输入、输出和 adapter hash 可复算。
- [ ] sealed600 与 system held-out denylist 生效。
- [ ] R100 分类和 R310/R420 判定代码有测试。
- [ ] NIAH/2Wiki train/model-val component/group overlap 为 0。
- [ ] 新 data target 经 automatic + stratified independent audit。
- [ ] GR-F/GR-C 只在 train/model-val 比较。
- [ ] qualification 前唯一配方、三 seed、统计和 run order 全部冻结。
- [ ] Retriever 与 F005 Selector 未被修改。
- [ ] GitHub/服务器当前实体核对完成，用户无关文件未被 stage。

只有以上检查全部完成，才允许启动正式 GPU 训练。
