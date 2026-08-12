# Selector–Generator 跨阶段桥接：实验与改造计划 v2

**问题：** 为什么 Selector 已经能保守删除一部分有害证据，但这种改善没有自动转化成更好的最终答案？
**研究主线：** Reliability across the full evidence flow。
**方法目标：** 让 Selector 的可靠性信息真正被 Generator 使用，而不是三个模块只在形式上串联。
**状态：** 计划已写明，尚未执行；第一阶段结果出来前，不冻结第二阶段的最终方法。运行时信号边界按 [`AMENDMENT_RUNTIME_SIGNAL_BOUNDARY_v2.md`](AMENDMENT_RUNTIME_SIGNAL_BOUNDARY_v2.md) 执行。

---

## 1. 一句话结论

执行顺序固定为：

```text
第一阶段：现有三模块联合并诊断
        ↓
第二阶段：按诊断结果给现有 Generator 增加最小跨阶段机制
        ↓
第三阶段：只有前两阶段仍显示上下文适应问题时，才做鲁棒训练
```

第二阶段不是第四个模块，而是改变 Selector 与 Generator 之间原来“只传文本”的接口，让 Generator 知道哪些保留证据更应该优先读取、哪些保留证据仍有风险。

---

## 2. 当前已经确定的事实

### 2.1 Selector 已经完成了什么

最新 Selector 在 1739 个问题上：

- harmful evidence reduction：约 13%；
- deletion precision：75%–81%；
- 标注必需证据 recall 损失：0；
- 标注多跳链损失：0；
- Seed 13 改动了 109 个问题。

但是搭配最基础 `GraniteGenerator` 后，109 个变化问题中出现 4 个错转对、6 个对转错，最终答案没有提高。

### 2.2 已完成的高级 Generator 做到了什么

现有 `VerifyAnnotateGenerator` 已实现：

```text
生成草稿 → 拆分事实 → 逐条查证 →
支持的事实附可靠引用；暂时无法查证的事实保留并标记
```

它在 ASQA 上基本保持答案正确率，同时显著提高引用精度和引用召回；QAMPARI 上引用改善得到复现。它尚未与最新 Selector 做正式联合。

### 2.3 L003 为什么不能当作这次联合实验

L003 使用的是基础 `GraniteGenerator`，并固定 `max_new_tokens=32`。它没有运行现有 `VerifyAnnotateGenerator` 的草稿、claim splitting、NLI 核验和逐句引用路径。

所以 L003 证明的是：

> 单纯清理证据，不足以保证最基础的一次生成稳定变好。

它没有证明：

> 最新 Selector 与已完成的高级 Generator 联合后仍然无效。

---

## 3. 两个研究主张

| 主张 | 含义 | 最低可信证据 |
|---|---|---|
| C1：局部可靠性不会自动组合 | Selector 的证据改善可能被 Generator 的上下文敏感性抵消 | 四臂联合实验显示模块主效应和 Selector×Generator 交互，并有逐题错误类型证据 |
| C2：跨阶段指导可以帮助组合 | Generator 使用 Selector 的保护/风险信号后，更完整地读取正确证据，同时不把高风险证据误当成唯一依据 | 相比“只有关键事实笔记”和“只给清理文本”，答案改善更好且引用可靠性不退化 |

需要排除的简单解释：

- 改善只是因为给 Generator 多写了一段提示词；
- 改善只是因为生成 token 更多；
- Generator 偷看了 gold answer、required evidence 或 gold chain；
- 删除的有害证据文本被重新放回 Generator；
- 结果只是一次 GPU 运行漂移。

---

## 4. 第一阶段：联合现有三模块

### 4.1 目的

先回答：

> 不增加任何新方法，只把最新 Selector 与已经完成的高级 Generator 连接起来，系统结果是什么？

### 4.2 四个比较组

所有比较组在同一个作业、同一个模型实例和相同解码配置下运行，避免项目已经测到的跨作业生成漂移。

| 组 | Selector 输入 | Generator | 回答的问题 |
|---|---|---|---|
| A | TopK10 | 基础 Granite | 原始基线 |
| B | 最新 Selector | 基础 Granite | Selector 单独造成什么变化 |
| C | TopK10 | Verify-and-annotate | 高级 Generator 单独造成什么变化 |
| D | 最新 Selector | Verify-and-annotate | 两者联合能否产生额外价值 |

核心不是只看 D 是否最高，还要计算：

```text
基础 Generator 中的 Selector 效果 = B − A
高级 Generator 中的 Selector 效果 = D − C
跨阶段交互 = (D − C) − (B − A)
```

零基础理解：如果换成高级 Generator 后，Selector 从“净伤害”变成“净帮助”，说明高级 Generator 确实接住了 Selector 的改善。

### 4.3 数据范围

第一轮优先运行 NIAH：

- 全部 NIAH 问题用于总体答案与覆盖率；
- Selector 实际改变上下文的 109 题用于主要诊断；
- 上下文完全相同的问题在同一 Generator 内复用同一结果，不重复生成；
- 2Wiki 本轮 Selector 删除数为 0，不能估计 Selector×Generator 交互，只作为 Generator-only 控制或后续验证。

已经查看过结果的 `decision-dev` 从现在起只能作为诊断/开发数据，不能再次包装成新方法的最终盲测。新方法的最终确认必须使用此前未用于选择方法的数据。

### 4.4 记录的指标

主指标：

- 最终答案 match；
- `wrong→right` 与 `right→wrong` 数量；
- 回答覆盖率和空答案率；
- 逐句 citation precision / recall；
- 每个答案事实能否追溯到 selected evidence 和原始来源。

诊断指标：

- 日期、月份、地点、实体全名、数量、条件是否丢失；
- claim splitter / JSON / parser 是否失败；
- 正确证据仍在但 Generator 没有使用的比例；
- 无支持内容是被引用、被标记，还是被错误删除；
- 单跳与多跳问题分别发生什么。

### 4.5 第一阶段不设置“强行通过门”

这是诊断实验。无论结果正负都进入错误分类，但第二阶段只能针对占主导、且有证据支持的失败原因。

| 第一阶段主要发现 | 第二阶段动作 |
|---|---|
| D 已经稳定优于 C，且没有新的明显错误 | 不堆功能；保留现有高级 Generator，直接准备独立确认 |
| 正确证据仍在，但日期/地点/实体等细节遗漏 | 启用“问题需求槽位 + 保护证据优先”的关键事实笔记 |
| 主要错误来自仍被保留的高风险证据 | 启用风险感知的引用/核验规则 |
| 主要错误来自 claim splitter、JSON 或空输出 | 只修可靠解析与 fallback；不把工程修复包装成方法贡献 |
| 真实所需证据被 Selector 删除 | 返回 Selector 修正；Generator 不负责猜回被删事实 |
| 多跳所需事实都在，但无法组合 | 才考虑运行时集合充分性/链准备度估计 |

---

## 5. 第二阶段：最小跨阶段机制

### 5.1 它属于哪里

它属于 Generator 的输入与生成策略调整：

```text
原来：Selector → 只传保留后的文本 → Generator

调整：Selector → 保留文本 + 选择信号 → 同一个 Generator
```

系统仍然只有 Retriever、Selector、Generator 三个模块。

### 5.2 当前代码的真实缺口

目前：

- `RiskControlledSelector.select_with_trace()` 已经产生每条候选的 `protect_score`、`harm_score`、动作和原因；
- 标准 `SelectedEvidenceSet` 只包含保留下来的证据文本；
- 标准 pipeline 调用 `selector.select()`，因此把 decision trace 丢掉；
- Generator 目前看不到 Selector 为什么保留或删除一条证据。

第二阶段的第一项改造只是补上这个信息接口，不改变 Selector 的删除结果。

### 5.3 最小接口：只承载 Selector 原生信号

增加一个可选的 `SelectionGuidance` sidecar。它只包含 Selector 在真实运行时已经产生的信息：

```text
query_id
selector_changed
dropped_count
每条保留证据的：
    evidence_id
    retrieval_rank
    protect signal
    harm signal
    keep / abstain-keep 原因
```

它不包含 Generator 后续自行估计的状态，也不包含：

- gold answer；
- official required evidence 标签；
- gold multi-hop chain 状态；
- 被删除证据的文本；
- counterfactual/mutation 的评测身份。

旧 Selector 或普通 TopK 没有 guidance 时，Generator 必须完全退回当前行为。这样接口是可选增强，不会破坏已有模块。

### 5.4 三类信息严格分离

| 类别 | 来源 | 运行时能否使用 | 作用 |
|---|---|---:|---|
| Selector 原生信号 | Selector 模型和动作规则 | 能 | 指导优先读取与风险处理 |
| Generator 运行时估计 | 问题 + 当前 selected evidence | 能 | 估计当前证据是否足够回答 |
| gold chain / required evidence | benchmark 标准答案 | 不能 | 只能在生成后评测 |

L003 报告中的 `chain loss=0` 是利用 official gold supporting evidence 计算的评测结果。真实用户提问时没有这个标准答案，因此不能把它传给 Generator，也不能把它写入 `SelectionGuidance`。

如果第一阶段显示主要问题确实是多跳组合，再由 Generator 根据“问题需要哪些信息槽位，以及每个槽位是否找到低风险支持”估计：

- `READY`：当前问题需求均找到可核验支持；
- `PARTIAL`：仍有需求没有找到支持；
- `CONFLICTED`：同一需求出现无法解决的冲突；
- `UNKNOWN`：估计器自己无法可靠判断，回退到当前 Generator 行为。

这个运行时状态正式命名为 `EvidenceReadiness`，只能由问题和当前 `SelectedEvidenceSet` 计算，不能读取 gold chain。它是可能出错的模型估计，不是真实链标签，也没有恢复被删证据或直接删除答案内容的权限。完整规则见 v2 修订。

### 5.5 Generator 如何使用信号

候选规则如下，最终启用哪些由第一阶段决定：

| Selector 信号 | Generator 行为 |
|---|---|
| 高 protect、低 harm | 优先提取姓名、完整日期、地点、数量和条件；优先作为答案依据 |
| 高 harm、低 protect 但因 cap 等原因仍保留 | 不作为唯一依据；需要低风险证据 corroboration，否则标记未核验 |
| protect 与 harm 同时高 | 不删除，但视为争议信息；生成时保守表达并要求第二证据 |
| 普通保留 | 按现有 Verify-and-annotate 路径处理 |
| 已删除 | 不把文本重新交给 Generator，也不能引用 |

这里“高/低”首先使用开发数据上冻结的相对等级或动作原因；在证明分数校准前，不把 sigmoid 数值宣传成真实概率。

### 5.6 关键事实笔记

如果第一阶段确认主要问题是答案细节遗漏，Generator 在写草稿前先形成内部表：

```text
问题需要的字段 | 找到的完整值 | 支持证据 | Selector 风险状态
```

例如：

```text
对象：The Dark Side of the Moon | evidence-2 | protected
完整发布日期：1 March 1973    | evidence-2 | protected
```

然后才生成最终答案，继续使用现有 claim splitting 和 Verify-and-annotate。

问题需求只能从问题文本识别为有限类型，例如 `PERSON / DATE / YEAR / LOCATION / NUMBER / TITLE / COMPARISON / MULTI_HOP_BRIDGE`。不复用已经失败的自由生成 checklist，因为它曾猜测答案、增加无关要求，并在历史验证中未通过门。

### 5.7 第二阶段最小消融

| 组 | 方法 | 作用 |
|---|---|---|
| G0 | Selected evidence + 当前 Verify-and-annotate | 直接基线 |
| G1 | G0 + 普通关键事实笔记，不看 Selector 信号 | 判断收益是否只是 Chain-of-Note 式多读一步 |
| G2 | G0 + 带 protect/harm 标签的关键事实笔记 | 判断真正的跨阶段信号是否有额外价值 |

只有 `G2 > G1`，才能声称 Selector→Generator 的跨阶段信号有作用；如果 G1 与 G2 相同，应采用更简单的 G1，并诚实地把它写成 Generator 改进，而不是跨阶段创新。

### 5.8 第二阶段开发成功条件

在已揭示的诊断/开发数据上：

- 相对 G0，`wrong→right` 多于 `right→wrong`；
- 总体答案点估计相对 TopK 为正；
- 原有 4 个正向 Selector 案例不能被大规模破坏；
- 空答案和日期/地点/实体遗漏明显减少；
- citation precision / recall 不发生有实际意义的下降；
- G2 必须优于 G1，才保留跨阶段信号。

开发数据只决定是否值得进入最终确认，不用于宣布正式提升。最终若声称 superiority，仍需在未参与选择的数据上报告配对置信区间。

---

## 6. 第三阶段：条件触发的鲁棒训练

### 6.1 不是默认步骤

只有同时满足下面两点才启动：

1. 第一、二阶段证明正确证据仍在，但同一问题在 TopK 与 Selected context 下仍经常给出不一致答案；
2. 关键事实笔记或风险指导出现正向案例，说明存在可以学习的信号，但纯提示还不稳定。

如果问题是 Selector 真删错证据、评测字段错误或解析器失败，不应通过训练 Generator 掩盖。

### 6.2 训练数据

参考 Yoran et al. ICLR 2024，只使用训练区构造约 1000 组：

```text
同一道问题 + TopK10 context       → 同一个完整目标答案
同一道问题 + Selector context     → 同一个完整目标答案
相关证据 + 受控无关/冲突证据混合 → 仍输出完整且有依据的答案
```

训练目标不是让模型记答案，而是让它学会：只要所需正确证据仍在，周围增加或删除少量材料都不应丢失日期、实体、地点和条件。

优先使用 LoRA/PEFT，不全量训练 Granite。训练数据、开发数据与最终确认数据严格分离。

### 6.3 最小比较

| 组 | 用途 |
|---|---|
| 未训练的 G2 | 训练前基线 |
| 只用普通 clean context 训练 | 排除收益只是一般微调 |
| relevant/irrelevant 混合 context 训练 | 检验真正的上下文鲁棒性 |

若混合训练不优于普通训练，第三阶段方法主张不成立，保留第二阶段最简单的通过版本。

---

## 7. 执行顺序与停止规则

| Milestone | 工作 | 是否改代码 | 结果决定什么 |
|---|---|---:|---|
| F000 | 冻结联合实验输入、四臂配置和同作业运行方式 | 很少 | 防止不同模型配置造成假差异 |
| F001 | 现有四臂联合实验 | 只接线 | 判断高级 Generator 是否已经接住 Selector |
| F002 | 109 个变化题错误分类 | 否 | 决定第二阶段只修哪一种主要错误 |
| F003A | 传递并只记录可选 `SelectionGuidance` | 是，小 | 让 Selector 原生信号不再在模块边界丢失，先不改变答案 |
| F003B | 条件实现问题需求与关键事实笔记 | 是，小 | 只在细节遗漏为主要错误时启用 |
| F003C | 条件实现 `EvidenceReadiness` | 是，中 | 只在多跳事实齐全但未组合时启用 |
| F004 | G0/G1/G2 小规模开发消融 | 是 | 决定 notes 和跨阶段信号是否保留 |
| F005 | 独立数据确认 | 配置为主 | 形成正式端到端结论 |
| F006 | 条件触发约 1000 条鲁棒训练 | 是，中 | 只在提示级方案不足时运行 |

停止规则：

- F001 已经解决问题：不执行不必要的 F003A–F006；
- F002 显示是工程解析问题：只修解析，不上跨阶段机制；
- G2 不优于 G1：删除风险信号路径，保留更简单方案；
- G1/G2 都无正向迹象：不启动鲁棒训练，先重查问题归因；
- 训练没有优于普通 clean-context 微调：不保留混合训练主张。

---

## 8. 预计资源

以下仅为排期量级，执行前以服务器 smoke 的真实吞吐修正：

| 阶段 | 预计资源 |
|---|---|
| F000–F002 | 约 3–6 GPU 小时；已有模型和数据可复用 |
| F003A/B | 主要是本地实现与测试，无正式 GPU 训练 |
| F003C | 条件功能；需增加运行时估计和无 gold 泄漏测试 |
| F004 | 约 2–5 GPU 小时，取决于关键事实笔记是否增加一次 Granite 调用 |
| F005 | 取决于最终数据规模，需一次冻结运行 |
| F006 | 约 2–8 GPU 小时的 3B LoRA 初始预算；只有条件满足才申请 |

最大成本不是 Selector，而是高级 Generator 的 claim splitting 与 NLI 逐条核验。运行时优先复用相同上下文结果，并在同一作业内完成配对比较。

---

## 9. 与论文故事的关系

如果第二阶段成立，最强但不过度的论文表述是：

> RAG 各阶段的局部可靠性不会自动组成端到端可靠性。我们让保守 Selector 的证据风险信息跨过模块边界，指导 Generator 进行风险感知的事实提取与逐句核验，从而使证据层改善更可靠地传递到最终答案。

如果 G2 不优于 G1，则降级为：

> 对经过保守筛选的证据进行结构化事实读取，可以缓解 Generator 对上下文删减的敏感性。

不能声称：

- 三个模块分别有效，因此完整系统必然有效；
- gold chain 被用于运行时指导；
- 简单组合多篇论文的方法本身就是新算法；
- 在独立最终确认前，系统已经优于 TopK。

---

## 10. 当前决定

现在只批准到“计划与接口设计”层面。正确的下一项执行是 F000/F001：现有四臂联合实验。

第二阶段的最终实现必须等待 F001/F002；第三阶段保持 `CONDITIONAL / NOT AUTHORISED`。

---

## 11. 方法依据

- [Making Retrieval-Augmented Language Models Robust to Irrelevant Context（ICLR 2024）](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8011b23e1dc3f57e1b6211ccad498919-Abstract-Conference.html)：支持在相关/无关 context 混合条件下训练 Generator；本计划只在第三阶段条件触发。
- [Chain-of-Note（EMNLP 2024）](https://aclanthology.org/2024.emnlp-main.813/)：支持先形成逐证据阅读笔记，再生成答案；本计划用 G1/G2 区分普通笔记收益与跨阶段信号收益。
- [R²AG（Findings of EMNLP 2024）](https://aclanthology.org/2024.findings-emnlp.678/)：支持检索阶段的信息可以帮助 Generator 理解证据；本计划不复现其 R²-Former，只借鉴“不要在模块边界丢掉有用信号”的原则。
- [Self-RAG（ICLR 2024）](https://proceedings.iclr.cc/paper_files/paper/2024/hash/25f7be9694d7b32d5cc670927b8091e1-Abstract-Conference.html)：支持生成与事实核验结合；本项目已有 Verify-and-annotate 实现，因此不重建完整 Self-RAG。

这些文献为设计提供依据，但不能把方法组合本身自动视为创新；G2 必须通过相对 G1 的消融，才能支持跨阶段机制的独立作用。
