# v2 修订：运行时信号、模型估计与评测标签的严格边界

**日期：** 2026-08-12
**作用：** 修订 `PLAN.md` 第二阶段中“证据链状态”的来源和实现顺序。
**不改变：** F000/F001 先联合现有系统、F002 再诊断、F006 训练保持条件触发。
**当前状态：** `PLANNED / NOT IMPLEMENTED`

## 1. 为什么需要修订

“Generator 根据证据链状态决定如何回答”容易产生误解，因为项目中有两种完全不同的“链状态”：

1. 实验评分时，根据 official supporting evidence 计算的 gold chain 状态；
2. 真实运行时，仅根据问题和当前可见证据做出的模型估计。

第一种很准确，但真实用户提问时不存在，不能作为模型输入。第二种可以运行，但只是估计，可能判断错误，不能写成已知事实。

因此，后续代码和报告不再把二者统称为 `chain_status`。

## 2. 三类信息必须严格分开

| 信息层 | 产生者 | 真实运行时可用 | 允许用途 | 示例 |
|---|---|---:|---|---|
| A. Selector 原生信号 | Selector 模型和动作规则 | 是 | 传给 Generator 作为阅读指导 | protect/harm signal、KEEP/ABSTAIN_KEEP、retrieval rank、是否发生删除 |
| B. Generator 运行时估计 | 问题 + 当前 selected evidence | 是 | 判断回答所需信息是否已找到；只能指导生成、核验或保守表达 | answer requirement、evidence support map、evidence readiness |
| C. 评测标签 | benchmark gold/reference/provenance | 否 | 生成完成后评分、分析估计是否准确 | gold answer、official required/supporting IDs、gold complete chain、chain loss、mutation identity |

硬规则：C 层任何字段不得进入 A/B 层对象、prompt、模型输入、缓存键或运行时决策。

## 3. Selector 真正可以传什么

`SelectionGuidance` 只允许包含运行时已经由 Selector 产生的信息：

```text
query_id
selector_changed
dropped_count
每条保留证据：
    evidence_id
    retrieval_rank
    protect_signal
    harm_signal
    action_reason
```

注意：

- protect/harm 是模型分数，不是 gold 标签，也暂不称为真实概率；
- 被删除证据的文本不重新交给 Generator；
- 不传 official required/supporting 身份；
- 不传 `chain_loss` 或“gold chain 完整”。

## 4. Generator 可以自己估计什么

只有 F002 显示“正确证据都在，但 Generator 没有完整读取或组合”是主要问题时，才增加运行时估计。

运行时过程为：

```text
问题文本
  ↓
识别回答需要的信息类型
  ↓
只在当前 selected evidence 中寻找支持
  ↓
形成 EvidenceReadiness（证据准备度）
  ↓
指导现有 Verify-and-annotate Generator
```

### 4.1 问题需求

优先使用有限、问题文本可确定的类型：

```text
PERSON / DATE / YEAR / LOCATION / NUMBER /
TITLE / COMPARISON / MULTI_HOP_BRIDGE
```

它只描述“需要什么类型的信息”，不能猜测具体答案。

### 4.2 当前证据支持图

对每个需求，记录：

```text
requirement_type
candidate_value
supporting_selected_evidence_ids
selector_signal_summary
verifier_outcome
```

只允许引用 `SelectedEvidenceSet` 中仍然可见的证据。

### 4.3 证据准备度

运行时状态命名为 `EvidenceReadiness`，不再叫 gold chain status：

| 状态 | 运行时含义 | Generator 行为 |
|---|---|---|
| `READY` | 当前问题需求均找到可核验支持 | 生成完整答案并逐句核验 |
| `PARTIAL` | 至少一个需求未找到支持 | 只回答已支持部分，明确缺口 |
| `CONFLICTED` | 同一需求存在无法解决的冲突 | 保守表达或标记需要复核 |
| `UNKNOWN` | 估计器本身无法可靠判断 | 回退到当前 Verify-and-annotate 行为，不作强制决定 |

`EvidenceReadiness` 不是事实真值，也没有删除证据或恢复已删除文本的权限。

## 5. 修改后的实施顺序

第二阶段拆为三个小步骤，避免一次实现过多功能：

| Run | 内容 | 触发条件 | 是否改变答案行为 |
|---|---|---|---:|
| F003A | 把 Selector trace 转成可选 `SelectionGuidance` 并传到 Generator 边界 | F002 证明 Selector 信号值得分析 | 否，先只记录 |
| F003B | 问题需求 + 关键事实笔记 | 细节遗漏/空答案为主要错误 | 是 |
| F003C | `EvidenceReadiness` 估计 | 多跳事实齐全但没有组合为主要错误 | 是；条件功能 |

F003C 不是 F003A/B 的必经步骤。单跳细节遗漏可以只做 F003B。

## 6. 修改后的消融

基本消融仍为：

| 组 | 方法 | 回答的问题 |
|---|---|---|
| G0 | 当前 Verify-and-annotate | 直接基线 |
| G1 | 问题需求 + 普通关键事实笔记 | 多读一步本身是否有效 |
| G2 | G1 + Selector protect/harm 指导 | 跨阶段信号是否有额外作用 |

只有 F002 证明多跳组合是主要问题时，再加：

| 组 | 方法 | 回答的问题 |
|---|---|---|
| G3 | G2 + EvidenceReadiness | 集合准备度估计是否带来额外价值 |

G3 不优于 G2 就删除 F003C，不保留额外复杂度。

## 7. 防止 gold 泄漏的实现检查

F003 开始前必须增加：

1. runtime schema 使用字段 allowlist，不能接受 `gold_answer`、`reference_answers`、`required_evidence_ids`、`supporting_facts`、`chain_loss` 或 mutation 身份；
2. 测试确认 prompt 和序列化 runtime payload 中不存在这些字段；
3. 评测器在 Generator 完成输出后才读取 gold 数据；
4. 使用 gold chain 计算的结果只能出现在 evaluation artifact，不能进入 generation trace；
5. 没有 `SelectionGuidance` 或估计失败时，行为必须回退到当前 Verify-and-annotate。

## 8. 对当前执行决定的影响

下一步仍然是 F000/F001，完全不需要先实现运行时估计。

F001/F002 之后：

- 如果高级 Generator 已经接住 Selector：直接准备独立确认；
- 如果主要是日期、实体、地点遗漏：只做 F003A/F003B；
- 如果主要是多跳组合失败：才增加 F003C；
- 如果 Selector 真删了必要事实：返回 Selector，不让 Generator 猜测或使用被删文本；
- 如果只是解析失败：修解析，不启用 EvidenceReadiness。
