# Gated Corroboration Selector — 设计规格

> **历史文档（已退役）:** 本路线已被实验判定不再继续。当前方案见
> [Beam Selector 实验计划](../../selector/BEAM_SELECTOR_EXPERIMENT_PLAN.md)，退役原因见
> [旧方法记录](../../selector/LEGACY_METHODS.md)。

**日期:** 2026-07-20

**状态:** 设计已过口头评审,待用户确认书面 spec

**模块:** Selector(模块二);不修改 Retriever(模块一)与 Generator(模块三)

**分支:** refactor/three-module-baseline

## 1. 目标与范围

在 Selector 模块内部,于重排之后新增一道"门"(gate):对被池内证据压倒的矛盾主张执行
硬剔除(drop),其余候选按重排顺序填充 budget,不足不硬凑。门与重排是互补关系——
重排负责把好证据往前排(容错),门负责把有害证据挡在窗口外(苛刻)。

适用范围:单答案事实题(single-answer factoid)。若问题合法地存在多个正确答案,
"答案不同"不构成矛盾,门必须关闭。本轮 NIAH 式任务按构造满足单答案假设。

## 2. 继承证据

本设计的每条决策都锚定在已验证的实验结论上:

1. **排除清单:** 基于相关性的重排(cross-encoder、listwise)在干扰环境下全部无效。
   → 门内不得出现任何 relevance-based 踢人信号;relevance 只参与排序与并列打破。
2. **corroboration +0.037(已认证):** 跨来源互证是目前唯一验证过携带增量的
   真伪信号。→ 门的主信号 = 池内互证票差。
3. **V1 选择器整轮验证:** 学习器组合信号后,排序质量与正确证据保留率显著提升,
   但 harmful-in-context 未降;72% 特征重要性塌回 relevance/rank;
   ContractNLI 迁移 −0.134 演示了绝对刻度跨域错位。
   → 本轮不用学习器做踢留决策(见 §13 非目标)。
4. **LLM-judge kappa 0.005–0.08:** 证据标签必须来自 official 标注与确定性规则,
   不得来自模型判断。→ 评估标签只用 qrels 与 counterfactual mutation log;
   LLM 只允许当抽取器(输出接受字符串投票校验),禁止当裁判。

## 3. 设计承诺

1. **凡是绝对判断都不许出现在踢人路径上。** 门内只有同一候选池内的相对比较
   (整数票差),没有任何跨题、跨域携带的分数刻度或阈值。
2. **凡是零票孤证都不许被踢。** 孤立假针与孤源真针观测上不可区分;误杀唯一真证据
   比放过一条毒严重(毒进窗口生成器仍可能答对;真证据被删则物理性无米下锅)。
3. **凡是标签都来自 official/deterministic provenance。** 沿用 V1 协议的
   Required Recall 非劣下界 −0.01 作为误杀保险丝。

## 4. 契约事实(不修改)

- `SelectionResult` 允许返回少于 `max_selected` 条(`contracts/models.py`
  只校验不超出与 rank 连续),"不足不硬凑"无需改接口。
- 接口不含 sufficiency / missing-fact / conflict status 字段
  (`docs/selector/README.md`)。"如实暴露冲突"的实现 = 把对峙双方都留在
  selection 里,让冲突以原文形式进入上下文;拒答是模块三的职责。
- 门决策记录不进冻结契约,走模块内部可观测通道(§11)。

## 5. 架构与数据流

```
CandidateSet (top-k, 来自 Retriever)
  ↓ 取 rerank 窗口 top_n(窗口外候选不做答案抽取,原序尾随)
  ↓ 答案抽取(复用 CorroborationSelector 的 EXTRACT_PROMPT,一次抽取两段共用)
  ↓ 答案归一化(升级版,含数值规范化)→ 按归一化答案聚簇
  ↓ 独立票统计:同 document_id 只算一票
  ├─ 重排段(容错): blended = α·relevance + (1−α)·corroboration   ← 现有逻辑
  ├─ 门段(苛刻): 按 §7 四条件判定 drop
  ↓ 幸存者按 blended 降序 + 窗口外尾随候选,截断至 max_selected
SelectionResult(可少于 max_selected)
```

门在重排之后、截断之前执行。drop 腾出的 budget 槽位由更低排位的幸存者填充——
这是 drop 相对 demote 的核心收益:毒出窗口的同时,替补进窗口。
被 drop 的候选不得从尾随段重新进入。

## 6. 共享底座

### 6.1 答案抽取

复用现有 `EXTRACT_PROMPT` 机制与既有缓存/确定性约定。门不产生新增 LLM 调用:
重排段已抽取的答案原样供门段使用。

### 6.2 答案归一化(新增,`selector/answer_norm.py`)

现有 `normalize_answer` 只做小写、冠词剥离、标点修剪。升级为确定性数值规范化:

- 千分位与空格:"1,200" ≡ "1200";
- 货币符号剥离:"$1.2B" 的数值部分参与比较;
- 数量级词:"1.2 billion" ≡ "1,200 million" ≡ 1.2e9;"M/B/k" 后缀同理;
- 百分号:"18%" ≡ "18 percent";
- 简单日期格式归一(有限白名单,不做自然语言日期解析)。

规范化后按精确相等比较,不引入模糊匹配。全部规则确定性、可单测。
该升级同时惠及已上线的重排段投票(缓解"财报数字投票碎裂"对排序的伤害)。

### 6.3 独立票

`independent_support(K)` = 簇 K 内候选去重后的 **distinct document_id 数**。
同一文档的多个 chunk 只算一票。跨文档同父来源(source_uri 级 SAME_SOURCE)
留作升级路径(§13),v1 不做。

parametric vote(模型自答)**不计入**门的独立票(§8 信号毕业制),
仅保留在重排段的 corroboration 分数中(现状不变)。

## 7. 门规则(冻结)

设候选 c 属于答案簇 K,竞争簇 K′ = 池内与 K 归一化答案不同的簇中
`independent_support` 最大者。**drop c 当且仅当同时满足:**

1. c 抽出了有效答案(`is_valid_answer` 为真)。抽不出答案的候选永不被门踢
   ——支持性细节文档(不含字面答案但有用)长得像"无关",绝对判"无关"又是
   一道判断题,故无关项一律交给重排+截断沉底,不进门。**[已决策]**
2. 存在竞争簇 K′(单答案假设下,不同答案即矛盾,无需 NLI)。
3. `independent_support(K′) − independent_support(K) ≥ margin`,
   margin 为整数票差,默认 2,仅在 dev 上扫描后冻结。
4. `independent_support(K) ≤ support_cap`,默认 1(保守挡板,默认开启)
   ——只剔"孤立且被压倒"签名,进一步收窄误杀面。**[已决策]**

### 7.1 性质(设计验收标准)

- **免校准:** 门内量全部是同池整数票;打分系统整体偏移不改变票差。
- **孤源真针结构性安全:** 无竞争簇 → 条件 2 假 → 不可能被踢(非调参结果)。
- **对峙如实暴露:** 1 比 1 → 条件 3 假 → 双方都留,冲突原文进上下文。
- **失效方向 = 闭嘴:** 投票碎裂域(如财报数字)所有簇票数同降 → 票差缩小 →
  门条件不满足 → 自动退化为纯重排。信号坏掉时门静默,不乱杀。

### 7.2 逐条测试映射

每条性质必须有对应单测:孤源真针不被踢;1v1 双留;3v1 踢;2v1(margin=2)不踢;
2v4(support_cap=1)不踢;归一化后等值答案合簇;同文档多 chunk 合票;
全员无答案时门零动作;drop 后替补进入 budget;幸存者不足时不硬凑。

## 8. 信号毕业制

未验证信号(来源权威性、时间新旧、语料一致性、模型先验、确定性硬约束)
只允许进入重排段影响排序;经逐域验证后方可"毕业"进入门段影响踢留。

- 现有 parametric vote 属于"模型先验",据此**留在重排、禁入门内**。
- 未来信号毕业的形式 = 修改独立票的权重函数,不改门的四条件结构。

该机制把"哪些特征在哪个域是真信号需要逐域测量"固化为架构约束:
排序容错,故可试;过滤苛刻,故须先证。

## 9. 失效方向分析(诚实边界)

**独立转抄的错误信息**(多个真独立的错源 vs 单一真源)会反转互证信号,
本门会踢错。document_id 去重只能拦截共享出处的转抄;真独立的错误来源是
corroboration 一族的原理性边界。出路只有两条:权威/时间信号完成逐域验证后
毕业进门,或维持保守 margin 让门在此局面下闭嘴。本设计不假装解决它,
报告中如实陈述。

## 10. 组件与文件清单

| 文件 | 内容 | 测试 |
|---|---|---|
| `src/evidence_rag/selector/answer_norm.py` | 确定性数值/日期归一化 | `tests/selector/test_answer_norm.py` |
| `src/evidence_rag/selector/gated.py` | `GatedCorroborationSelector`:共享底座 + 重排 + 门 + 截断 | `tests/selector/test_gated.py`(§7.2 全表) |
| `src/evidence_rag/composition.py` | `build_selector` 注册 `gated-corroboration`,参数 `alpha`/`margin`/`support_cap`/`top_n`,按 `_bm25_parameters` 模式校验 | swap-matrix 补行 |
| `src/evidence_rag/selector/corroboration.py` | 抽取/归一化底座抽出共享(重排行为不变) | 现有测试保持绿 |

实现顺序:answer_norm → 底座抽出 → gated → 注册。全程 TDD。

## 11. 可观测性

每次门判定产生结构化记录:候选 id、所属簇答案、own votes、winner 簇答案、
winner votes、票差、四条件逐条布尔值、最终动作(keep/drop)。

记录通过构造时注入的 sink(如 `on_gate_decision` 回调或 decisions 列表)暴露,
供评估 harness 读取;**不进入冻结的 `SelectionResult`/`PipelineRun` 契约**。
若团队后续决定纳入 trace,按 schema_version 升级流程另行走契约变更。

## 12. 评估协议(骨架)

- **主对照:** 同一重排配置下 gate-on vs gate-off 的配对比较,单独隔离门的因果贡献。
- **双 Gate(沿用 V1 协议):** harmful-in-context 显著下降(标签来自
  deterministic counterfactual mutation log)+ Required Recall 非劣下界 −0.01。
- **dev-only 扫参后冻结:** margin ∈ {1,2,3}、support_cap ∈ {1,2,off}、
  归一化开关、去重开关;测试集一次性运行全部冻结配置。
- **迁移测试内置:** 在域 1 dev 上调参冻结,原样跑其他域——把 ContractNLI −0.134
  的事故形态变为设计内必测项。
- **oracle-drop 上界:** 按标注精确踢除 harmful 项,给出门的理论天花板。
- **边/簇检测组件评估(2026-07-20 增补,导师要求):** 门的判据链条每一环单独量,
  end-to-end 不够。
  - **false-conflict rate:** 同池内两条均含 gold 答案(按 official aliases 判定)的候选
    被归一化聚进不同答案簇的比率——假冲突即门误杀的机制通道,是首要监控量;
  - **missed-conflict rate:** 含 gold 候选与含 deterministic counterfactual 值的候选
    被聚进同一簇的比率——漏判冲突即门失明;
  - **dedup 确定性单测:** document_id 合票规则要求 100% 通过(对应 Graph 2.0
    Gate 0B 的 SAME_SOURCE exact-rule tests);
  - **标签来源:** gold aliases 为 official 标签 + counterfactual mutation log,
    零新增人工标注——与 ArbGraph(arXiv 2604.18362)Table 4 用 200 对人工标注做
    组件评估形成方法学对照:可复现、无标注者偏差,呼应 judge-kappa≈0 发现。
- **统计单位 query,配对检验**,沿用项目现行显著性协议。
- 数据集选择遵循 `docs/selector/README.md` 六条待确认项,由团队会议决定,
  本 spec 不硬编码;FinanceBench 按团队约定排除于 Selector 工作之外。

## 13. 非目标(v1 明确不做)

- 不引入 NLI 判矛盾(单答案假设下不同答案即矛盾;NLI 及 CLAIM_REFUTES 边
  属 Graph 2.0 升级路径,且须先过其自身的 Relation Gate);
- 不做学习型门(V1 证据反对;若未来做,特征限池内相对量,且须重批数据协议);
- 不做跨文档 SAME_SOURCE(v1 只做 document_id 去重);
- 不做多答案/列表题支持(门在该任务形态下必须关闭);
- 不修改三模块契约与 Generator 拒答逻辑。

Graph 2.0（旧训练计划已删除，见 [旧方法记录](../../selector/LEGACY_METHODS.md)）与本设计的关系:不是竞争者,
是独立票底座的升级路径——SAME_SOURCE/CLAIM_REFUTES 边替换"怎么数票",
不替换门的四条件结构。

## 14. 待定项

1. dev/test 数据集与域划分(团队会议,README 六条);
2. margin 与 support_cap 的最终数值(dev 扫描后冻结);
3. 门决策记录是否升级进 PipelineRun trace(契约变更,另行走流程)。
