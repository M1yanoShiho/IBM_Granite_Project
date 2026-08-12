# 自适应保守 Selector 执行计划（v2）

**问题：** 如何在默认 Hybrid RRF Top20 → TopK10 的基础上，可靠删除少量错误证据，同时基本保留正确证据和多跳证据链。

**方法主张：** Selector 默认不行动，只对“高 harmful、低 protect”的极高置信候选逐条删除；当前测试 `0–cap` 的保守策略族，`cap∈{1,2,3}`，最终 cap 在 calibration 前冻结；风险控制无法支持非零动作时自动退回 TopK10。

**冻结日期：** 2026-08-11

**版本：** v2；替代 v1 作为当前执行版本，但保留 v1 供审计。

**代码快照：** branch `refactor/three-module-baseline`，检查时 HEAD `74026c0`；R001 执行时必须重新记录实际 commit。

**所有新实验状态：** `TODO`。本文只定义实验，不预填结果。

---

## 0. 零基础版结论

### 0.1 我们现在不追求“删掉所有错误证据”

假设 TopK10 中有 8 条正确/有用证据和 2 条错误证据：

| 输出 | 正确证据 | 错误证据 | 判断 |
|---|---:|---:|---|
| TopK10 | 8 | 2 | 默认基线 |
| 新 Selector | 8 | 1 | 成功：只改善一点，但没有误伤 |
| 激进 Selector | 5 | 0 | 失败：错误全删了，但正确证据损失太多 |
| 新 Selector 无安全候选 | 8 | 2 | 合法回退：与 TopK10 相同 |

第一版的目标是第二行，不是第三行。也就是说，优化重点从“错误证据删除召回率”转为“删除精度”：**允许漏删，不允许为了多删而大量误删。**

### 0.2 不固定每题只删一条，也不预设最后一定删到三条

“最多删一条”只作为候选和对照，不是预先写死的最终规则。实验测试 `0–cap` 策略族：

- 没有安全候选：删 0 条；
- 只有一个安全候选：删 1 条；
- 有多个非常安全的候选：当候选 cap 允许时可以删 2 或 3 条；
- 每删一条都重新检查停止条件；
- TopK10 专用的第一版保险丝为 `min_keep=7`、`max_delete=3`。

这个保险丝只用于本次 TopK10 Selector 实验，不修改仓库其他 `max_selected=2/3/5/6` 的实验配置。

`min_keep=7` 不是在假设“7 一定最优”，只是防止第一轮再次从 10 条骤降到历史 Beam 的约 5 条。真正判断 1、2、3 条删除上限是否限制效果，要看 R007 的 harm–recall 前沿；如果 cap1 落在第 4.6 节冻结的 `ε_harm` 最优范围内，最终就是自适应 0–1；只有 cap2/3 的额外安全收益超过该数值规则，最终才是 0–2 或 0–3。

`max_delete=3` 确实可能放弃一部分“单题删四条以上”才能得到的 harmful reduction，但这是第一版有意选择的安全边界，不代表 3 是理论最优，也不声称已经搜索所有删除数量。项目当前只需要比 TopK 可信地好一点，因此先限制最坏动作比追求最大删除量更合适。

仍保留一个硬保险丝，是因为 CRC 控制的是平均 required/supporting recall 损失与断链风险：如果完全没有逐题上限，模型可能在多数问题不删、却在少数问题一次删很多，平均数仍看似合格。`max_delete≤3` 负责限制单题最坏动作，CRC 负责限制整体平均正确证据损失；deletion precision 仍由独立评测判断，不由 CRC 直接认证。

四个部件的分工是：

1. **scorer：** 给每条证据计算 harm 与 protect 分数；
2. **threshold：** 判断哪些证据有资格进入“可删除候选”；
3. **`max_delete/min_keep`：** 限制一道题最多能采取多大动作；
4. **CRC：** 在独立 calibration 上，从 R007 已冻结的一条策略梯子中选择全局强度；正式测试时不看答案，也不会逐题临时改 cap。

R007 必须输出唯一的 `policy_family`、最终 cap 和 P0–P6 构造规则；R008 只能在这条已冻结梯子里选强度，不能再比较 cap1/2/3 或换回另一种模型。

### 0.3 `ABSTAIN` 的含义彻底冻结

第一版只有三种审计动作：

| 动作 | 实际行为 | 是否计入 harmful reduction |
|---|---|---|
| `KEEP` | 保留并交给 Generator | 否 |
| `ABSTAIN_KEEP` | 模型不确定，仍然保留 | 否 |
| `DROP_HARM` | 从 TopK10 active context 中移除 | 是，但只有它真的属于 harmful 时才算成功 |

`ABSTAIN_KEEP` 不能一边被保留，一边被写成“已经处理 harmful”。未来的隔离查证不属于本轮核心实验。

### 0.4 当前进度：R001–R004 已完成，下一步只做 R005 双头 sanity

R001–R004 已通过各自的前置门；当前只允许进入 R005 的小样本双头 sanity，仍不开始 R006 全量 scorer 训练。已经完成的顺序为：

1. **R001A — inventory（COMPLETE）：** candidate pool、gold、provenance、ParentIndex、run/index manifest 和模型资产的本地/原 HPC 路径、存在性、schema、bytes 与 SHA-256 已记录；
2. **R001B — recover-or-rebuild decision（COMPLETE）：** 六个历史 Hybrid pool 均按原 hash 精确恢复，决策为 `EXACT_RECOVERY / REPACKAGE`；
3. **R001C — pool code integrity（COMPLETE）：** 独立 Hybrid-v2 pool manifest 已实现，六个真实池的 retriever、Top20、query/document/corpus 对齐和逐题 hash 均通过 freeze + verify-only；
4. **R002 — metric/component/CRC protocol（COMPLETE / SAMPLE-SIZE GO）：** component map、派生角色 crossing、CRC representatives、指标符号和 expected-risk 规则已冻结；四项风险代表数均 `n≥99`；
5. **R003 — TopK/count controls（COMPLETE / BASELINE-PROTOCOL PASS）：** TopK10/9/8/7 数量基线和 count-matched random/bottom-rank 生成协议已冻结并复验；固定 TopK9 在 NIAH dev 虽改善约 `2.04 pp` harmful，却损失约 `1.90 pp` recall 和 `4.47 pp` 完整链，因此不能作为保守方案；
6. **R004 — label/resource preflight（COMPLETE / RESOURCE-PREFLIGHT PASS）：** 两数据源全量 active-mask 标签、固定 200-query GPU 前向与短训练资源探针均通过，且没有读取 sealed/heldout 效果、保存 checkpoint 或执行删除；
7. **Gate 0：PASS；Gate 1：PASS；Gate 2：RUNNING；R005：RUNNING（实现与运行前审计通过，formal 尚未运行）。** R004 只证明严格标签和双头 scorer 的资源路线可行，不代表 Selector 或非零删除策略成功；真实 count-matched 结果仍必须等 R005 formal 的真实 Selector trace 决定逐题删除数后生成。

---

## 1. 为什么要从 v1 修订

v1 有三处与实际目标或代码不完全一致：

1. **`DEFER` 计分矛盾。** v1 说可疑证据保留但附标记；如果仍进入 Generator，它就没有降低 harmful exposure。v2 改成 `ABSTAIN_KEEP`，并明确不记功。
2. **utility 定义过宽。** “与问题相关”不等于“应该保护”。v2 改为 `protect score`：估计该候选是否属于正确回答所需的官方 required/supporting evidence。
3. **固定删除预算过于僵硬。** 每题最多删 1 条可能限制多 harmful 问题；完全不设上限又可能重演 Beam 平均只保留约 4.85 条的过度删除。v2 比较 0–cap1/2/3，并用 component-aware expected-risk 规则选择安全强度。

### 1.1 前面的实验不是没用，而是用途改变了

下面的数字来自历史归档的 [`SELECTOR_FINAL_REPORT.md`](../docs/selector/SELECTOR_FINAL_REPORT.md)、[`results-summary.md`](../docs/results-summary.md) 和 v1 计划，只用于定位失败模式；因为完整旧候选池和 checkpoint 当前不在本地，不能直接冒充 v2 的同池新结果。

| 历史阶段 | 已观察到什么 | 现在保留什么 | 不再沿用什么 |
|---|---|---|---|
| 默认 TopK | 保留正确证据最好，也是当前唯一生产 Selector | 继续作为所有实验的主锚点和自动 fallback | 不把“完全不删”误写成 Selector 已有贡献 |
| Beam | harmful exposure 曾从 88.13% 降至 11.64%，但 required recall 从 81.51% 降至 64.21%，平均只保留约 4.85 条 | 说明激进删除能降低 exposure，并给出过删的明确警戒线；没有等量删除对照前，不能证明它有选择性地找到了 harmful | Beam 路径搜索、激进累计删除和“删得越多越好”的目标 |
| MIS | harmful reduction 约 60.42 pp，但 recall loss 约 37.17 pp | 说明只看删错指标会接受一个实际不可用的系统 | 原 MIS 决策规则 |
| 早期 gated | harmful reduction 约 11.2 pp、recall loss 约 4.8 pp | 提供“保守化可能有效”的历史信号，但 recall 仍超线，必须由新 CI 和等量删除对照重新确认 | 只靠一个固定阈值就宣布安全 |
| 数据与评测 | qrels、counterfactual provenance、source hashes、paired evaluation 骨架已形成 | 通过 R001 的 hash/对齐审计后，条件复用数据角色、标签来源、TopK/retrieval、DeBERTa 基座与三种子设置 | 有歧义的三分类 head、把未标注当负例、把保留的 `DEFER` 算作已删除 |

因此，旧实验的结论不是“全部推倒重来”，而是：**保留数据、基线、评测工具、模型基座和失败证据；替换真正导致过删的决策层与计分口径。** 这也解释了为什么 v2 的第一目标不是追求很大的 harmful reduction，而是先找到一个 deletion precision 很高的小安全区域。

### 1.2 论文具体怎样改变本计划

| 文献 | 给我们的直接启发 | v2 真正采用的部分 | 明确不照搬的部分 |
|---|---|---|---|
| [Conformal Risk Control（ICLR 2024）](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf) | 不要凭一个漂亮阈值宣称安全；应在独立 calibration 上按实际 loss 选策略 | 第 4.6 节的 expected-risk 修正、P0 fallback、recall/chain 四风险取最保守策略 | CRC 不负责识别真假，也不是每题保证或“95%安全认证” |
| [Principled Context Engineering for RAG（2025）](https://arxiv.org/abs/2511.17908) | conformal filtering 可以在 RAG 中以 coverage/recall 为先压缩上下文 | “先保护 supporting evidence，再谈删噪声”的 calibration 思路 | 它的单片段 coverage 不能直接替代本项目的多跳完整链与 harmful 指标 |
| [Provence（ICLR 2025）](https://proceedings.iclr.cc/paper_files/paper/2025/file/5e956fef0946dc1e39760f94b78045fe-Paper-Conference.pdf) | 上下文可按每题所需数量动态裁剪；轻量 scorer/pruner 可以有效 | DeBERTa 类 scorer、每题 0–cap 而非固定输出长度、作为 scorer/pruner 结构参考 | relevance pruning 不等于 counterfactual harm 判断；不能直接拿它的删除分数当本项目安全证明 |
| [SetR（ACL 2025）](https://aclanthology.org/2025.acl-long.861/) 与 [Beam Retrieval（NAACL 2024）](https://aclanthology.org/2024.naacl-long.96/) | 多跳证据的价值属于“集合/链”，不是十条彼此独立的 relevance 分数 | complete-chain 条件风险、protect head、集合完整性评测 | 本轮不重新引入无约束 beam path 或大型 CoT set selector；历史结果已显示其过删和实现成本风险 |
| [NEST（ACL Industry 2026）](https://aclanthology.org/2026.acl-industry.35/) | recall amplification 与 precision selection 应分开，先守住证据再做精简 | 冻结 Hybrid Top20/TopK10，只在 TopK10 内有限删除；不把补位与删除混成一个实验 | 本轮不实现 nested retrieval，也不把 NEST 的下游增益当成我们已经得到的结果 |
| [RA-RAG（EMNLP 2025）](https://aclanthology.org/2025.emnlp-main.1738/) | 多来源交叉核验与来源可靠性可能帮助抵抗错误证据 | ParentIndex 只作防泄漏、可选 source 消融和审计 | 来源多数/网站身份不是真值标签；当前 Candidate 也没有足够 lineage，不能把 source 模块写成已完成核心贡献 |

这些论文不是拼成一个庞大系统。v2 只抽取与当前失败最相关的三个原则：**动态但有限的删除、集合级正确证据保护、独立 calibration 风险控制**。其余复杂部件先做消融或推迟，避免再次因为方法太大而无法判断到底哪里有效。

---

## 2. 当前代码与资产的真实情况

### 2.1 当前生产路径

- 当前唯一注册的 Selector 是 `src/evidence_rag/selector/top_k.py::TopKSelector`。
- `src/evidence_rag/composition.py::build_selector` 只接受 `name="top-k"`；Beam、MIS、gated selector 都已退休。
- `configs/selector/beam_v1.toml` 是历史实验配置，当前工厂不能执行它。
- Selector 接口是：

```text
select(query, candidates, max_selected) -> SelectionResult
```

- 这个接口允许返回少于 `max_selected` 的子集，所以“TopK10 内自适应删 0–3 条”可以自然接入，不需要先修改 Selector protocol。
- 本轮候选方法不从 rank 11–20 补位，也不改变保留证据的冻结 retrieval order。

### 2.2 当前 Candidate 和输出里有什么

`EvidenceCandidate` 当前可用字段：

```text
evidence_id, document_id, chunk_id, text, source_uri,
retrieval_score, retrieval_rank, metadata?
```

限制：

- Candidate 中没有 `source_parent_id`、region/view lineage 或 mutation 标签；
- `metadata` 只有 source type、file name、page number、image path，并且当前 dataset normalization 可能丢失 metadata；
- source grouping 必须读取带 hash 的 `ParentIndex` sidecar，不能假设 Candidate 自带完整来源链；
- counterfactual provenance 只能用于训练标签和评测，绝不能作为推理特征；
- `SelectionItem` 只有 evidence id、一个 score 和 rank，不能记录 protect/harm/action。

因此本轮不修改冻结的 `SelectionResult` schema，而额外写 `decision_trace.jsonl`。保留结果仍按原 retrieval rank 输出，`selection_score` 继续使用 retrieval score；双头分数和动作理由全部进入 trace sidecar。

### 2.3 可以直接复用的代码

- TopK 排序和 selector protocol；
- `resolve_selection` 的“只能从原候选池选择”验证；
- selector stage 的 query/candidate/gold 对齐；
- official qrels、NIAH counterfactual provenance、2Wiki supporting facts；
- `ParentIndex` source-parent sidecar；
- Sealed600 fingerprint、BM25 candidate pin、泄漏审计；
- `paired_metric.py` 的 query-paired bootstrap/sign-flip 骨架；
- `harm.py` 的 harmful-in-context 与 pool-hit 计算；
- DeBERTa 基座、三种子、训练依赖和历史训练参数；
- relation training 中的 grouped split、checkpoint fingerprint 思路。

需要补强：

- paired comparison 必须拒绝 query ID 不一致，不能静默取交集；
- Monte Carlo p-value 改为 `(extreme+1)/(iterations+1)`，不再输出 `p=0.0`；
- paired bootstrap/sign-flip 必须新增连通分量成组重采样；共享 parent/family 时不能继续逐 query 当独立样本；
- 新增相对 TopK recall loss、complete-chain loss、删除精度、删除数量分布；
- relation 三分类 head 不能直接当作 protect/harm 双头。
- 当前 `src/evidence_rag/materializer/sealed600.py` / `src/evidence_rag/cli/pin_candidates.py` 的 `CandidateFreeze` 明确只接受 `bm25`，不能拿它给 Hybrid pool 背书；v2 若坚持 Hybrid，必须使用新的、独立的 Selector-v2 pool manifest。

### 2.4 R001A/B 已确认的资产状态

完整 Beam candidate pools 和训练 checkpoint 没有随当前工作树保存；本地主要只有报告、manifest、部分逐题结果和 `niah-train` 的 source-parent sidecar。不过，R001A/B 已在原 HPC 存储上完成只读核验：六个历史 Hybrid RRF Top20 candidate pool、对应 dataset/gold/provenance/source-parent、run/index manifest，以及 Beam seed-13 checkpoint 均存在；六个 pool 的字节级 SHA-256 全部与历史 M0 manifest 一致。详细路径、大小、hash 和核验边界记录在 [`R001_RECOVERY_AUDIT.md`](../results/selector-adaptive-risk-v1/R001/R001_RECOVERY_AUDIT.md)。

因此当前决策不是重建，而是 **`EXACT_RECOVERY / REPACKAGE`**：

1. 六个旧 pool 作为字节级恢复的输入，不重新运行 retrieval，也不把新生成池伪装成旧池；
2. 历史 v1 manifest 只能证明整文件身份，不能代替 v2 所需的逐 query hash、严格对齐和 component 规则；
3. R001C 已在六个恢复文件上生成并重新验证独立 `SelectorCandidatePoolManifestV2`；完整证据见 [`R001C_GATE_EVIDENCE.md`](../results/selector-adaptive-risk-v1/R001/R001C_GATE_EVIDENCE.md)；
4. sealed600 与 2Wiki heldout 在最终阶段前只允许做存在性、内容 hash、schema/结构完整性核验，不得查看 Selector 效果或用于选模型、阈值和策略上限；
5. Beam seed-13 checkpoint 的精确恢复可支持历史诊断，但不是新 dual-head scorer 的训练依赖。

还有一个必须先解决的代码冲突：本计划的目标基线是 Hybrid RRF Top20 → TopK10，但现有 sealed candidate pin 是 **BM25-only**。因此 R001 必须创建并测试独立的 `SelectorCandidatePoolManifestV2`，冻结 retriever 名称、版本、参数 hash、`top_n=20`、query/data signature 和逐题候选 hash。旧 BM25 pin 继续原样保留，不能被放宽后混用。若 Hybrid pool 无法合法恢复或重建，必须在训练前 STOP 并正式修订主张为 BM25；不允许开发集用 Hybrid、正式集偷偷换 BM25。

---

## 3. 两项可证伪主张

| Claim | 主张 | 最低可信证据 | 对应实验块 |
|---|---|---|---|
| C1 主要 | 相比同一冻结池上的 TopK10，自适应保守 Selector 能获得严格为正、统计可信的 NIAH Top20 pool-conditional harmful reduction，同时 NIAH required recall 和 2Wiki supporting recall 基本不下降 | `harm_reduction` 的 paired 95% CI 下界 > 0；两数据集 recall 点损失 ≤1 pp，损失 CI 上界 ≤3 pp；优于逐题等量随机删除 | B1–B3、B5 |
| C2 支持 | 改善来自 protect-aware 的自适应风险控制，而不是“少给 Generator 几条证据” | 在相同逐题删除数量下优于 random/bottom-rank；冻结的 0–cap 策略优于固定输出 TopK9/8/7、harm-only 和无 CRC；cap1/2/3 按安全前沿与简洁性选择 | B3–B4 |

反主张必须排除：

- 只是因为上下文数量减少；
- 只是删除低排名文档；
- 只在 seed 13 成立；
- 只在合成 harmful 成立却被写成现实真伪识别；
- calibration、decision-dev 和 sealed test 被混用。

来源分组是 C2 的辅助消融，不是 C1 首次通过的前置条件。只有 parent sidecar 消融产生独立价值后，才升级为论文核心贡献；当前代码不支持完整 document→page→region→view 主张。

本方法有意不是 frontier LLM/VLM 贡献：核心是可审计的小型 DeBERTa 双头评分和风险控制。大模型查证、生成端隔离和 rank 11–20 补位全部从核心实验中切除，避免用额外复杂度掩盖 Selector 本身是否有效。

### 3.1 Paper storyline

- **Main paper must prove：** C1 主结果、等量删除对照、protect/adaptive 的决定性消融、三种子与 formal。
- **Appendix can support：** scorer calibration 图、rank/label/source 分层、ParentIndex 消融和完整失败案例。
- **Intentionally cut from this round：** LLM verifier、Generator quarantine、set-sufficiency head、rank 11–20 replacement、完整 region/view provenance benchmark。

---

## 4. 精确的方法定义

### 4.1 冻结基线集合

对每个 query：

1. 候选池为同一个冻结 Hybrid RRF Top20，且只能由新的 Selector-v2 pool manifest 验证；现有 BM25-only `CandidateFreeze` 不适用；
2. 按 `(retrieval_rank, evidence_id)` 取前 `max_selected=10` 得到 `S0`；
3. 新 Selector 只能返回 `S0` 的子集；
4. 不允许从 rank 11–20 补位；
5. 保留项继续按原 retrieval rank 排列。

### 4.2 双头分数的正确含义

共享一个 DeBERTa encoder，输出两个独立 sigmoid score：

- `protect_score(q,d)`：候选属于正确回答所需 required/supporting evidence 的可能性；
- `harm_score(q,d)`：候选属于已定义 harmful counterfactual/冲突风险的可能性。

它们不是互斥 softmax。模型可以同时对一条证据给出高 protect 和高 harm；这种冲突表示不确定，动作必须是 `ABSTAIN_KEEP`。

训练标签：

| 数据证据 | protect label | harm label |
|---|---:|---:|
| NIAH official required clean document | 1 | 0，仅在 clean counterpart 映射可靠时 |
| NIAH injected counterfactual document | 0 | 1 |
| NIAH 明确构造的 irrelevant | 0 | 0 |
| 2Wiki official supporting document | 1 | mask |
| 2Wiki unjudged Top20 | mask | mask |
| 其他未明确判断候选 | mask | mask |

这里的 `mask` 表示不参与对应 loss，避免把“未标注”误当成“错误”。若一条现实证据同时包含正确和错误内容，需要未来的片段级多标签标注；不能在本轮凭位置自动赋成双阳性。

冻结模型设置沿用已有可复用部分：

- base：`cross-encoder/nli-deberta-v3-base`；
- revision/hash 在 R001 重新验证；
- max length 512；输入只含 query + 单 candidate，不拼接历史 Beam path；
- seeds `[13,42,73]`；
- seed 13 先行，只有通过单种子 gate 才训练 42/73；
- 学习率、epoch、batch 与旧 `beam_v1.toml` 对齐，除非 200-query 预检证明无法运行；任何变化先写 amendment。

### 4.3 安全删除分数

候选的保守删除分数定义为：

\[
s_{safe}(q,d)=\min(harm\_score(q,d),\ 1-protect\_score(q,d))
\]

它表达“两个条件中较弱的那个”：

- harm 很高但 protect 也高，`s_safe` 仍然低，保留；
- protect 很低但 harm 不高，`s_safe` 仍然低，保留；
- 只有 harm 高且 protect 低，`s_safe` 才高。

这不是在声称分数天然校准成真实概率；最终阈值由独立 CRC calibration 的实际 query loss 决定。

### 4.4 自适应逐步删除算法

每个 query 的运行过程：

```text
S = TopK10
如果冻结策略要求 ParentIndex，但 sidecar 缺失或 hash 不匹配：整题返回 S0，全部 ABSTAIN_KEEP
按 safe_score 从高到低排列候选；完全相同时优先考虑较靠后的 retrieval rank（数值更大），再按 evidence_id

逐个查看候选：
    如果 safe_score 低于冻结阈值：停止
    如果已删除数量达到 max_delete：停止
    如果删除后保留数少于 min_keep：停止
    否则删除该候选，记录 DROP_HARM

所有未删除候选：KEEP 或 ABSTAIN_KEEP
```

第一阶段候选保险丝：

- `max_delete ∈ {1,2,3}` 都实验，不预先宣布某一个最好；
- TopK10 专用 `min_keep=7`；
- 候选 Selector 配置冻结 `baseline_k=10`；pipeline 传入的 `max_selected != 10` 时 fail fast，防止误套到现有 TopK2/3/5/6 配置；
- 候选不足 7 条时全部保留并 fallback，不报错；
- ParentIndex 中单个未解析 document 按当前安全实现退化为“自身就是独立 parent”，只会漏合并，不会错误合并两个来源；
- 每题实际删除数量自然为 0–`max_delete`，不是强制删满。

本文的 `cap1/2/3` 都表示“达到 threshold 才删，因此实际是 0–cap”，不是强制每题删满。强制减少上下文数量的对照是 TopK9/8/7；random/bottom-rank 则严格匹配真实 Selector 的逐题删除数。

### 4.5 为什么不是“每删一条就保证多跳链没断”

当前 Selector protocol 只接收 `query + candidates + max_selected`，不接收官方 gold，也不接收 pipeline 已产生的 `QueryChecklist`。生产时当然也没有 gold。

因此本轮 Selector 不能诚实声称逐题知道“这条就是唯一 hop”。它采用两层保护：

1. 运行时：protect score 高或不确定就不删；
2. 离线 calibration：直接测量每个策略造成的 required recall loss 和 complete-chain failure，只允许风险满足约束的策略上线。

显式 set-sufficiency/hop model 需要扩展接口和新标签，列为后续版本，不用它拖慢 C1 的最小验证。

### 4.6 Component-aware query-risk CRC 如何选策略

CRC 不看单条证据“分数多漂亮”，而直接看每个问题相对 TopK10 损失了什么：

先把 Selector 输出转换成文档 ID 集合：

\[
D(S)=\{d.document\_id:d\in S\}
\]

official qrels 的 gold `G_q` 也是文档 ID 集合，所以 recall 和完整链只能在 `D(S)` 与 `G_q` 之间计算，不能把 `evidence_id`、Candidate 对象和 `document_id` 混在同一个集合公式里。

设策略 `P_j` 的输出为 `S_j`。对有非空 gold 的 query：

\[
L_{rec}(q,P_j)=\max(0, Recall(D(S_0),G_q)-Recall(D(S_j),G_q))
\]

`L_rec` 表示该问题相对 TopK10 少保留了多少 gold evidence。

完整链风险采用**条件口径**，不能让 TopK10 原本就没有完整链的 query 用大量 0 稀释风险。先定义：

\[
E_q=1[G_q\subseteq D(S_0)]
\]

只对 `E_q=1` 的 chain-eligible query 定义：

\[
L_{chain}(q,P_j)=1[G_q\nsubseteq D(S_j)]
\]

所以这里控制的是“TopK10 原本拥有完整 gold chain 时，Selector 把链删断的条件风险”。报告必须同时给出 `n_chain_eligible`，不能把分母换成全部 query。NIAH 和 2Wiki 分开控制，不做平均抵消。

v2 的 calibration 目标为两类平均损失各 `≤1 pp`，decision/formal 的硬不确定性边界为 CI upper `≤3 pp`。P0 与 TopK10 完全相同，按定义始终是合法 fallback；非零策略必须通过实际风险上界。

R006 只冻结 seed13 checkpoint、保存 modelval 分数与候选分位点，不提前宣布策略梯子。R007 再在 train-modelval 上按下述确定性规则比较候选 family/cap，并冻结唯一最终 cap `c*`、`policy_family` 和六个分位点级别：

1. 某个候选 family 如果找不到任何 point 同时满足两数据 recall loss `≤1 pp`、TopK10-chain-eligible conditional chain loss `≤1 pp`、`harm_reduction>0` 且 deletion precision 高于 count-matched random，则淘汰；
2. 对每个剩余 family，记其安全 points 中最大的 NIAH pool-conditional harmful reduction 为 `H_f`，记全体最大值为 `H*`；
3. 冻结简化容忍度 `ε_harm=max(0.5 pp, 100/n_harm_eligible pp)`，其中 `n_harm_eligible` 是 modelval 主 harmful 指标的 eligible query 数；也就是至少允许一个 query 的离散分辨率；
4. 只保留 `H_f≥H*−ε_harm` 的 family，再按预注册复杂度顺序选择最简单者：较小 cap优先，其次不依赖 source sidecar，最后较少 head；完全相同按配置 ID 字典序；
5. 若没有 family 通过第1步，R007 FAIL；不能进入 calibration 后再回来换规则。

这个规则体现本项目目标：大 cap 只有带来超过预设容忍度的额外安全收益才值得保留；“明显更好”不再由看到结果后主观解释。选定 family 后，只用 safe-score threshold 构建一条由保守到激进的嵌套策略梯子，例如：

```text
P0: 不删除
P1: 最高预设 safe_score threshold，max_delete=c*
P2: 次高预设 threshold，max_delete=c*
P3: 更低预设 threshold，max_delete=c*
...
P6: 最低预设 threshold，max_delete=c*
```

seed13 的具体 score threshold 由其 train-modelval 分数和冻结分位点产生，并在查看 CRC-calibration 结果前写入配置。seed42/73 不复用 seed13 的绝对 score 数值，而是复用同一 `policy_family`、cap 与分位点级别，再从各自 train-modelval 分数确定绝对 threshold；它们不得重新选择 family、cap 或分位点。所有策略的实际输出必须逐题嵌套：

\[
S_0\supseteq S_1\supseteq\cdots\supseteq S_6
\]

并因此满足：

\[
L_r(q,P_0)\le L_r(q,P_1)\le\cdots\le L_r(q,P_6)
\]

R002 必须同时测试输出集合嵌套和 loss 单调，不能只看一次运行中 loss 恰好递增。R008 只能从这条梯子选一个 `j`，不得再次比较 cap 或模型。

### 4.6.1 CRC 的统计单位：相关 query 不能重复充当独立样本

query/parent/family 连通分量用于防泄漏，也揭示同一分量内的 query 可能相关。标准 CRC 不能把这些相关变体当成多个独立 calibration 单位。component key 必须由**官方正确证据关系**定义，不能把一次 retrieval 的所有 Top20 candidate parent 都并入：后者会让常见 distractor 把无关 query 错误连成巨型分量，也会让统计单位随 retriever 输出变化。

数据集规则冻结为：

- **NIAH：** 使用 query、gold/needle 对应 source parent 和 synthetic family 多 key 连通分量；counterfactual twin 与其原始 needle 必须处于同一关系范围；
- **2Wiki：** 使用 query 和 official `relevant_document_ids` / supporting facts 对应的规范化 document-title parent 建图；未标注 Top20 candidate、普通 distractor 和 Selector 预测不得成为 component edge；
- 每个既有官方 split 分别生成 component map。2Wiki train/dev/heldout 之间可能共享 supporting parent，这种重叠必须如实计数，不能把它报告为 0，也不能为了制造 0 overlap 而移动官方 heldout query。

R001 的只读结构审计已经量化了这一区别：official supporting-parent 图得到 train `2,324` 个 component（最大 `20` 个 query）、dev `1,732` 个（最大 `8`）、heldout `1,692` 个（最大 `13`）；若错误地把所有 Top20 candidate parent 当边，三个 split **各自都会塌成一个巨型 component**。跨官方 split 的 query overlap 三对均为 `0`，但 distinct official supporting-parent overlap 明确非零：train–dev `449`、train–heldout `422`、dev–heldout `434`。因此后续不得把“query 不重叠”误写成“supporting parent 也不重叠”。

因此本轮对每个“数据集 × 风险”执行以下预冻结规则：

1. recall 的 eligible 集合是有非空 gold 的 query；chain 的 eligible 集合是 `E_q=1` 的 query；
2. 每个连通分量至多贡献一个 eligible query；
3. 若一个分量有多个 eligible query，在查看 scorer 分数或 loss 前，用固定 seed `20260811` 对 `(dataset_id, risk_name, component_id, query_id)` 稳定选出一个代表；
4. `n_r` 是有代表 query 的连通分量数量，不是原始 query 行数；选择清单和 hash 写入 CRC artifact。

这样 CRC 对应的是“随机新连通分量中的预冻结代表 query”的 component-balanced 风险。所有 query 的平均效果仍在 decision/formal 阶段完整报告，并用当前评估 split 内的 official-support component 做成组 bootstrap；两种口径必须并列，不能把代表样本的理论口径偷换成所有 query 独立同分布。

2Wiki 跨既有 split 的 supporting-parent overlap 是**分布相关性与可能记忆效应的敏感性变量**，不是可以通过重分 heldout 消掉的开发自由度。冻结报告必须给出 overlap parent/query 数；正式评估保持全部 heldout query 的主结果，同时预注册 `parent_seen_in_train` 与 `parent_unseen_in_train` 分层，以及排除 seen-parent query 的敏感性结果。该标记只用于评测分层，不进入 scorer 输入，也不改变主样本。heldout 的这些统计只能在 R013 正式冻结后按预注册脚本产生，不能提前查看效果。

R001/R002 必须审计 calibration 与未来数据的 component key、规模和数据来源分布。query/family 的零重叠只能证明相应标识没有直接泄漏，不能证明可交换性；2Wiki supporting-parent overlap 则必须单独披露而不能并入“零重叠”口号。若研究设计无法合理支持“calibration component 与未来 component 可交换/同分布”，则只能把 CRC 当保守选择规则，不能声称有限样本理论保证，并且生产建议退回 P0。

### 4.6.2 expected-risk CRC 的精确选择规则

本轮使用 Angelopoulos 等人的 **expected-risk CRC**，不把它误写成“95% 高概率上界”。对某个 seed、某个风险 `r` 和策略 `P_j`，设 calibration 中有 `n_r` 个有效代表 query。recall 与 chain loss 都在 `[0,1]`，所以论文修正项中的 loss 上界 `B=1`：

\[
\widetilde R_r(P_j)=\frac{\sum_{i=1}^{n_r}L_{r,i}(P_j)+1}{n_r+1}
=\frac{n_r}{n_r+1}\widehat R_r(P_j)+\frac{1}{n_r+1}
\]

风险目标冻结为 `α_rec=α_chain=0.01`。对四项风险分别执行同一规则：

1. NIAH required recall loss；
2. NIAH TopK10-chain-eligible conditional chain loss；
3. 2Wiki supporting recall loss；
4. 2Wiki TopK10-chain-eligible conditional chain loss。

对每项风险定义：

\[
A_r=\left\{j\in\{1,\ldots,6\}:\widetilde R_r(P_j)\le\alpha_r\right\}
\]

\[
\widehat j_r=
\begin{cases}
\max A_r,&A_r\neq\varnothing\\
0,&A_r=\varnothing
\end{cases}
\qquad
\widehat j=\min_r\widehat j_r
\]

也就是：每一项先找满足风险目标的最激进策略；最终取四项中**最保守的那个**。因为四项共享同一条嵌套策略梯子，退到更保守策略不会增加任何一项风险。这里的 expected-risk CRC 规则本身没有需要在“风险×策略×seed”之间分摊的 `δ`，也不额外套用 Bonferroni。每个 seed 独立使用完全相同的冻结规则校准，三个 seed 全部报告，不能从中挑最好者。

P0 与 TopK10 完全相同，其相对损失按结构恒为 0，因此永远是 fallback，但不把它写成“通过了修正经验风险检验”。当 `α=0.01` 时，任何非零策略即使观察到零损失，也必须有 `1/(n_r+1)≤0.01`，即 `n_r≥99`，才可能由此规则选中；若 `n_r=0`，该项直接令 `\widehat j_r=0` 并记录 `NO_CALIBRATION_EVIDENCE`。计划必须报告四项各自的有效 component 数，尤其是 chain-eligible component 数。

在上述可交换性假设下，保证范围是对 calibration 抽样与未来代表 query 共同取期望的边际保证：

\[
\mathbb E_{calibration,\ q_{new}}[L_r(q_{new},P_{\widehat j})]\le\alpha_r
\]

对 chain 风险，这个式子按前述条件总体理解为：

\[
\mathbb E[L_{chain}(q_{new},P_{\widehat j})\mid E_{new}=1]\le\alpha_{chain}
\]

recall 风险则对应“有非空 gold 的代表 query”总体。

它不是“给定这一次冻结 calibration 后，以 95% 概率风险低于 1%”，也不是每一道题的绝对保证。因此 artifact 应写成“由 expected-risk CRC 规则选出”，而不是“这次固定策略已获 99% 安全认证”。第 5 节的 paired 95% bootstrap CI 是在未参与选择的数据上判断观察效果是否稳定的另一个工具，不能替代 CRC，也不能被写成 CRC 的理论保证。方法依据见 [Conformal Risk Control（ICLR 2024）](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf)。

---

## 5. 指标、符号和 95% 置信区间

### 5.1 统一符号

所有报告固定使用“正数代表改善”的定义：

\[
harm\_reduction = Harm(TopK10)-Harm(Selector)
\]

\[
recall\_loss = Recall(TopK10)-Recall(Selector)
\]

因此：

- `harm_reduction > 0` 是好事；
- `recall_loss > 0` 是坏事；
- harmful 的可信改善要求 `95% CI lower > 0`；
- recall 的保护要求 `loss point ≤1 pp`，并且 `95% CI upper ≤3 pp`。

### 5.2 95% CI 的零基础解释

先把 `0` 想成“Selector 和 TopK10 没有差别”。例如 TopK10 的 harmful rate 是 40%，Selector 是 38%，观察到的 `harm_reduction=40%-38%=2 pp`。但有限的一批 query 会有抽样运气，所以不能只看这个 2 pp，还要看误差范围。

同一批 query 同时运行 TopK10 与 Selector，对每个 query 算成对差值。若每个 query 都是独立单位，可以按 query 做 paired bootstrap；本项目存在共享 parent/family 的相关 query，所以正式口径必须按第 6 节连通分量做 **paired cluster bootstrap**：抽取分量时把其中全部 query 一起抽取。得到的 95% 区间表示冻结策略在评估数据上的抽样不确定性：如果反复抽取许多类似的评估 component，并在同一个冻结 Selector 策略上重新计算指标，约 95% 的这类区间会覆盖对应的评估总体平均差值。它不包含重新训练 scorer、重新校准 CRC 或更换 seed 的不确定性，也不是“真实差值有 95% 概率落在本次区间内”的逐题概率承诺。

每个指标先按预注册条件确定 eligible query，再把包含这些 query 的完整 component 作为不可拆分的抽样簇；抽中一个簇后，指标分子和分母仍只使用该指标的 eligible query，chain-ineligible query 不会被重新放进 chain 分母。报告既要写 eligible query 数，也要写有效 component 数。不能为了得到更窄区间，在不同指标中临时拆散 component。

例子：

- harmful reduction 为 `2.0 pp`，CI `[0.5,3.5]`：最保守端仍大于 0，可作为可信改善；
- harmful reduction 为 `2.0 pp`，CI `[-0.8,4.2]`：观察值虽变好，但数据也允许没有改善，只能记作探索性信号；
- harmful reduction 的 CI 为 `[-3.0,-0.5]`：它也“不跨 0”，但整段在 0 左侧，表示 Selector 可信地更差；本项目要求的是 `lower>0`，`lower=0` 也不通过；
- recall loss 为 `0.6 pp`，CI `[-0.2,1.7]`：点损失小，最坏可信端也低于 3 pp，满足保护门槛；
- recall loss 为 `0.6 pp`，CI `[-1.0,4.1]`：平均看似安全，但不确定性太大，不能正式通过。

置信区间不是“模型每题有 95% 正确率”，也不是单题保证。

两个门槛看区间的方向不同，是因为问题不同：

- harmful 问的是“能否证明真的比 TopK 好”，所以整段区间都必须在 0 右边，即 `lower>0`；
- recall 问的是“能否排除不可接受的大损失”，所以允许区间跨 0，但最坏上端必须不超过 3 pp，即 `upper≤3 pp`。

paired bootstrap CI 主要反映“换一批相似 query”带来的抽样不确定性，不替代训练随机性分析。三种子阶段必须另外报告每个 seed、均值、标准差和最差 seed；两种不确定性不能混成一个数字。

### 5.3 主指标

1. pool-conditional harmful exposure：只在 harmful 已进入冻结 Top20 pool 的 query 上；这是与历史 Beam 口径可比的主 harmful 指标；
2. baseline-exposed harmful deletion：只在 harmful 已进入 TopK10 `S0` 的 query 上，观察 Selector 真正删除了多少；
3. unconditional harmful exposure：全部 injected query；
4. pool-conditional harmful reduction 及 paired cluster-bootstrap 95% CI；
5. required/supporting recall loss 及 paired cluster-bootstrap 95% CI；
6. TopK10-chain-eligible conditional chain failure/survival，以及确切 `n_chain_eligible`；
7. deletion precision：所有 `DROP_HARM` 中真实 harmful 的比例；
8. required deletion rate：被删除候选中 official required/supporting 的比例；
9. 每题删除 0/1/2/3 条的分布和平均保留数；
10. 与逐题 count-matched random/bottom-rank 的差值；
11. risk–coverage / harm–recall frontier。

三个 harmful 分母必须同时写出确切 `n`。不能把“Top20 pool hit”“TopK10 已暴露”和“全部 injected query”混成一个数字。

2Wiki 没有 harmful 标签，只用于证明多跳正确证据没有因为 NIAH 的删除规则而受损。不得给 2Wiki 编造 harmful 指标。

---

## 6. 数据角色与防泄漏

| 数据 | 已知规模 | v2 角色 |
|---|---:|---|
| NIAH train assignments | 1,023 | 约90% train-fit；约10% train-modelval，按 component 整组分配 |
| 2Wiki train | 3,000 | 约90% train-fit；约10% train-modelval，按 component 整组分配 |
| NIAH dev assignments | 1,479 | 约50% CRC-calibration；约50% decision-dev，按 component 整组分配 |
| 2Wiki dev | 2,000 | 约50% CRC-calibration；约50% decision-dev，按 component 整组分配 |
| NIAH sealed600 | 600 | M5 一次性正式测试 |
| 2Wiki heldout | 2,000 | M5 一次性正式测试 |

规则：

- train-modelval 选择 checkpoint、供 R007 冻结唯一 family/cap/分位点策略梯子；
- CRC-calibration 只选择策略，不训练 scorer；
- decision-dev 只做 go/no-go，不回传调阈值；
- sealed/heldout 只在所有配置和 hashes 冻结后运行一次；
- grouped split 不能把 `query_id + source_parent_id + synthetic_family` 简单拼成一个字符串；每个样本同时携带多个关系 key，只要两个样本共享**任意一个被允许的 key**，就用连通分量/union-find 放进同一组；
- NIAH 允许的 key 是 query、gold/needle source parent 和 synthetic family；2Wiki 允许的 key 只有 query 与 official supporting/gold document-title parent，**不包含所有 Top20 candidate parent**；
- 在同一个既有 source split 内创建派生角色时必须按 component 整组分配：NIAH-train 的 train-fit/train-modelval、2Wiki-train 的 train-fit/train-modelval、NIAH-dev 的 CRC-calibration/decision-dev、2Wiki-dev 的 CRC-calibration/decision-dev，各自输出 query/component crossing=0；
- 对 NIAH 继续执行既有 query/source-parent/family fingerprint 防泄漏审计；对 2Wiki，跨官方 train/dev/heldout 的 query ID overlap 必须为 0，但 supporting-parent overlap 必须报告实测数量，不得冒充为 0；
- R001 已记录 2Wiki query overlap 三对均为0；supporting-parent overlap 为 train–dev `449`、train–heldout `422`、dev–heldout `434`。R001C/R002 必须在正式 artifact 中复算并冻结这些计数；
- 2Wiki heldout 保持官方成员完全不变，不与 train/dev 重新联合分组或重新分配。正式主结果使用全部 heldout；同 split component 用于 cluster bootstrap，跨 split parent-seen/unseen 用于预注册敏感性分层；
- 每个既有 split 的冻结 component map 在适用环节用于派生数据拆分、CRC representative 选择和 paired cluster bootstrap，不能为不同环节另造互相矛盾的分组；
- provenance/source-parent/group key 只参与拆分、标签、评测和策略 sidecar，不进入文本 scorer 的输入。

---

## 7. 三个基线家族

| 家族 | 核心比较 | 排除的错误解释 |
|---|---|---|
| 排序/数量 | TopK10、TopK9、TopK8、TopK7、逐题 count-matched random、逐题 count-matched bottom-rank | 提升是否只是少给文档或删低排名 |
| 分数 | harm-only、protect-only、双头 safe score；可恢复时历史 Beam score 仅作诊断 | protect head 是否真的减少误删 |
| 决策 | 固定输出 TopK9/8/7、0–cap1/2/3 threshold family、无 CRC、component-aware CRC | 改善是否来自风险控制而非阈值运气 |

历史 Beam/MIS 代码不在当前工作树，不能作为可运行核心基线。它们的归档报告用于说明旧方法的失败边界，不伪装成当前可复现 run。

---

## 8. 五个核心实验块

### B1. 资产、指标与基线完整性

- **Claim tested：** 所有后续比较发生在同一 pool、同一 query 和可复算口径上。
- **数据：** NIAH train/dev，2Wiki train/dev；sealed/heldout 只做 hash/存在性审计，不读取结果。
- **系统：** TopK10/9/8/7、count-matched random、bottom-rank。
- **指标：** pool/query hashes、TopK 指标、conditional/unconditional harm、paired CI、删除数量。
- **随机对照协议：** R003 先冻结生成器、master seed `20260811` 和 100 个重复的派生 seed；等某个实际 Selector run 产生逐题删除数量后，再对每个 query 删除完全相同的数量，同时报告均值和重复间变异。
- **成功：** query 对齐严格一致；旧指标在同池可复算；plus-one 修复；所有逐题输出可追溯。
- **失败解释：** 不能开始模型实验；优先恢复/重建 artifacts。
- **目标：** 主文方法可信度；MUST-RUN。

### B2. 双头 scorer 是否存在“安全删除角落”

- **Claim tested：** 至少有一小部分 harmful 能被高精度识别，同时 required/supporting 被 protect。
- **数据：** train-fit 训练；只在 train-modelval 诊断。
- **系统：** retrieval score、harm-only、protect-only、dual-head。
- **指标：** deletion precision、PR-AUC、Brier/ECE 诊断、safe-score 排名、rank/数据源/截断分层。
- **宽松成功：** 至少一个非零策略在 modelval 上 `harm_reduction point > 0`、两数据 recall loss 均 ≤3 pp，且 deletion precision 高于 count-matched random。
- **失败解释：** 当前表示没有可利用的保守区域；停止自动删除，保留 TopK10。
- **目标：** 主文前置；MUST-RUN。

### B3. 自适应保守 Selector 主结果

- **Claim tested：** C1。
- **数据：** CRC-calibration 选策略；decision-dev 只评估 seed 13。
- **系统：** TopK10、最佳数量基线、harm-only fixed threshold、dual-head fixed threshold、adaptive + CRC。
- **主成功门槛：**
  - NIAH pool-conditional `harm_reduction CI lower > 0`；
  - unconditional harm 同方向报告，不要求替代主门槛；
  - NIAH required recall loss point ≤1 pp、CI upper ≤3 pp；
  - 2Wiki supporting recall loss point ≤1 pp、CI upper ≤3 pp；
  - TopK10-chain-eligible conditional chain loss point ≤1 pp、CI upper ≤3 pp；
  - 优于逐题 count-matched random 和 bottom-rank；
  - 非零 action coverage：至少有部分 query 真正 `DROP_HARM`。
- **失败解释：** 没有证明优于 TopK；不训练额外种子、不进 sealed。
- **目标：** 主表 Table 1 与 frontier Figure 1；MUST-RUN。

### B4. 自适应机制与来源消融

- **Claim tested：** C2。
- **数据：** train-fit 上的 grouped OOF / train-modelval，seed 13；在 CRC-calibration 和 decision-dev 之前完成方法选择。
- **系统：** 0–cap1/2/3、固定输出 TopK9/8/7、harm-only/去掉 protect、去掉/加入 ParentIndex、纯文本 dedup；CRC 与无 CRC 的风险差异由 R008 在 calibration 上按预注册规则补充，不用于回头改 scorer。
- **指标：** 相同删除数量下的 harmful reduction、recall loss、deletion precision、0/1/2/3 action 分布。
- **成功：** 选中的 0–cap + protect 位于更好的 harm–recall 前沿；CRC 的作用在 R008 的风险校准与 P0 fallback 单独验证；source parent 若无额外价值则降级为 appendix，不影响 C1。
- **失败解释：** 如果 0–cap1 落在冻结的 `ε_harm` 最优范围内，在看到 calibration/decision-dev 前就冻结它；不为了“方法复杂”保留无贡献组件。
- **目标：** 主文机制表或 appendix；MUST-RUN for simplicity，source 部分 CONDITIONAL。

### B5. 三种子和一次性正式确认

- **Claim tested：** C1 不依赖一次随机训练，且能推广到 sealed/heldout。
- **设置：** `[13,42,73]`；只在 B3 通过后补 seed 42/73。
- **多种子门槛：** 三个 seed 的 harmful point reduction 都必须为正；对每个 query 先平均三个 seed 的差值，再按连通分量做 paired cluster bootstrap，其 CI lower 必须 >0；任一 seed recall loss point >3 pp 直接失败。
- **正式设置：** 冻结三个 checkpoint、各自 CRC artifact、三种子汇总规则、source sidecar、配置和 hashes 后，把三个 seed 作为一个预注册批次一次性运行 sealed600 与 2Wiki heldout；不根据正式结果挑 seed。
- **正式成功：** sealed600 重复 B3 harmful/recall 门槛；2Wiki heldout 重复 recall/chain 门槛。
- **失败解释：** TopK10 保持默认；不得调阈值后再次称为同一次正式测试。
- **目标：** 主表 final columns；MUST-RUN。

---

## 9. Run 顺序与决策门

| Milestone | Runs | 目标 | Go/Stop | 预算 | 主要风险 |
|---|---|---|---|---|---|
| M0 完整性 | R001–R003 | 恢复/冻结 artifacts，修指标，建立数量基线 | 对齐失败或任一必需风险 component n<99 则 STOP | 0 GPU | 历史池只在旧 HPC |
| M1 scorer sanity | R004–R005 | 200-query 标签审计、小样本过拟合和吞吐 | 找不到安全删除角落则 STOP | 小样本 GPU，≤0.25 训练单位 | sparse/masked 标签 |
| M2 方法冻结 | R006–R008 | 训练双头、在 modelval 做消融并冻结最简方法、CRC calibration | 只有非零策略被校准才 GO | ≤1 训练单位 | scorer 可分但策略不安全 |
| M3 单种子决策 | R009 | 在未使用的 decision-dev 上一次性检验 B3 | B3 全门槛通过才 GO | 复用 seed 13 | 小提升 CI 跨 0 |
| M4 稳定性 | R010–R012 | seeds 42/73 与三种子 decision-dev | 任一 hard recall gate 失败则 STOP | 新增 2 训练单位 | seed variance |
| M5 正式 | R013–R015 | 冻结并一次性 sealed/heldout | 全门槛通过才建议注册 | 推理 | test 诱导调参 |

核心训练上限仍为 3 个完整 seed。绝对 GPU 小时在 R004 用实际吞吐测量后写入 tracker；当前历史报告没有可信 wall-time，不虚构小时数。

所有新产物统一写入 `results/selector-adaptive-risk-v1/<run_id>/`。每个目录至少包含 config、独立且升版的 `selector_experiment_manifest.json`、日志和 checksums；逐 candidate scores、decision trace、selected sets、聚合指标与 paired CI 按当前阶段是否适用生成。阶段上不适用的产物必须在 manifest 中显式写成 `NOT_APPLICABLE`，依赖未来真实 trace 的产物必须写成 `DEFERRED`，不能伪造占位结果；一旦进入 scorer/Selector 运行阶段，缺少当阶段必需产物的 Run 不得标记 PASS。

现有严格 `RunManifest` 没有 scorer checkpoint、CRC artifact、source sidecar、label/data/pool 等全部字段，不能假装已经覆盖。v2 新 manifest 必须显式保存这些内容 hash 和 schema version，并由独立实验 runner 同时写出 `(SelectionResult, SelectorDecisionTrace)` 或等价的两个 sidecar；生产 `Selector.select(...) -> SelectionResult` protocol 保持不变。

---

## 10. 具体 Run 定义

### R001 — artifact-availability-and-freeze

- **R001A inventory — COMPLETE：** 已盘点本地/远端 candidate pools、gold、provenance、source-parent、run/index manifest 和 checkpoint，并记录路径、存在性、schema、bytes 与 SHA-256；
- **R001B recover/rebuild — COMPLETE：** 六个 Hybrid RRF Top20 pool 均与历史 SHA-256 精确匹配，决策冻结为 `EXACT_RECOVERY / REPACKAGE`，不重跑 retrieval；
- **R001C pool code integrity — COMPLETE：** 六个恢复池均已生成并通过独立 `SelectorCandidatePoolManifestV2` 的二次验证；只接受冻结的 Hybrid 名称、版本、参数 hash、Top20 和 query/data signatures；BM25-only pin 未修改或复用；
- **R002 联合收尾 — COMPLETE：** 数据角色、连通分量和 CRC 代表 query 选择规则已经冻结；NIAH/2Wiki train/dev 的派生角色 query/component crossing 均为 `0`；
- **Gate 0 — PASS：** 已提交完整证据；没有读取 sealed/heldout 的 Selector 效果，也没有训练 scorer。

### R002 — metric-and-crc-unit-protocol — COMPLETE / SAMPLE-SIZE GO

- 修复 strict paired query alignment、paired cluster bootstrap/sign-flip 与 plus-one p-value；
- 新增 `harm_reduction`、relative recall loss、conditional chain loss、deletion precision；
- 用手工 toy cases 和 simulated bounded losses 验证文档 ID 口径、chain-eligible 分母、每 component 一个预冻结代表、CRC 修正式 `（ΣL+1）/(n+1)`、集合嵌套、四风险取最保守策略和 P0 fallback；
- 计算四项风险各自有效 component `n`；任一 `n<99` 时当前 `α=1%` 路线在训练前 CUT/BLOCKED，不进入 R004；
- 冻结符号，并明确 CRC expected-risk 与 bootstrap CI 是两个不同关卡。
- **实际结果：** NIAH recall `n=523`、NIAH conditional chain `n=402`、2Wiki recall `n=866`、2Wiki conditional chain `n=468`，全部满足 `n≥99`；四套 artifact 的独立 verify-only 为 `4/4 PASS`，本地/服务器 24 文件 hash 为 `24/24 MATCH`，预注册 projection 为 `10/10 MATCH`。
- **证据：** `results/selector-adaptive-risk-v1/R002/R002_PROTOCOL_REPORT.md` 与 `R002_VALIDATION_REPORT.json`。
- **解释边界：** 当前没有 scorer、真实策略 loss 或非零 CRC 通过结果；`SAMPLE-SIZE GO` 只允许继续 R003。

### R003 — topk-and-count-controls — COMPLETE / BASELINE-PROTOCOL PASS

- 已在 NIAH/2Wiki train/dev 的冻结 Hybrid Top20 池上运行并复验 TopK10/9/8/7；四个任务加一个 protocol 任务正式生成 `5/5 PASS`、独立 verify-only `5/5 PASS`，本地/服务器原始文件 `14/14 SHA-256 MATCH`；
- 已冻结 TopK10 主锚点、三种 NIAH harmful 分母、document-ID recall、TopK10-chain-eligible 条件分母与 component-cluster paired CI；
- 已冻结逐题 count-matched random/bottom-rank 生成协议、master seed `20260811` 和 100 个重复的派生 seed；因为没有真实 Selector trace，数值结果按协议保持 `DEFERRED`；
- **关键结果：** NIAH dev 的固定 TopK9 相对 TopK10，pool-conditional harmful reduction 约 `+2.04 pp`，但 required recall loss 约 `1.90 pp`、conditional chain loss 约 `4.47 pp`，不满足保守目标；
- **证据：** `results/selector-adaptive-risk-v1/R003/R003_PROTOCOL_REPORT.md`、`R003_VALIDATION_REPORT.json`、`selector_experiment_manifest.json` 与 `CHECKSUMS.sha256`；
- **解释边界：** R003 PASS 仅证明基线/协议完整且可复算，不表示 Selector、阈值或非零删除策略通过。

### R004 — label-audit-and-200q-preflight — COMPLETE / RESOURCE-PREFLIGHT PASS

- 在两个 source-train 的全部合格 query 上生成并审计 protect/harm/mask 标签；确保 provenance 只用于造标签和核验，模型可见投影严格只有 `question + candidate_text`；
- NIAH 只有 own counterfactual 使用 `(protect=0,harm=1)`；official required 使用 `protect=1`，且只有通过 MutationRecord 的 query、needle/twin、文本双 hash、替换 span 和可逆恢复联合核验的 clean needle 才使用 `harm=0`；其余候选全部 mask；
- 2Wiki 只有 official supporting 使用 `(protect=1,harm=mask)`，所有 unjudged Top20 使用 `(mask,mask)`；不再复用旧 Beam 的“非 gold 即 negative”规则；
- 200-query 前向样本事前固定为 NIAH/2Wiki 各 100 个 `train-modelval` query，按 `sha256("selector-r004-preflight-v1\n{dataset_kind}\n{query_id}\n20260811")` 从小到大选择，不按标签、长度或模型结果挑样本；
- 每题前向完整冻结 Top20，共 4,000 个 text pair；这只是覆盖 scorer 训练的最大候选负载，后续 Selector 动作域仍是 TopK10；另对全部 403 个 `train-modelval` query、8,060 个 pair 做 tokenizer-only 截断审计；
- 用未训练的独立双头只测 forward；最多 12 个临时 micro-batch 可用于 masked-BCE backward/optimizer 显存和速度探针，进程结束即丢弃且不写 checkpoint，因此不构成 R005/R006 训练；
- 以实测训练 micro-batch 时间、正式 batch/accumulation/epoch 和两数据源 1:1 调度公式记录 seed-13 GPU 小时点估计与保守估计；估计假设必须写入报告，不能把 forward 吞吐直接冒充完整训练速度；
- **Go：** 标签/配对/输入 pin 审计零异常，两个 head 均为独立 sigmoid，空 mask loss 有限，固定模型离线加载成功，4,000-pair forward 与临时训练探针无 OOM/NaN，并生成可复算 GPU 时间估计；否则 `FAIL/BLOCKED`，不得进入 R005。
- **正式结果：** NIAH `20,460` 条与 2Wiki `60,000` 条标签全部冻结并通过重新计算式 verify-only；NIAH `965` 条 `(protect=0,harm=1)`、`973` 条 `(protect=1,harm=0)`，候选文本语义违规为 `0`；2Wiki `5,922` 条 official supporting 仅监督 protect，其余 `54,078` 条保持双 mask；
- **资源结果：** 固定 `100+100` query、`4,000` pair 前向为 `145.431 pair/s`；全部 `403` query、`8,060` pair 的最大长度为 `298<512`，截断 `0`；12 个临时训练 micro-batch 无 OOM/NaN，峰值 allocated 显存 `4,331,716,096` bytes；seed-13 全量训练点估计 `0.2142 GPU-hour`，p95 保守估计 `0.2818 GPU-hour`；
- **封存：** 正式 clean Git 为 `3170d154351cc620e9cee5cefbdf49cae7e831a2`；staging 内 freeze/verify、原子改名后 preflight verify 与顶层 manifest verify 全部 PASS；独立 post-run 审计重算标签、样本、token、双头输出、训练探针和 GPU 时间后为 PASS、无 P0/P1；顶层 manifest SHA-256 为 `4d7c393f5943583039f85020a8d3de41995e21683f5979e264db70d25f1b6539`；
- **证据：** [`resource_preflight_report.json`](../results/selector-adaptive-risk-v1/R004/preflight/resource_preflight_report.json)、[`selector_experiment_manifest.json`](../results/selector-adaptive-risk-v1/R004/selector_experiment_manifest.json) 与 [`CHECKSUMS.sha256`](../results/selector-adaptive-risk-v1/R004/CHECKSUMS.sha256)；
- **解释边界：** 未训练 scorer checkpoint、未构造删除策略、未读取 sealed/heldout 效果、未改变生产 TopK10；R004 PASS 只允许进入 R005 sanity。

### R005 — dual-head-sanity

- 从两数据源的 `train-fit` 各取 16 个 query；抽样顺序固定为 `sha256("selector-r005-sanity-v1\n{dataset_kind}\n{query_id}\n20260811")` 升序，不查看文本长度、标签分布或模型结果后换样本；抽中样本若缺少必要 active class 覆盖则 R005 FAIL，不重抽“更容易”的题；
- 上述固定样本覆盖不足时也必须留下完整、可复验的 FAIL 证据：保存 seed-13 初始化态的 epoch-0 checkpoint，明确写出缺失类别/有效配对数，并令 candidate scores、decision trace、selected sets 与 count-matched 文件为空；若 epoch 1–30 的训练循环内部遇到 CUDA OOM 或 NaN/Inf，则丢弃失败 epoch 的任何部分更新、回滚并保存最后一个完整 epoch 的 checkpoint，记录实际完成 epoch 和失败类别，且同样禁止继续阈值/modelval；普通代码错误、输入 hash 损坏、非法标签，以及初始 snapshot/基线评分、checkpoint 保存/重载、最终样本评分/梯度探针或 modelval 执行阶段的技术异常仍应直接报错并令该次运行保持未完成，不能伪装成实验 FAIL；
- 只用抽中 query 的 active-mask 行、按 NIAH/2Wiki `1:1` micro-batch 调度训练 30 epochs；class weight 仅由这批 `train-fit` active labels 按“source × head × observed class 的 inverse-sqrt-frequency，再归一到 active 平均权重为 1”计算，零频类别不造样本、不除零；加权 BCE 在每个 head 内固定除以该 micro-batch 的 active 样本数，而不是再除以 active weight 总和，否则单一类别 micro-batch 中权重会被自身抵消；
- 最终 epoch checkpoint 是唯一 checkpoint；NIAH 必须实际覆盖 protect 0/1 与 harm 0/1 四类，且四类各自准确率 `≥0.95`；2Wiki 必须有 protect positive 且准确率 `≥0.95`，其 protect negative 与 harm 0/1 因没有合法标签明确记为 N/A，不能用 mask 行补分母；至少 12 个 NIAH verified clean/counterfactual 同题配对，并要求 `protect(clean)>protect(cf)`、`harm(cf)>harm(clean)`、`safe(cf)>safe(clean)` 三个严格方向同时成立的比例 `≥0.95`，tie 计失败；两项 active head loss 都必须有限并较初始化下降，两个 head 的参数都必须改变，2Wiki 全 mask 的 harm head 梯度必须为零；
- checkpoint 只用于证明训练链能学习，标记为 `R005-sanity-only`，R006 必须重新从冻结 base 初始化，不能沿用该 checkpoint；
- manifest 中的通用 model identity 固定表示与 R004 相同的 base snapshot；训练后模型状态另用严格 checkpoint fingerprint 和 checkpoint 文件 hash 绑定，不能把两种 hash 混成同一含义；正式目录只从同一文件系统的 staging 在完整 freeze + verify 后原子改名发布，不依赖可被复制改变的文件 mtime；同一 pinned server 环境负责加载 checkpoint 后的分数/阈值/策略语义重算，跨机器只声称内容 hash 完整性核验，不把任意硬件上的浮点逐字节一致夸大为可移植保证；
- 阈值不得从 held-out 结果反推：最终 checkpoint 先对两数据源**全部** `train-fit` query 的 TopK10 评分，合并且不按标签筛选这些 safe scores，再用 nearest-rank 按保守到激进顺序映射固定分位点 `[0.99,0.975,0.95,0.90]`；重复分位点阈值照样保留并报告，不能临时补新阈值；R005 仅运行诊断性的 `0–cap1`，这些阈值、cap 和 checkpoint 均不得带入 R006/R007 的正式方法选择；
- 冻结 checkpoint/阈值后，一次性评分全部 `train-modelval`：NIAH 103 题、2Wiki 300 题，每题保存 Top20 双头分数，但删除动作严格只在 TopK10；输出阈值形成所需的全量 train-fit TopK10 scores，以及所有预注册诊断 point 的 modelval candidate scores、decision trace、selected sets、metrics 和基于真实逐题删除数的 100-repeat count-matched random/bottom-rank 对照；若多个 point 通过，只把配置中从 `.99` 开始的第一个通过者记为 diagnostic witness；
- **过拟合 Go：** 对完整执行并成功得到的 loss/score，数值必须全部有限，epoch 1–30 训练循环内无 OOM/NaN，且两头为独立 sigmoid、所有训练准确率/配对方向达到上述门槛；这些模型/训练判据不满足则 FAIL，不进入 R006；若 checkpoint、最终评分/梯度探针等执行本身中断，则本次运行不产生实验状态，修复执行问题后仍须按同一冻结协议重跑；
- **safe-corner Go：** 至少一个事前分位点有真实非零 `DROP_HARM`，NIAH Top20 pool-conditional harmful reduction point `>0`；NIAH recall、NIAH conditional chain、2Wiki recall、2Wiki conditional chain 四项损失必须分别 `≤3 pp`，禁止跨数据或跨指标平均抵消；NIAH deletion precision 必须严格高于 100 次逐题等量随机删除 precision 的均值，并同时报告这 100 次的分布，而不是要求高于任意一次或全部 100 次；否则 CUT，不进入 R006；
- 检查显式 P0 的 0 删除严格等于 TopK10；高 harm/高 protect 冲突、达到 cap/min-keep、缺分数或依赖不完整时使用 `ABSTAIN_KEEP`，它仍是保留且不得记作 harmful 改善；候选少于 7 条整题 fallback，`max_selected!=10` fail fast，不从 rank 11–20 补位；
- **解释边界：** R005 只问“链路能否学会、是否存在值得继续的安全角落”，不要求 harmful 95% CI 下界大于 0，也不冻结正式方法；R009 才进行未参与选择的单种子可信效果检验。

### R006 — train-seed13

- 全量 train-fit 训练双头；
- train-modelval 按预注册 checkpoint rule 冻结 checkpoint，只保存分数分布和候选分位点，不冻结 P0–P6；
- 保存逐 candidate 双头分数和模型 hash。

### R007 — method-selection-ablations-seed13

- 只在 grouped OOF/train-modelval 比较 0–cap1/2/3、固定输出 TopK9/8/7、harm-only 和无 protect；
- ParentIndex/文本 dedup 只在 sidecar 完整时比较；
- 按第 4.6 节的 `ε_harm` 与复杂度顺序冻结唯一 `policy_family`、最终 cap、分位点级别和只改变 threshold 的 P0–P6；
- 从各候选 family 的 modelval trace 生成预注册 count-matched 对照；
- 不查看 CRC-calibration 或 decision-dev。

### R008 — crc-calibrate-seed13

- 只在 CRC-calibration 上选择 P0–P6；
- 写不可变 CRC artifact；
- 如果所有非零 `P_j` 均未通过、最终只能结构性回退 P0，记录 CUT，不运行 R009。

### R009 — decision-dev-seed13

- 一次性评估已冻结 seed13 策略；
- 从本次真实 decision trace 的逐题删除数生成 count-matched random/bottom-rank 对照；
- 判定 B3；
- 失败不通过改阈值补救。

### R010/R011 — train-and-calibrate-seed42/73

- 只在 R009 PASS 后训练；
- 对每个 seed 复用 R007 的唯一 family、cap 和分位点级别；绝对 threshold 只由该 seed 自己的 train-modelval 分数产生，再在同一 CRC-calibration 角色上选择 `j`；
- 不改变数据 split、模型选择规则、风险目标或策略梯子构造方法。

### R012 — three-seed-decision-dev

- 报告全部种子，不挑最优；
- 为三个 seed 的实际 trace 分别生成预注册 count-matched 对照；
- 冻结三个 checkpoint、各自 CRC policy 和三种子汇总规则；不根据 formal test 挑 seed。

### R013 — formal-freeze

- 写 model/config/CRC/source/data hashes；
- 确认 sealed600/heldout 未参与选择；
- 只有 manifest 完整才允许 R014/R015。

### R014/R015 — formal-sealed600 / formal-2wiki-heldout

- 作为一个预注册正式批次各运行一次；
- 从正式 trace 生成对应 count-matched 对照，不复用开发集的删除数量；
- 不在两个结果之间调整任何配置；
- 写最终 PASS/FAIL 和 TopK 是否保持默认。

---

## 11. 与当前代码匹配的最小实现清单

M0/M1 计划新增：

- `src/evidence_rag/selector/dual_head.py`
  - scorer protocol、protect/harm output、checkpoint fingerprint；
- `src/evidence_rag/selector/risk_controlled.py`
  - TopK10 子集、0–3 逐步删除、min_keep、fallback；
- `src/evidence_rag/selector/models.py`
  - `KEEP / DROP_HARM / ABSTAIN_KEEP` 和 decision trace sidecar；
- `src/evidence_rag/materializer/selector_labels.py`
  - qrels + MutationRecord 标签、mask、多 key 连通分量 split 与 representative list；
- `src/evidence_rag/materializer/selector_pool.py`
  - 独立 Hybrid `SelectorCandidatePoolManifestV2`、逐题候选 hash 和严格验证；
- `src/evidence_rag/evaluation/selector_risk.py`
  - strict paired alignment、conditional chain、relative recall/harm/action metrics、cluster bootstrap；
- `src/evidence_rag/evaluation/crc.py`
  - component-representative expected-risk rule、嵌套 policy selection、immutable calibration artifact；
- `src/evidence_rag/evaluation/selector_runner.py`
  - 保持生产 protocol 不变，同时写 `SelectionResult` 与 `SelectorDecisionTrace`；
- `src/evidence_rag/evaluation/selector_artifacts.py`
  - v2 experiment manifest schema 与 scorer/CRC/source/label/data/pool hashes；
- `src/evidence_rag/cli/calibrate_selector_crc.py`；
- `src/evidence_rag/cli/run_selector_risk_experiment.py`；
- `configs/selector/adaptive_risk_v1.toml`（R004 已封存配置，不得原位修改）；
- `configs/selector/adaptive_risk_r005_sanity.toml`（R005 独立冻结配置）。

测试至少新增：

- `tests/selector/test_dual_head.py`；
- `tests/selector/test_risk_controlled.py`；
- `tests/materializer/test_selector_labels.py`；
- `tests/materializer/test_selector_pool.py`；
- `tests/evaluation/test_selector_risk.py`；
- `tests/evaluation/test_crc.py`；
- `tests/evaluation/test_selector_runner.py`；
- `tests/evaluation/test_selector_artifacts.py`。

并扩展：

- `tests/pipeline/test_selector_registration.py`；
- `tests/contracts/test_validation.py`；
- `tests/evaluation/test_paired_metric.py`；
- `tests/evaluation/test_harm.py`。

### 注册顺序

1. R001–R012 阶段由独立实验 CLI 实例化候选 Selector；
2. 不修改当前 `build_selector(name="top-k")` 的默认行为；
3. 只有 R014/R015 PASS 后，才在 `composition.py` 增加显式 `adaptive-risk` 注册；
4. scorer、CRC artifact 和 ParentIndex 必须以依赖注入方式构造，并把内容 hash 写入独立的 v2 experiment manifest；
5. selector 参数中的路径当前不会自动相对 TOML 解析，新增 loader 必须显式解析和测试。

---

## 12. 预注册停止条件

以下任一情况出现立即停止对应路线：

1. candidate pool、query、gold 或 provenance 无法严格对齐；
2. 目标写 Hybrid，却使用现有 BM25-only candidate pin，或不同数据角色混用 retriever；
3. 新 v2 pool 与旧 hash 不同却仍试图直接复用旧 Beam 数字作同池比较；
4. 同一既有 source split 派生出的角色发生 query/component crossing；或把 2Wiki 所有 Top20 candidate parent 错当 component key；或隐藏跨官方 split 的 supporting-parent overlap 并冒充为 0；
5. component map/representative 规则无法冻结，或研究设计不能合理支持 calibration 与未来 component 的可交换性却仍声称 CRC 理论保证；
6. 任一必需风险的有效 component `n<99`，使 `α=1%` 下非零策略不可能被选中；此时训练前 STOP，除非先基于科学容忍度提交 amendment，不能看过模型结果再放宽；
7. R005 无法小样本学习 protect/harm 或双头被实现成互斥类别；
8. modelval 不存在任何非零策略同时改善 harm 且 recall loss ≤3 pp；
9. 所有非零 `P_j` 均未通过、最终只能结构性回退 P0；此时结论是“当前 scorer 没有安全自动删除作用”；
10. decision-dev harmful reduction CI 跨 0；可以记录趋势，但不得作为正式 Selector 提升；
11. 任一数据集 recall loss point >1 pp 或 CI upper >3 pp；
12. 不优于逐题 count-matched random/bottom-rank；
13. 三种子任一出现 recall loss point >3 pp；
14. source sidecar 缺失时不得假装 Candidate metadata 足够完成来源分组；
15. sealed/heldout 结果出来后不得重新调参并再次称作同一正式测试；
16. 所有非零策略失败时，TopK10 保持唯一生产默认。

---

## 13. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| harmful 只是合成 counterfactual | 不能外推现实真伪 | 明确写 synthetic proxy；现实来源验证另立后续研究 |
| protect 标签稀疏 | 模型把未标注当负例 | masked loss；只使用明确 qrel/counterfactual/irrelevant |
| 小提升 CI 跨 0 | 有趋势但不能强结论 | paired query 设计、完整开发集、三种子；不降低 recall 红线换显著性 |
| 自适应策略过度删除 | 重演 Beam | P0 fallback、min_keep=7、max_delete≤3、CRC、decision gate |
| 固定 cap 过于保守 | 漏掉多个明显 harmful | 同时比较 cap1/2/3；每题实际 0–cap，自适应停止 |
| source metadata 不完整 | 来源规则错误 | 使用 hash-pinned ParentIndex；缺失时 ABSTAIN_KEEP |
| 历史 artifacts 不在本地 | 本机无法直接运行同池实验 | R001 已在原 HPC 做到六池 exact recovery；后续原地 repackage v2，不重新检索 |
| Hybrid 目标与现有 BM25 pin 冲突 | formal 无法验证或比较换池 | 新建严格 Selector-v2 Hybrid manifest；无法完成就训练前 STOP/amend |
| 同 family/parent query 被当独立样本 | CRC 保证失效、CI 过窄 | 每 component 一个 CRC 代表；CI 做 cluster bootstrap；报告有效 component n |
| 2Wiki 用全部 Top20 parent 分组 | distractor 形成伪相关巨型 component | component 只用 official supporting/gold parent；candidate parent 不建边 |
| 2Wiki 跨官方 split 共享 supporting parent | 可能存在 parent 记忆效应，且不能诚实宣称 parent overlap=0 | 保持官方 heldout 不重分；报告 overlap，并给 parent-seen/unseen 与 unseen-only 敏感性结果 |
| chain-eligible component 少于99 | α=1% 时非零策略数学上不可能通过 | R001/R002 先计数；训练前 STOP 或事前 amendment |
| Selector protocol 没有 checklist/gold | 无法逐题保证链完整 | protect score + 离线 conditional chain risk；set model 推迟 |

---

## 14. 最终决策表

| 结果 | 决策 |
|---|---|
| scorer 没有安全角落 | 停止学习型删除，TopK10 保持默认 |
| harmful point 改善但 CI 跨 0 | 记录探索性信号，不宣称可靠提升 |
| harmful CI 通过但 recall 超线 | 失败；不能用删错效果抵消正确证据损失 |
| C1 通过、复杂消融无贡献 | 采用通过门槛的最简单方案，可能是 0–cap1 |
| C1 通过、adaptive 明显更优 | 进入三种子和 formal |
| source grouping 无额外价值 | 降级为 appendix，不影响 C1 |
| sealed/heldout 失败 | TopK10 保持生产默认 |
| 全部门槛通过 | 才提出注册 `adaptive-risk` 和端到端 Generator 验证 |

---

## 15. Final Checklist

- [ ] TopK10 是同池、同 query 主锚点。
- [ ] R007 已从 0–cap1/2/3 冻结唯一最终 cap；实际删除不是强制删满。
- [ ] R007 按冻结 `ε_harm` 和复杂度顺序选 family/cap；R008 没有重新做方法选择。
- [ ] `ABSTAIN_KEEP` 仍计入 harmful exposure。
- [ ] protect 是正确 required/supporting 保护分数，不是普通相关性。
- [ ] 未标注 2Wiki 候选没有被自动当负例。
- [ ] 2Wiki component 只由 official supporting/gold parent 建立，没有把所有 Top20 candidate parent 当作关系边。
- [ ] 2Wiki 跨官方 split supporting-parent overlap 已如实报告；heldout 未重分，正式结果附 parent-seen/unseen 敏感性分析。
- [ ] harmful reduction 固定为 TopK−Selector，正数代表改善。
- [ ] paired cluster-bootstrap 95% CI、plus-one p-value 和严格 query 对齐已实现。
- [ ] recall/chain 全部在 `document_id` 集合上计算。
- [ ] chain 只使用 TopK10 chain-eligible 条件分母。
- [ ] expected-risk CRC 每 component 一个预冻结代表，使用 `（ΣL+1）/(n+1)`，四风险取最保守策略，并与 95% CI 分开解释。
- [ ] recall 目标损失 ≤1 pp，硬 CI 上限 3 pp。
- [ ] count-matched random/bottom-rank 已覆盖。
- [ ] 当前 frozen contracts 未为 trace 被破坏。
- [ ] source sidecar 和 CRC artifact 内容 hash 已写入 v2 experiment manifest。
- [ ] Hybrid pool 使用独立 v2 manifest；没有误用 BM25-only pin。
- [ ] experiment runner 已输出 decision trace，且未破坏生产 Selector protocol。
- [ ] 只有 formal PASS 后才注册生产 Selector。
