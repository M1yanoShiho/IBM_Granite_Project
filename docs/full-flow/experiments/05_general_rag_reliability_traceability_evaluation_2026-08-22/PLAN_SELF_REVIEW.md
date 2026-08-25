# Experiment 05 — 计划自一致性审查

> **历史文档说明（2026-08-24）：** 本文件保留的是执行前 v3 计划复审快照，因此下文的
> “实验尚未启动”只描述当时状态。实验后来按用户授权的 v4 快速方案执行：完整语料 BM25
> Top-1000 后由 Granite 做候选内 dense scoring/RRF，不声称构建独立全语料 dense index。
> Goal 1–5 已全部完成，最终技术状态见 [TRACKER.md](TRACKER.md) 与
> [`results/final/`](results/final/)。

**审查日期：** 2026-08-22  
**审查对象：** [PLAN.md](PLAN.md) v3 reviewed  
**结论：** `INDEPENDENT FULL-PLAN RE-REVIEW PASS / READY FOR HANDOFF / EXPERIMENT NOT STARTED`

v3 已同步用户在逐部分讲解中确认的诊断层级修改。第一次独立全计划审核发现三个阻断项：gold 解锁时点不一致、Provence 裁剪后文本边界不足、科学标签存在自由解释空间；逐项修复后，同一独立 agent 已从头复审四份文件并给出 `PASS`，未发现新的交接阻断。该 PASS 只代表计划可交接，实验尚未启动。

## 1. 用户已经明确的决定是否落实

| 决定 | 计划位置 | 结果 |
|---|---|---|
| 正好三个普通 RAG 数据集 | §3 | PASS：KILT–NQ、KILT–TriviaQA、ALCE–ASQA |
| 不把多跳/噪声压力任务作为新主实验 | §3.1、§12 | PASS |
| baseline 不更换，沿用上一轮完整矩阵 | §4.1 | PASS：Dense/Hybrid/Granite Rerank/Provence/Ours |
| 只做三个模块级消融 | §4.2 | PASS |
| 不增加 TRUE 内部消融 | §4.2、§12 | PASS |
| Selector 去掉最多删两条 | §5 | PASS |
| Selector 同时去掉 minimum retained | §5 | PASS |
| 只有 harm 高且 protect 低才删除 | §5 | PASS |
| 全部删除时不强留，直接 abstain | §5 | PASS |
| 不再用短 gold 与长回答的 token precision 决定质量 | §7 | PASS |
| Table 1/2 使用同一评分体系 | §7.2、§8.4 | PASS：两表均只含六个主指标 |
| 流程信息不与主指标混层 | §7.3 | PASS：ER@10/SELR/CRR 仅放附录 |
| 每个 Goal 专注单一阶段，PASS 后自动接力 | §9–10 | PASS |
| 用户审查前不得执行 | 文首、Goal 0、TRACKER | PASS：全部执行 Goal 为 NOT STARTED |

## 2. 指标是否互相矛盾

### 答案正确性与答案长度

`RFC` 按 reference fact 是否被语义正确表达计分，不计算长回答相对于短 gold 的 token precision。因此正确扩展说明不会因“多写了词”被扣分；额外错误事实则由 `UCR` 单独惩罚。

### 正确性与可追溯性

- `RFC`：事实是否答对；
- `VRFC`：答对的事实是否还有有效引用支持；
- `CP/CR`：引用链接是否准确、是否覆盖需要引用的 claims；
- `UCR`：即使没有正确匹配 reference fact，回答中额外生成的事实是否有证据；
- `RR`：防止系统靠大量 abstain 得到虚假的低 UCR。

六者角色不重复，也没有再造旧 RAR 式全有或全无主指标。

### Selector 的可见性

- 下游因果证据：Full 与 `w/o Selector / Keep-all Top10` 的 RFC/VRFC/UCR/CP/CR/RR 配对差异决定 Selector 是否有系统价值。
- 附录机制解释：`SELR` 直接报告正确支持证据误删率；`CRR` 只报告上下文删除量，不作为质量分数。

因此 Selector 既没有从整体实验中消失，也没有要求读者合成两个诊断数字才能判断质量。

## 3. 公平性检查

| 风险 | 计划中的控制 | 结果 |
|---|---|---|
| 只让 Ours 输出引用 | 同数据集所有系统使用相同回答说明、引用格式和 schema | PASS |
| TRUE 给自己评分 | MiniCheck scorer-only；TRUE 禁止参与评分 | PASS |
| 通过拒答刷低 UCR | UCR 与 RR/RFC/VRFC 联合判断；abstention 在后三者记 0 | PASS |
| 用未实际进入 prompt 的 evidence 给答案补依据 | UCR/VRFC/CP/CR 只读取带 ordinal/URI/hash/token 的 `presented_evidence_records` 精确文本 | PASS |
| Provence 保留 ID 却裁掉支持句 | SELR 与主 scorer 均按 post-pruning 精确文本，而非 canonical ID 判断 | PASS |
| fact 与无关引用在答案不同位置却通过 | VRFC 要求实现该 fact 的同一个 claim 具有有效支持引用 | PASS |
| 重复 claim/citation 稀释错误 | semantic claim 去重；CP 使用唯一 `(claim_id,evidence_id)` | PASS |
| Retriever 与 Selector 错误混淆 | SELR 只以 Retriever 已找到的支持单元作分母 | PASS |
| 看结果后换指标/数据 | formal 前冻结 IDs、hash、scorer 和 gates | PASS |
| 公开数据此前被开发使用 | Goal 1 先做历史暴露 ID 注册与排除 | PASS，执行时有硬 gate |
| 使用 gold/oracle passage 作为检索池 | full-corpus retrieval；禁止 oracle Top-5 | PASS |
| 只报最好 seed | Ours seeds 13/42/73 全部报告 | PASS |
| 负结果被当作执行失败 | 技术 PASS 与科学 supported/not supported 分开 | PASS |

## 4. 可执行性检查

| 项目 | 已固定内容 | 结果 |
|---|---|---|
| 数据规模 | 3 × 400 formal；每集 120 revealed dev | PASS |
| 系统臂 | 4 baseline + 3 Ours seeds + 3 ablations | PASS |
| 正式输出量 | 8,400 + 3,600 = 12,000 | PASS |
| Retriever/Selector/Generator 预算 | Top10、2,304 input tokens、256 new tokens、greedy | PASS |
| Selector 谓词与失败行为 | exact threshold、无数量上限、全删 abstain、异常 keep-all | PASS |
| 统计 | query-paired bootstrap 10,000、seed 13、三 seed-level mean±SD、主 UCR claim-micro | PASS |
| UCR 零分母 | UNDEFINED/NOT ESTIMABLE 导致 Claim A NOT SUPPORTED，不误报技术失败 | PASS |
| scorer validation | 160 FactMatch + 160 support + 60 answers/≥180 claims + 80 contract cases，固定数值 gates | PASS |
| 结果表 | Table 1/2 同为六主指标；诊断只在 appendix | PASS |
| 阶段边界 | Goal 0–5 的依赖、交付、PASS 和禁止项 | PASS |

## 5. 执行前仍需由 Goal 1 验证、但不应在计划阶段假装已经完成的事项

以下不是计划漏洞，而是明确的执行 gate：

1. 三个数据集各自是否确有 400 个未用于本项目开发的 canonical IDs；
2. KILT 与 ALCE full-corpus index 的本地可用性、版本和 hash；
3. KILT page-to-passage provenance mapping 是否完整；
4. scorer 在简短正确与扩展正确回答上的长度公平 gate；
5. scorer-only claim extraction 和 MiniCheck 对 synthetic/revealed fixtures 的准确性；
6. 10 个运行臂在每集固定 5-query revealed smoke 上是否满足统一输出契约；
7. 360 条 development Retriever/Selector 诊断是否严格符合双门槛与 presented-evidence 边界。

这些事项只能在用户于交接后的新对话中明确发出“开始执行”指令后，分别由 Goal 1/2 完成。目前不得以“预检查”为名提前执行。

## 6. 已审查并冻结的五个决定

1. 是否接受三个数据集固定为 KILT–NQ、KILT–TriviaQA、ALCE–ASQA；
2. 是否接受 formal 每集 400、development 每集 120；
3. 是否接受六个主指标 `RFC/VRFC/UCR/CP/CR/RR`；
4. 是否接受主比较以 `Hybrid RAG` 为首要参照，其他 baseline 完整保留；
5. 是否接受 Claim A/B 的 `5pp` non-inferiority margin 与“至少 2/3 数据集显著改善”规则。

上述五项已经在本轮逐部分讲解、修改与完整复审中冻结。若未来要改变其中任何一项，必须先显式修订计划，不能在正式结果产生后临时更换。

## 7. 只读审查记录

| Review | 首轮发现 | v2 修复 | 最终结果 |
|---|---|---|---|
| v2 执行与阶段边界 | UCR/评分规则不唯一；120 dev 来源不完整 | 固定 dev/formal hash 顺序、零分母规则、具体评分和 bootstrap | PASS |
| v2 指标与统计 | prompt 实际证据边界、fact-claim-citation 对齐、三 seed/UCR CI 不完整 | 加 `presented_evidence_ids`、唯一 link/dedup、locked gates、完整三 seed bootstrap 与 harm guard | PASS |
| v3 用户审查修改 | SPR/CRR 容易被误解为必须联合判断的质量指标 | SPR 改为直观的 SELR；CRR 降为无方向描述；三项诊断全部移入附录 | PASS |
| v3 独立首审 | gold 解锁时点、Pruning 后文本边界、三值标签边界有阻断 | Goal 3/4 只生成，Goal 5 一次性解锁；冻结 exact presented text；穷尽式三值规则 | FAIL → REPAIRED |
| v3 独立完整复审 | 从头复核 PLAN/TRACKER/README/SELF_REVIEW 一致性与可执行性 | 无新增修改要求 | PASS |

所有审查均为纯文档审查，没有下载、切分或读取 formal 内容，没有运行 Retriever、Selector、Generator、TRUE、MiniCheck 或任何 smoke。
