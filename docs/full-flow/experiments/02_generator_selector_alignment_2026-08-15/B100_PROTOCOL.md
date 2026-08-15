# B100 受控上下文矩阵协议

**日期：** 2026-08-15
**状态：** `IMPLEMENTATION IN PROGRESS / NOT RUN`

## 1. 目的与边界

B100 只诊断当前 Granite Base Verify-and-annotate Generator 在不同证据上下文下的行为，不训练模型、不改变正式系统，也不把 oracle 上下文当作可部署方法。

正式系统仍是：

```text
Retriever -> Selector -> Generator -> one answer
```

B100 是离线平行对照。每个上下文臂独立生成一次答案；它们不是正式系统先后回答多次。

## 2. 数据与样本

- 数据：已经揭示的 NIAH decision-dev 739 题；
- 诊断样本：全部 109 个 Legacy Selector changed 问题；
- 对照样本：从 630 个 unchanged 问题中一对一、无放回匹配 109 题；
- 不读取 sealed600，不读取 system held-out；
- 匹配只发生在生成前的离线准备阶段。

匹配使用以下生成前特征：问题疑问类型、reference answer 形态、TopK10 中支持/反事实/普通证据的数量与可见性、第一条支持证据位置、是否为单支持或多支持问题，以及冻结的 chain-eligible 标记。使用确定性的最小总代价一对一匹配；query ID 只用于稳定处理并列，不作为语义特征。报告匹配前后分布，不能只声称“已经匹配”。

## 3. 六个上下文臂

TopK10 中的证据按生成后可审计的官方文档标签分为：

- `support`：`document_id` 属于该题的 `relevant_document_ids`；
- `harmful`：NIAH 合成反事实文档，`document_id` 以 `cf::` 开头且 `source_uri` 位于 `synthetic://cf/`；
- `benign`：其余 TopK10 文档。

| 臂 | 上下文构造 | 诊断问题 |
|---|---|---|
| K | 原始 TopK10，保持 retrieval rank | 当前真实基线 |
| S | 冻结 Legacy Selector 输出 | 当前删除行为后的结果 |
| O | TopK10 中全部 support，保持相对顺序 | 没有干扰时 Generator 能否使用已找到的支持证据 |
| O+B | O 后追加排名最靠前的 `m` 条 benign | 等量普通噪声的影响 |
| O+H | O 后追加排名最靠前的 `m` 条 harmful | 等量有害噪声的影响 |
| O-P | 与 K 完全相同的十条证据；非 support 在前、support 在后，各组保持相对顺序 | 支持证据位置和重新编号的影响 |

其中 `m = min(benign_count, harmful_count)`。因此 O+B 与 O+H 的上下文长度、support 集合和 support 位置完全相同，差别只在追加的噪声类型。`m=0` 的题保留在总体矩阵中，但不进入 O+B 对 O+H 的噪声类型配对结论。

O-P 必须满足：证据 ID 集合与 K 完全一致、无重复、仅顺序变化。若 TopK10 没有 support，则该题记录为 Retriever bottleneck，不用 O 结果证明 Generator 能力。

## 4. Gold 隔离

B100 分成三个命令：

1. `prepare`：允许读取 gold，只构造匹配样本和 oracle 上下文；
2. `run`：只读取已经构造好的 runtime contexts，不接受 `--gold`，不包含 reference answer、relevant document IDs 或证据角色字段；
3. `score`：全部生成完成后才读取 gold、context audit 和答案。

Generator 在任何一臂只接收 `Query`、空的 `QueryChecklist.required_facts` 与该臂 `SelectedEvidenceSet`。Gold 决定 O/O+B/O+H/O-P 的离线证据排列，但不会作为文本、标签或 sidecar 进入 Granite 或 TRUE。

## 5. 模型、顺序与完整性

- Generator：`ibm-granite/granite-4.1-3b@c065040...`；
- Claim verifier：冻结 TRUE `google/t5_xxl_true_nli_mixture@aa6cfe...`；
- 六臂在同一进程共享同一个 Granite 与 TRUE 实例；
- 每题的六臂执行顺序由 seed 13 和 query ID 确定，在 12 种循环/反向顺序中轮换；
- 每次调用必须保存 A001 trace；错误率超过 5% 时整次运行无效；
- 正式输出目录必须为空，失败 smoke 与正式运行不能合并。

## 6. 指标与分层

每臂报告：

- answer match、coverage、运行错误；
- `empty_draft`、`splitter_no_claims`、`all_claims_unfaithful`、routing/assembly empty；
- gold-document citation precision；
- 当前上下文可见 support document 的 citation recall；
- support 已存在但答案错误的数量与比例。

配对报告至少包括 `S-K`、`O-K`、`O+B-O`、`O+H-O`、`O+H-(O+B)` 和 `O-P-K` 的点估计、component-cluster bootstrap 95% CI 与答案翻转。另报告逐题答案文本一致率和 answer-match 一致率。

预注册分层：

- Selector changed / matched unchanged；
- 单个可见 support / 多个可见 support；
- O+B/O+H 噪声等量对照 eligible / ineligible；
- TopK10 support visible / absent。

## 7. 产物与判定边界

机器产物：

- `runtime_contexts.jsonl`：run 唯一读取的数据；
- `context_audit.jsonl`：gold-derived 构造审计；
- `prepare_manifest.json`、`run_manifest.json`；
- `generations.jsonl`、`B100_cases.jsonl`、`report.json`；
- `B100_DIAGNOSTIC_REPORT.md`。

B100 只提供逐题机制证据。最终将问题路由到 Retriever、Generator context robustness 或 Selector 充分性，由后续 B110 按预注册规则完成；B100 本身不宣布新方法有效。
