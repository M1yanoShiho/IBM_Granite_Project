# 三分类 Beam Selector 实验计划

**日期：** 2026-08-09

**状态：** 待数据与代码检查完成后冻结

**范围：** 只验证 Selector；Retriever 固定，Generator 不参与

## 1. 一句话目标

在完全相同的 Hybrid RRF Top-20 候选池上，训练一个按证据组合逐步选择的三分类模型，争取比 TopK 少保留误导证据，同时把必要证据损失控制在合理范围。

这不是要追求巨大提升。TopK 已经有较高的必要证据召回率，新方法的价值在于用很小的信息损失换来稳定、可证明的误导暴露下降。

## 2. 必须证明的两项主张

| 编号 | 主张 | 最低可信证据 |
|---|---|---|
| C1 | Selector 能识别一部分高度相关但误导的证据 | 相同 Top-20 下，harmful-in-context 比 TopK 至少下降 3 个百分点，且 95% CI 上界小于 0 |
| C2 | 过滤没有过度破坏回答所需的信息链 | 同时报告 harmful 降幅与 recall 损失的权衡曲线；recall 损失不再用 1% 一刀切，按 §7 的三级标准判断 |

**需要排除的错误解释：** 提升来自 Retriever、Generator、候选数量变化或 sealed600 调参。为此，Retriever、Top-20、最大输出 10 条和 Generator 接口全部固定；sealed600 只在最后运行。

## 3. 原方法与项目改造边界

### 3.1 Beam Retrieval 原来解决什么

Beam Retrieval 面向多跳问答：从 10–20 条候选 passage 中，逐步选择一条完整的 supporting passage chain。每一步都把“问题＋已经选择的证据＋下一条候选”一起判断，并保留多个可能的部分路径，减少第一步选错导致整条证据链丢失的风险。

论文与代码：

- https://aclanthology.org/2024.naacl-long.96/
- https://github.com/canghongjian/beam_retriever

### 3.2 原方法不能直接解决什么

原论文只区分 relevant / irrelevant，没有用“内容高度相关但关键事实错误”的候选进行验证。它在 2Wiki 上的高分只能证明多跳证据选择能力，不能证明反事实过滤能力。

### 3.3 本项目只做一项必要改造

将两个二分类 head 改为相同语义的三分类 head：

- `REQUIRED`：回答问题必须保留的证据；
- `HARMFUL`：高度相关但会把回答引向错误事实的证据；
- `IRRELEVANT`：与正确回答无帮助的普通干扰证据。

模型仍然只选择原始 passage ID，不抽取新 claim、不改写证据、不生成答案。

## 4. 冻结的模块接口

### 输入

- `Query`；
- 同一 Hybrid RRF 产生的 `CandidateSet`；
- 每题恰好 Top-20；
- 保留 `evidence_id`、`document_id`、`source_parent_id`、原始 retrieval rank/score 和原文。

### 输出

- 最多 10 个原始 `evidence_id`；
- 每个候选的三类概率；
- 最终选择顺序与拒绝原因；
- 模型版本、阈值、seed 和候选池 SHA。

### 明确禁止

- 不调用最终 Generator；
- 不让 Selector 生成或改写证据文本；
- 不使用 sealed600 标签、`cf::` 前缀或 provenance 字段作为模型输入；
- 不改变 Hybrid RRF；
- 不把 `source_parent_id` 当“来源可信度”。它只用于泄漏审计和同源记录。

## 5. 模型与选择规则

### 5.1 模型

- 编码器：`cross-encoder/nli-deberta-v3-base`，在 M0 冻结具体 revision 与文件 SHA；
- 第一跳 head：输入“问题＋候选证据”，输出三类概率；
- 后续 head：输入“问题＋已经选中的证据链＋新候选”，输出三类概率；
- beam size：2；
- 最大选择数：10；
- 训练损失：各 hop 三分类交叉熵之和；
- 三个正式训练 seed：13、42、73。

这里的 `seed` 是“随机种子”。训练神经网络时，初始参数和数据出现顺序带有随机性；同一套代码训练三次，结果可能稍有不同。先跑一个 seed 是低成本试跑，只判断方法是否值得继续；通过后跑三个 seed，是为了证明结果不是某一次碰巧运气好。TopK 没有训练随机性，因此不需要三个 seed。

选择 DeBERTa 而不是强行使用 Granite 的原因：这里做的是小候选池证据分类与组合选择，不是答案生成。IBM 项目不要求每个子模块都使用 Granite；Generator 继续由原同学维护。

### 5.2 保守输出规则

模型不直接把 TopK 全部推翻，而是在 TopK 基础上做有限替换：

1. Beam 路径先提出一组 `REQUIRED` 证据；
2. 这些证据优先进入最终集合；
3. 其余位置按原 Hybrid RRF 顺序补齐；
4. 只有候选同时满足“`HARMFUL` 或 `IRRELEVANT` 高置信”且“`REQUIRED` 概率低”时才允许排除；
5. 若模型不确定，保留原 TopK 顺序，避免为追求过滤而误删必要证据；
6. 最终最多 10 条，不足时不得使用 Generator 或参数知识补证据。

阈值只能在开发集选择。候选网格在第一次开发运行前写入 frozen config，并画出“减少多少 harmful、同时损失多少 recall”的权衡曲线。先删除同时在两个指标上都更差的配置，再选择 recall 损失较小、harmful 降幅明显的拐点。不能看 sealed600 的结果再改阈值。

## 6. 数据准备

### 6.1 NIAH：学习正确与反事实的区别

| 数据角色 | 标签 |
|---|---|
| 官方 gold/needle passage | `REQUIRED` |
| mutation log 对应的 counterfactual twin | `HARMFUL` |
| Top-20 中其他未标注候选 | `IRRELEVANT` |

现有服务器目录的 `split` 字段不能直接相信：历史交接记录表明多个目录都写成 `dev`，并出现过 query/family 重合。因此 M0 必须重新物化 Selector 专用 train/dev manifest：

- 按 `query_id + source_parent_id + synthetic_family` 分组划分；
- train、dev、sealed600 三方在这三个轴上零重合；
- sealed600 的 query、parent 和 family 只用于排除泄漏，不用于训练；
- `cf::` ID 与 provenance 只用于生成训练标签和离线评估，必须在模型输入前剥离。

### 6.2 2Wiki：学习保留多跳证据

- 官方 train：训练；
- 官方 dev：阈值与开发 Gate；
- test 或预先冻结的 held-out split：最后一次正式检查；
- supporting documents 为 `REQUIRED`；其余 Top-20 为 `IRRELEVANT`；
- 2Wiki 不制造 `HARMFUL` 标签，它只负责约束多跳召回不能被破坏。

### 6.3 训练混合

- NIAH 与 2Wiki 采用 1:1 的 batch 来源交替，避免大数据集压倒另一任务；
- 每道题保留全部 `REQUIRED` 和 `HARMFUL`；
- `IRRELEVANT` 从 Top-20 的高排名困难负例中按固定 seed 采样；
- 数据比例、采样数和哈希必须在数据准备阶段结束时冻结，不得根据 sealed600 结果修改。

## 7. 实验阶段与停止顺序

### M0：代码与数据冻结

1. 先同步服务器正式实验提交 `39e57393`；
2. 运行现有 TopK 与 Selector 相关测试，保存干净基线；
3. 建立 NIAH Selector train/dev 和 2Wiki train/dev Top-20；
4. 完成 query、parent、synthetic family 零泄漏审计；
5. 冻结模型 revision、候选池 SHA、标签统计和阈值网格。

**停止条件：** 任一标签无法映射、Top-20 数量错误、训练与 sealed600 有任何重合，均不得训练。

### M1：最小正确性检查

1. 用 32 题运行完整训练和推理；
2. 确认模型能够过拟合这 32 题；
3. 确认输出 ID 都来自输入 Top-20；
4. 确认 Selector 全程没有调用 Generator；
5. 确认未把 `cf::`、provenance 或 gold answer 泄漏给模型。

**通过条件：** 32 题训练集分类/选择正确率至少 95%。未通过说明实现或标签管线有问题，不允许通过增加模型或延长正式训练掩盖。

### M2：单 seed 可行性检查

只训练 seed 13，依次在 NIAH dev 和 2Wiki dev 比较 TopK。这只是便宜的“值不值得继续”检查，不作为最终结论。

先看两件事：

- NIAH harmful-in-context 至少下降 3 个百分点，95% CI 上界 < 0；
- 同时记录 NIAH required recall 和 2Wiki supporting recall 各损失多少。

判断分为三级，而不是用 1% 一刀切：

| 单 seed 结果 | 判断 | 下一步 |
|---|---|---|
| harmful 没有可信下降，或任一 recall 损失超过 5 个百分点 | 明确失败 | 停止，不跑另外两个 seed；TopK 保留为最终方法 |
| harmful 可信下降，两个 recall 损失都不超过 3 个百分点 | 清晰通过 | 进入三 seed 稳定性检查 |
| harmful 可信下降，但某个 recall 损失在 3–5 个百分点 | 有用但存在取舍 | 允许进入三 seed，确认这种取舍是否稳定，不能提前宣布成功 |

Evidence precision 继续报告，但不再因为一次很小的波动直接判死刑。

### M3：三 seed 稳定性

在同一冻结配方上训练 13、42、73 三个 seed。报告 mean ± standard deviation，不能只挑最好的一次。最终看三次的平均结果和波动：

- 平均 recall 损失不超过 3 个百分点，且没有一次超过 5：稳定、清晰通过；
- 平均损失在 3–5 个百分点：稳定但属于 reliability/completeness 取舍；
- 平均损失超过 5，或不同 seed 波动很大：失败。

### M4：一次性正式测试

只在全部配置冻结后运行：

- sealed600：主测误导过滤和必要证据保护；
- 2Wiki held-out：主测多跳证据保护；
- TopK 和新 Selector 读取完全相同的候选文件；
- 每个训练 seed 运行一次；
- 看见正式结果后不再改阈值、训练比例或选择规则。

### M5：报告与代码 cutover

- 生成三张主表、per-query 文件、manifest 和 checksums；
- 写明通过或失败，不只报告有利指标；
- 无论新方法通过还是失败，都删除已否决的 MIS 与旧 Graph/corroboration 运行代码；
- 新方法通过则注册三分类 Beam Selector；失败则只保留 TopK。

## 8. 指标与统计

### 决策指标

- `Harmful-in-context (%)`：越低越好；
- `Required-evidence recall (%)`：越高越好；
- `2Wiki supporting-document recall (%)`：越高越好；
- `Evidence precision (%)`：越高越好。

### 诊断但不决定成败

- 平均选择条数；
- 三类混淆矩阵；
- 不同 hop 的错误分布；
- backend failure 数；
- 实际训练时间与峰值显存只记录在运行日志，不进入核心贡献表。

### 统计方法

- 统计单位是 query；
- 同题 TopK 与新 Selector 做配对比较；
- 10,000 次 query-level paired bootstrap，固定统计 seed `20260809`；
- 报告差值和 95% CI；
- 三个训练 seed 报告 mean ± standard deviation；单 seed 结果只用于开发阶段止损，不进入最终主表。

## 9. 最终需要呈现的三张表

### 表 1：候选池前置检查

| 指标 | sealed600 | 2Wiki |
|---|---:|---:|
| Questions (#) | 待实验 | 待实验 |
| Candidates per query | 20 | 20 |
| Required/supporting recall@20 (%) ↑ | 待实验 | 待实验 |
| Harmful pool-hit (%) | 待实验 | N/A |
| Label mapping failures (#) | 待实验 | 待实验 |
| Candidate pool SHA | 待实验 | 待实验 |

### 表 2：sealed600 Selector 核心结果

| 指标 | TopK | 三分类 Beam Selector | 差值 |
|---|---:|---:|---:|
| Harmful-in-context (%) ↓ | 待实验 | 待实验（mean ± sd） | 待实验 [95% CI] |
| Required-evidence recall (%) ↑ | 待实验 | 待实验（mean ± sd） | 待实验 [95% CI] |
| Evidence precision (%) ↑ | 待实验 | 待实验（mean ± sd） | 待实验 [95% CI] |
| Selected evidence (#) | 待实验 | 待实验（mean ± sd） | 待实验 |

### 表 3：2Wiki 多跳保护结果

| 指标 | TopK | 三分类 Beam Selector | 差值 |
|---|---:|---:|---:|
| Supporting-document recall (%) ↑ | 待实验 | 待实验（mean ± sd） | 待实验 [95% CI] |
| Evidence precision (%) ↑ | 待实验 | 待实验（mean ± sd） | 待实验 [95% CI] |
| Selected evidence (#) | 待实验 | 待实验（mean ± sd） | 待实验 |

## 10. 结果存储

服务器运行目录：

```text
runs/selector-beam-v1/
├── data/
│   ├── niah-train/
│   ├── niah-dev/
│   └── twowiki/
├── models/
│   ├── seed-13/
│   ├── seed-42/
│   └── seed-73/
├── dev/
├── formal/
└── logs/
```

回传到项目的小型结果包：

```text
results/selector-beam-v1/
├── FINAL_REPORT_CN.md
├── pool_quality_table.csv
├── sealed600_selector_table.csv
├── twowiki_selector_table.csv
├── per_query_metrics.jsonl
├── manifests/
└── checksums.sha256
```

模型权重与大候选文件留在服务器，不提交 Git；Git 中只保留 manifest、表格、报告和校验值。

## 11. 最终裁决

| 结果 | 决定 |
|---|---|
| harmful 明确下降，平均 recall 损失 ≤3 个百分点，且无单次损失 >5 | 确定三分类 Beam Selector，进入系统集成 |
| harmful 没有可信下降 | 新方法没有证明比 TopK 有用，最终使用 TopK |
| harmful 下降，平均 recall 损失在 3–5 个百分点 | 方法有效但存在明确取舍；如项目优先 reliability 可采用，如优先完整性则保留 TopK，报告必须同时呈现两项变化 |
| harmful 下降但平均 recall 损失 >5 个百分点 | 仍属过度过滤，最终使用 TopK |
| 单 seed 好看但三 seed 不稳定 | 不采用，最终使用 TopK |

失败不是继续堆第四种方法的理由。本计划是 Selector 的最后一次有界验证；如果失败，模块以 TopK 和完整负结果结束。
