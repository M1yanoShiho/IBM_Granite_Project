# R005 失败后只读诊断：为什么不能只调阈值

**日期：** 2026-08-12  
**状态：** `POST-HOC DIAGNOSTIC ONLY`  
**数据边界：** 只读正式 R005 已公开的 train-fit sanity 分数、冻结 R004 标签/provenance 和实现代码  
**明确未做：** 未查看 train-modelval/dev/sealed/heldout 效果；未训练模型；未推导可部署阈值；未改变 R005 `FAIL`

## 1. 零基础结论

可以把模型的分数想成把证据放在一条从“应保留”到“可能有害”的尺子上。若问题只是分界线放错了，我们挪动阈值后，正确证据和错误证据应当能分开。

R005 的真实情况是：大部分顺序已经正确，但少数正确/错误证据在尺子上 **互相穿插**。所以无论把同一条分界线放在哪里，都不能同时达到原先要求的两类准确率。这意味着：

1. 只调 `0.5` 阈值不够；
2. 只做单调 calibration 也不会改变样本先后顺序，因此不够；
3. 下一步应先修复/比较模型表示与训练目标，再谈删除阈值；
4. R005 的排序信号仍然有用，但只能用于形成新实验假设，不能事后改判 PASS。

## 2. 数据与身份复核

- `candidate_scores.jsonl` 共 640 行：NIAH 320 + 2Wiki 320。
- 每个数据源恰好 16 个 query，每题 20 个候选；全部为 `train-fit`。
- 前 10 名 320 行属于 train-fit quantile/overfit 范围，11–20 名 320 行只用于 sanity overfit 扩展。
- 32 个 query 与 `sanity_sample.jsonl` 完全一致；重新执行冻结 SHA-256 抽样规则后，两个数据源的 first-16 和 digest 均完全一致。
- 640 个 `(dataset, query, evidence)` identity 唯一；candidate、label、document、rank、role、text hash 全部连接一致。
- 640/640 的 `safe_score = min(harm_score, 1-protect_score)`。
- NIAH 的 16 组 clean/counterfactual provenance、全文 hash、mutation span、gold alias 和单处替换均复核通过；未发现明显数据或标签实现错误。

## 3. 原门槛复算

| 来源 | 头 | 标签 | 正确 / n | 准确率 | 原门结果 |
|---|---|---:|---:|---:|---|
| 2Wiki | protect | 1 | 32/32 | 1.0000 | PASS；仅代表官方 protect 正类 |
| NIAH | protect | 0 | 16/16 | 1.0000 | PASS |
| NIAH | protect | 1 | 71/79 | 0.8987 | FAIL |
| NIAH | harm | 0 | 12/16 | 0.7500 | FAIL |
| NIAH | harm | 1 | 14/16 | 0.8750 | FAIL |

全部 14 个 head-label 错误集中在 8 个 NIAH query、10 个候选上；其中 4 个 clean candidate 同时被 protect 和 harm 两头判错。错误没有显示为全局随机崩坏，而是集中在少量困难样本。

## 4. 分数分布与排序能力

| 来源 / 头 / 标签 | n | 最小 | Q25 | 中位数 | Q75 | 最大 |
|---|---:|---:|---:|---:|---:|---:|
| 2Wiki protect:1 | 32 | 0.992958 | 0.996604 | 0.997311 | 0.997886 | 0.998400 |
| NIAH protect:0 | 16 | 0.048014 | 0.101523 | 0.196424 | 0.209064 | 0.300551 |
| NIAH protect:1 | 79 | 0.191138 | 0.909313 | 0.987901 | 0.994324 | 0.997907 |
| NIAH harm:0 | 16 | 0.040127 | 0.075402 | 0.122902 | 0.484081 | 0.557743 |
| NIAH harm:1 | 16 | 0.480203 | 0.540738 | 0.601315 | 0.836152 | 0.929738 |

NIAH 两类都存在，故可计算排序 AUC：

- protect ROC-AUC：`0.982595`；
- harm ROC-AUC：`0.929688`。

AUC 较高表示“随便抽一个正类和一个负类，模型通常把两者顺序排对”；它不等于存在一条能把所有类别同时切好的阈值。

## 5. 为什么任何单一阈值都救不了原门

预测规则是 `score >= threshold` 判为正类。原门要求每类准确率至少 `0.95`。

### Protect 头

- 16 个负类要达到至少 95%，实际必须 `16/16` 正确，因此阈值必须高于最高负类分数 `0.30055094`。
- 79 个正类最多只能错 3 个，因此阈值必须不高于第四低正类分数 `0.21464671`。
- 所需区间变成 `(0.30055094, 0.21464671]`，是空集。

### Harm 头

- 正负两类各 16 个，至少 95% 都等价于 `16/16`。
- 负类全对要求阈值高于最高负类 `0.55774343`；正类全对要求阈值不高于最低正类 `0.48020273`。
- 所需区间 `(0.55774343, 0.48020273]` 也是空集。

即使事后选择“让两类中较差准确率尽可能高”的阈值，protect 的较差类也最多为 `15/16 = 0.9375`，harm 最多为 `13/16 = 0.8125`。因此 calibration-only 不是足够的新方案。

## 6. 配对方向为什么全对，但绝对分类仍失败

16 个 NIAH clean/counterfactual pair 的 protect、harm 和 safe 相对方向全部正确。但最小 margin 很小：

| Margin | 最小 | 中位数 | 最大 |
|---|---:|---:|---:|
| Protect | 0.000610 | 0.480644 | 0.871872 |
| Harm | 0.001140 | 0.516781 | 0.866053 |
| Safe | 0.001140 | 0.527457 | 0.866053 |

最困难的 query 10049、10400、10679 等 clean/counterfactual 虽然顺序没有颠倒，但差距只有约 `0.0006–0.0024`。这像两张纸的上下顺序是对的，但几乎贴在一起；加入其他 query 后就无法用一条全局线稳定切开。

## 7. 一个已确认的实现事实

冻结模型 snapshot 声明自己是 `DebertaV2ForSequenceClassification`，并包含预训练/微调后的 `pooler.dense.*` 与三分类 `classifier.*` 参数，标签映射为 contradiction / entailment / neutral。

当前 R005 实现却：

1. 通过 `AutoModel` 只加载 `DebertaV2Model` 基础 encoder；正式环境加载日志确认 pooler 与 classifier 四个参数未被采用；
2. 直接取 `last_hidden_state[:, 0, :]`；
3. 在其上随机初始化两个 `Linear(768,1)` head；
4. encoder 与两个随机 head 共用同一个 `2e-5` 学习率。

这是可以核实的实现事实，不是对失败原因的猜测。它说明当前代码没有沿用该 cross-encoder 已训练好的 sequence-classification pooling 路径。它 **可能** 使模型对一处 span 修改不够敏感，但因果关系仍必须由新的预注册对照实验验证，不能直接宣称这就是根因。

## 8. 假设优先级

1. **表示重叠 / 对一处 span 修改不够敏感：支持最强。** 少数 pair margin 接近 0，且正式与旧 unverified 重复得到相同困难样本。
2. **训练目标与门槛不一致：支持较强。** weighted average BCE 明显下降，却没有直接优化最差类别或 pair margin。
3. **阈值/校准位置：只有部分作用。** 0.5 可能不是最佳位置，但无单一阈值可通过原门，故不是完整答案。
4. **小样本或单种子波动：仍可能。** 每个 NIAH 二分类小类只有 16 个，95% 变成全对；但同样本的两次代码版本分类完全一致，暂不支持纯数值不稳定。
5. **provenance/标签/identity 缺陷：本次复核不支持。** 这不代表永远不可能，只代表已检查证据中没有发现。

## 9. 旧实验哪些保留、哪些不能复用

### 保留

- R001–R004 的 pool、component、指标、标签/mask、输入隔离与资源证据；
- R005 的 fail-closed runner/finalizer、artifact schema、checkpoint/hash/verify 链；
- 16/16 配对方向、AUC、困难 query 和 score overlap，作为新假设的依据；
- TopK10 默认与 P0 结构性 fallback。

### 不能复用为成功证据

- R005 checkpoint、事后阈值、困难样本或其分数不能进入新的正式效果选择；
- 旧 unverified run 只能说明同样本代码修订稳定，不能冒充独立 seed/样本；
- 当前诊断不能升级 R005，也不能直接解封 R006。

## 10. 新实验可用的未触碰空间

排除所有含 R005 query 的 component 后，train-fit 仍有充足的 component-disjoint 候选：

- NIAH：约 893 个 fresh query；其中 846 个具备严格 clean/counterfactual pair，分布于 766 个可用 fresh component；
- 2Wiki：约 2,677 个 fresh protect-positive query，分布于 2,078 个 fresh component。

这允许新的 sanity 使用完全不同的 component，而不必查看 train-modelval 或重复在已经看过的 32 题上调参。具体抽样数量、salt、种子、variant 和门槛必须在运行前写入新的 amendment。

## 11. 稳定性旁证的边界

旧 `R005.unverified-26fa4221` 与正式 R005 具有相同 640 candidate identity/text hash：两头 640/640 二值分类一致，14 个错误完全相同，NIAH 16 个 pair 排序全部相同；平均 query 内 Spearman `0.999953`。

这支持“失败不是一次偶然浮点抖动”，但因为样本和 seed 相同，它不能证明跨种子、跨样本稳定，也不能被合并成第二个正式实验。
