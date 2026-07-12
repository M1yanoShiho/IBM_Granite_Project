# ML Evidence Selector 交接入口

**当前状态：** V1 已结束并归档为探索性诊断；V2 尚未开始运行。

**接手目标：** 从冻结的 Selector 2.0 计划继续，不把 V1 的诊断数字当作最终 benchmark 结论。

> **FinanceBench 已暴露：** V1 已使用 FinanceBench 全部 150 个问题进行诊断。其状态固定为 `EXPOSED_DIAGNOSTIC_ONLY`，不得再用于 V2 调参、模型选择、停止决策或最终盲测。

## 阅读顺序

1. 本页：五分钟了解当前状态。
2. [V1 实验记录](V1_EXPERIMENT_RECORD.md)：已经做过什么、怎么做、观察到什么、为什么停止。
3. [V2 实验计划](V2_EXPERIMENT_PLAN.md)：重新训练 Graph-Assisted Selector 2.0 的冻结协议。
4. [V2 实验 Tracker](V2_EXPERIMENT_TRACKER.md)：从第一个未完成的 MUST Run 继续并记录每次运行。

## 五分钟摘要

V1 把 Evidence Selector 限定为固定接口：

```text
Query2Doc + Granite dense Top-20
→ Core/Full ML selector 重排序
→ Top-10 进入生成上下文
```

V1 已完成数据与候选构造、五级 utility 标签、特征缓存、LightGBM/Logistic 模型、三个随机种子、消融、负控以及 NIAH、RAMDocs、FinanceBench、ContractNLI 诊断。它没有通过预先定义的 Gate：

- NIAH 的 `NDCG@10` 和 required-evidence recall 提高，但 harmful rate 没有下降；
- RAMDocs 只有部分排序指标改善，未形成稳定的 harmful-evidence 收益；
- FinanceBench 没有显著改善，而且已经被提前使用；
- ContractNLI 跨域迁移明显失败；
- 因 Gate 1 失败，完整的 V1 RAG 答案实验按协议停止。

因此 V1 的价值是提供设计信息，而不是最终性能结论：相关性信号仍然主导模型；现有 support/judge 特征没有学会稳定过滤 harmful evidence；下一轮需要显式关系建模、严格 OOF 特征和新的未见最终评估。

V2 从零重新训练 selector，使用三类核心关系 `CLAIM_SUPPORTS`、`CLAIM_REFUTES`、`SAME_SOURCE`。它不能继承 V1 的“已验证过滤能力”表述，因为该能力尚未被证明。

## 结果保存在哪里

V1 可共享的精简证据已经进入 Git：

- [NIAH 主结果](../data/ml_selector_validation/pilot/results.json)
- [RAMDocs 结果](../data/ml_selector_validation/ramdocs/results.json)
- [FinanceBench 诊断结果](../data/ml_selector_validation/financebench/results.json)
- [ContractNLI 结果](../data/ml_selector_validation/contractnli/results.json)
- [全部 V1 审计与结果目录](../data/ml_selector_validation/)
- [三张 V1 结果图](../assets/ml_selector_validation/)

V1 的约 986 MB 全量中间产物位于老师的私有服务器 `it097952`：

```text
/scratch/fl25387/IBM_Granite_Project/results/ml_selector/
```

这些文件主要是候选池、JSONL 缓存、shards 和运行日志，没有迁移，也不是继续 V2 的依赖。接手人应从 Git 中的精简证据和可复现代码继续，而不是依赖该私有路径。

## 接手后的第一步

先阅读并冻结 [V2 实验计划](V2_EXPERIMENT_PLAN.md)，然后从 [V2 Tracker](V2_EXPERIMENT_TRACKER.md) 的 `R000` 开始。任何正式运行都必须同时记录 Git commit、命令/配置、data split hash、seed、host/job、精简输出路径、Gate 和一句话解释。

项目级历史检索结果仍保留在 [results-summary.md](../results-summary.md)；它不替代本目录的 Selector 交接记录。
