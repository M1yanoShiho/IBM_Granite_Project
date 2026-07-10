# ML Evidence Selector 实验执行 Tracker

**分支：** `week5_MLSelector`

**计划：** `docs/ml-selector-experiment-plan.md`

**当前状态：** 验证实验已按 Gate 1 停止规则收束；会议报告与可追溯结果已生成

## 执行记录

| 阶段 | 实际完成内容 | 产出 | 状态 |
|---|---|---|---|
| M0 协议 | 冻结五级 utility、top-20→10、公平条件、指标和 Gate | `data/selector/protocol_manifest.json` | DONE |
| M1 数据 | 固定 NIAH 父页面 split、ContractNLI 官方 split、FinanceBench company fold 和 RAMDocs 协议 | `data/selector/split_manifest.json` | DONE |
| M2 标签 | NIAH、RAMDocs、FinanceBench、ContractNLI 确定性标签映射；80 组模型辅助审计 | `docs/data/ml_selector_validation/pilot/label_audit_summary.json` | PARTIAL：人工双标待完成 |
| M3 候选 | NIAH、RAMDocs、ContractNLI、FinanceBench 固定候选池和 hash 审计 | 各数据集 `top20_audit.json` | DONE |
| M3 特征 | NIAH、RAMDocs、FinanceBench QA 投票与 Granite reliability 特征 | 各数据集 feature cache audit | DONE |
| M4 基线 | Q2D、fixed 0.6/0.4、alpha*、source dedup、support-only、Oracle@20 | per-query metrics | DONE |
| M5–M6 模型 | NIAH train/dev 上训练 Logistic、Core、Full，3 seeds；shuffled-label 负控 | frozen LightGBM models | DONE |
| M7 盲测 | sealed NIAH 300、RAMDocs 500、FinanceBench 150、ContractNLI test 诊断 | `docs/data/ml_selector_validation/` | DONE |
| M7 消融 | relevance-only、Full、Full-no-relevance、support-only、shuffled labels、feature importance | `feature_importance.json` + figures | DONE |
| M8 RAG | Gate 1 未通过后停止完整答案生成实验 | 无 | STOPPED BY PROTOCOL |
| M9 接入 | 未注册生产 retriever arm，避免把未通过方法并入团队主线 | 无 | DEFERRED |
| 报告 | 三张结果图、会议版 HTML、原始 JSON 快照 | `docs/ml_selector_validation_results.html` | DONE |

## Gate 记录

| Gate | 条件 | 状态 | 结论 |
|---|---|---|---|
| Gate 0 | 无 split / feature leakage；人工 weighted kappa ≥ 0.70 | **未完成** | 自动泄漏检查通过；两名人工标注者仍待完成。双 Granite 与 canonical label 的 kappa 为 0.005–0.080 |
| Gate 1 | sealed NIAH：NDCG +0.02、Harmful -0.02、Recall 非劣 | **FAIL** | NDCG +0.0206、Recall +0.0368；Harmful 增加 0.0017 |
| Gate 2 | 至少两个真实/外部域正向，且 Recall 非劣 | **FAIL** | RAMDocs 仅 NDCG 改善；Finance 改善不显著且 harmful 变差；ContractNLI 迁移失败 |
| Gate 3 | 至少两个答案主指标改善 | **NOT RUN** | 按 Gate 1 停止规则不运行完整 RAG |

## 最终判断

实验支持两个结论：

1. 相关性和证据可靠性会在误导、冲突、版本错误环境中脱钩。
2. 当前 ML selector 能提高排序质量和正确证据保留率，尚未稳定降低误导证据进入 context 的比例。

下一轮先完成人工标签审计，加入实体、时间、条件和冲突特征，并把 harmful penalty 纳入训练目标。Granite 微调、完整 RAG 和系统接入保持 deferred。

## 主要产出

- [会议版结果报告](ml_selector_validation_results.html)
- [完整实验计划](ml-selector-experiment-plan.md)
- [NIAH 原始结果](data/ml_selector_validation/pilot/results.json)
- [RAMDocs 原始结果](data/ml_selector_validation/ramdocs/results.json)
- [FinanceBench 原始结果](data/ml_selector_validation/financebench/results.json)
- [ContractNLI 原始结果](data/ml_selector_validation/contractnli/results.json)
