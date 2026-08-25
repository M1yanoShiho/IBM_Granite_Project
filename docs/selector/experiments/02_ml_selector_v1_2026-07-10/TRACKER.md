# 状态记录

当时的逐项 tracker 原文保存在 [snapshots/ML_SELECTOR_V1_EXPERIMENT_TRACKER.md](snapshots/ML_SELECTOR_V1_EXPERIMENT_TRACKER.md)。下面只做最终状态摘要：

| 项目 | 最终状态 |
|---|---|
| V1 数据、候选、标签与模型管线 | COMPLETE |
| Gate 0 人工双标 | INCOMPLETE |
| NIAH/RAMDocs/FinanceBench/ContractNLI 诊断 | COMPLETE |
| Gate 1 / Gate 2 | FAIL |
| 完整 RAG 答案实验 | NOT RUN BY STOP RULE |
| V2 计划 | SUPERSEDED / NOT RUN |

FinanceBench 状态固定为 `EXPOSED_DIAGNOSTIC_ONLY`，不能再作为未见确认集。
