# 状态对照

> 这是 2026-08-12 根据原计划与 Git 结果记录生成的归档对照，不是当时存在的独立 tracker。

| 阶段 | 状态 | 结论 |
|---|---|---|
| Corroboration score 与 reranker 实现 | COMPLETE | 训练-free 信号链可运行 |
| gated 诊断 | COMPLETE | 用于检查互证信号的条件与失败面 |
| nested-CV 复验 | COMPLETE | honest out-of-fold `n=300` |
| needle-found@10 | LIMITED POSITIVE | 约 `+3.7pp`，p 值约 `.026/.036` |
| MRR | NO MATERIAL CHANGE | p 值约 `.81/.94`，没有稳定提升 |
| harmful evidence 安全删除 | NOT ESTABLISHED | 本路线没有形成该主张所需的保护/删除联合门 |

最终状态：`COMPLETED / LIMITED POSITIVE SIGNAL / NOT A PRODUCTION SELECTOR`。
