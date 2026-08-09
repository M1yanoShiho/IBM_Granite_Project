# Selector 旧方法与清理记录

## 必须保留的两条历史结论

### 旧 corroboration / gated / Graph 路线

它能够降低一部分 harmful exposure，但必要证据损失明显，后续抽取与关系层路线越来越复杂，仍未形成可采用的最终 Selector。

### Reliability-MIS

正式冻结结果：

| 指标 | TopK | Reliability-MIS |
|---|---:|---:|
| Harmful-in-context (%) ↓ | 91.77 | 31.35 |
| Required-evidence recall (%) ↑ | 90.97 | 53.80 |
| Evidence precision (%) ↑ | 33.38 | 23.72 |
| 2Wiki supporting recall (%) ↑ | 74.49 | 63.30 |

结论：MIS 会发现冲突，但不能稳定判断冲突双方谁正确，因而严重误删必要证据。该方法不得进入系统。

完整结果保存在 `results/selector-reliability-mis/`；Git 历史与服务器提交 `39e57393` 保留实现和运行工具，因此没有必要继续把失败实现注册在生产代码中。

## 已立即删除的文档

下列文件都是旧路线的重复计划、阶段性复盘或本地论文副本；其有效结论已压缩到本文件和正式结果报告：

- `DATASET_PROPOSAL.md`
- 旧版 `EXPERIMENT_TRACKER.md`（必要执行顺序已合并进新实验计划）
- `RELATED_WORK.md`
- `RELIABILITY_MIS_IMPLEMENTATION_PLAN.md`
- `RELIABILITY_MIS_EXPERIMENT_PLAN.md`
- `TRAINING_PLAN.md`
- `gate0b-review-prep.md`
- `selector.md`
- `research_proposal_decision_worthy_evidence_rag.html`
- `assets/rag_evidence_retrieval_flow.svg`
- `references/ReliabilityRAG_NeurIPS_2025.pdf`

## 历史协议的保留方式

旧 `M0_PROTOCOL_FREEZE.md` 不是当前计划，因此不再放在 Selector 文档目录。
冻结原文仍保存在 Git 历史和服务器最终实验提交 `39e57393` 中；其 SHA-256 为
`0a2e74d30d1a509f76ab7750cd7a2282772d7d41ab1ee177698f0d2fdd554b59`，需要复核 sealed600 时可按提交恢复。

## 服务器最终提交同步后的代码 cutover

本地 `df8c28c` 缺少服务器 `39e57393` 的最终实验 harness，因此本轮不在旧 checkout 上拆一半依赖。同步后一次性处理：

### 删除旧生产实现

- MIS：`selector/reliability_mis.py`、`selector/deberta_contradiction.py`；
- 旧 corroboration/Graph：`selector/corroboration.py`、`selector/gated.py`、`selector/coverage.py`、`selector/clusters.py`、`selector/extraction.py`；
- 仅服务旧路线的 `relations/`、CLI、evaluation、SLURM、experiment config 与对应测试。

### 保留

- `selector/top_k.py`：正式基线和失败回退；
- `selector/answer_norm.py`：反事实数据构建仍在使用；
- 通用 CandidateSet、SelectionResult、harm/recall 评估和 frozen-pool 审计工具；
- `results/selector-reliability-mis/` 中的小型最终报告、表格、manifest 与 checksums。

### 删除验收

- `composition.py` 不再注册旧方法名；
- 全仓搜索不存在旧 selector 的运行入口；
- TopK、数据构建、候选池审计和新 Selector 测试全部通过；
- 删除只发生在一个独立、可回滚的 Git commit 中。
