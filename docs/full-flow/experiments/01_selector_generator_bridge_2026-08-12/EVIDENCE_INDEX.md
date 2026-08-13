# 证据与结果索引

F000–F004 已完成。F004 的普通关键事实笔记局部正向但整体未超过 TopK；G2 跨阶段风险信号无额外价值。F006 的 Mixed-LoRA 在 109 题 development 比较中同时超过 Base、Clean 和 TopK，开发门通过；F005 已获准进行数据隔离与运行前检查。

## 已有上游证据

- Selector 最终报告：[`docs/selector/experiments/08_r005ab_repair_current_2026-08-12/L003_FINAL_REPORT.md`](../../../selector/experiments/08_r005ab_repair_current_2026-08-12/L003_FINAL_REPORT.md)
- Generator 冻结结果：[`docs/generator/frozen-results.md`](../../../generator/frozen-results.md)
- Generator 设计复核：[`docs/generator/design-review.md`](../../../generator/design-review.md)
- Generator 主实现：`src/evidence_rag/generator/verify_annotate.py`
- 当前三模块 pipeline：`src/evidence_rag/pipeline/service.py`
- Selector decision trace 合同：`src/evidence_rag/selector/models.py`

## 本路线产物

| Run | 预期产物 | 当前状态 |
|---|---|---|
| F000 | [`F000_IMPLEMENTATION_REPORT.md`](F000_IMPLEMENTATION_REPORT.md)；[`artifacts/F000_INPUT_MANIFEST.json`](artifacts/F000_INPUT_MANIFEST.json) | PASS |
| F001 | [`F001_F002_RESULTS.md`](F001_F002_RESULTS.md)；[`artifacts/F001/REPORT.md`](artifacts/F001/REPORT.md)；[`artifacts/F001/CITATION_REPORT.md`](artifacts/F001/CITATION_REPORT.md)；逐题输出 | COMPLETE-NO-WIN |
| F002 | [`artifacts/F001/F002_REPORT.md`](artifacts/F001/F002_REPORT.md)；109题 [`F002_diagnosis_rows.jsonl`](artifacts/F001/F002_diagnosis_rows.jsonl) | COMPLETE |
| F003A | [`F003_IMPLEMENTATION_REPORT.md`](F003_IMPLEMENTATION_REPORT.md)；`SelectionGuidance` 接口、allowlist 与泄漏测试 | COMPLETE，commit `6d1b831` |
| F003B | [`F003_IMPLEMENTATION_REPORT.md`](F003_IMPLEMENTATION_REPORT.md)；问题需求/关键事实笔记实现 | COMPLETE，commit `9cb742f` |
| F003C | 条件 `EvidenceReadiness` 实现与结果 | NOT AUTHORISED |
| F004 | [`F004_RESULTS.md`](F004_RESULTS.md)；[`artifacts/F004/`](artifacts/F004/) 的 G0/G1/G2 输出、独立引用评分、失败转移诊断和 TopK 对照 | COMPLETE-NO-GATE |
| F005 | 独立最终结果 | AUTHORISED / PREFLIGHT |
| F006 | [`F006_TRAINING_REPORT.md`](F006_TRAINING_REPORT.md)；[`F006_RESULTS.md`](F006_RESULTS.md)；[`artifacts/F006/`](artifacts/F006/) 的数据、训练、输出与独立引用评分 | COMPLETE / DEV-PASS |
