# 证据与结果索引

F000–F003B 已完成。F001 执行完整但方法未胜出；F002 批准的 F003A/F003B 已实施，F004 正在运行。

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
| F004 | G0/G1/G2 消融、独立引用评分、失败转移诊断 | RUNNING；runner commit `6d8f982` |
| F005 | 独立最终结果 | NOT CREATED |
| F006 | 条件训练配置、checkpoint 与结果 | NOT AUTHORISED |
