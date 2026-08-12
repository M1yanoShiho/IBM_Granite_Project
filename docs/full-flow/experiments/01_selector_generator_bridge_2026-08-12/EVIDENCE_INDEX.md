# 证据与结果索引

F000 已完成，F001 尚未运行。

## 已有上游证据

- Selector 最终报告：[`docs/selector/experiments/08_r005ab_repair_current_2026-08-12/L003_FINAL_REPORT.md`](../../../selector/experiments/08_r005ab_repair_current_2026-08-12/L003_FINAL_REPORT.md)
- Generator 冻结结果：[`docs/generator/frozen-results.md`](../../../generator/frozen-results.md)
- Generator 设计复核：[`docs/generator/design-review.md`](../../../generator/design-review.md)
- Generator 主实现：`src/evidence_rag/generator/verify_annotate.py`
- 当前三模块 pipeline：`src/evidence_rag/pipeline/service.py`
- Selector decision trace 合同：`src/evidence_rag/selector/models.py`

## 本路线未来产物

| Run | 预期产物 | 当前状态 |
|---|---|---|
| F000 | [`F000_IMPLEMENTATION_REPORT.md`](F000_IMPLEMENTATION_REPORT.md)；[`artifacts/F000_INPUT_MANIFEST.json`](artifacts/F000_INPUT_MANIFEST.json) | PASS |
| F001 | 四臂逐题输出与联合报告 | NOT CREATED |
| F002 | 109 题错误分类表 | NOT CREATED |
| F003A | `SelectionGuidance` 接口、allowlist 与泄漏测试 | NOT CREATED |
| F003B | 问题需求/关键事实笔记实现与结果 | NOT CREATED |
| F003C | 条件 `EvidenceReadiness` 实现与结果 | NOT AUTHORISED |
| F004 | G0/G1/G2 消融报告 | NOT CREATED |
| F005 | 独立最终结果 | NOT CREATED |
| F006 | 条件训练配置、checkpoint 与结果 | NOT AUTHORISED |
