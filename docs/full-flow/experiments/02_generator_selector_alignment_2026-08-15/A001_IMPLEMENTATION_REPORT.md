# A001 Generator 全链路 Trace 实现报告

**日期：** 2026-08-15
**状态：** `COMPLETE / PASS`
**机器产物：** [`artifacts/A001/A001_TRACE_SCHEMA.json`](artifacts/A001/A001_TRACE_SCHEMA.json)

## 1. 实际调用链

当前 Verify-and-annotate 主路径保持不变：

```text
DraftGenerator / KeyFactDraftAnswerGenerator
  -> ClaimSplitter structured split
  -> ClaimSplitter faithfulness check
  -> CitationRoutedVerifier (TRUE entailment + entity observe/gate)
  -> VerifyAnnotateGenerator final assembly
  -> one GenerationResult
```

A001 没有增加 Generator 调用、重新生成、gold 输入或新的决策规则。Trace 默认关闭；开启后只记录现有中间状态。

## 2. 新增可观测字段

每题 `GeneratorTrace` 记录：

- Selector 输出的 evidence IDs；
- Granite 原始 draft 输出、标准化后的 draft 和空 draft 原因；
- splitter 原始输出、faithfulness 原始输出、结构化成功/降级 fallback/失败状态及 claims 数量；
- 每个 claim 的 span、faithfulness、degraded 标记；
- TRUE routing outcome、最终 citation、declared citation、entity gate observe outcome 和 conflict evidence；
- 每个 claim 最终是 verified、unverified、entity-conflict drop、unfaithful skip 还是 empty-sentence skip；
- 最终答案、citation IDs 和明确的 final-empty reason。

空答案现在可区分为：模型 draft 为空、非空 draft 得到零 claims、全部 claims 不 faithful、全部 claims 被移除/为空、以及 capped arm 没有 verified claim。Parser failure 单独记录为 `degraded_fallback`，不会再与模型主动空答混淆。

## 3. 行为与 gold 边界

- `trace_enabled=False` 是默认值，历史调用方行为不变；
- 同一固定输入下，trace on/off 的 LLM prompt 序列与最终 `GenerationResult` 完全相同；
- runtime `Query` 继续由 `extra="forbid"` 拒绝 `gold_answer` 等额外字段；
- trace schema 不包含 reference answer、gold provenance、utility label 或评分字段；
- 外部注入且没有实现内部 trace 的 legacy draft producer 会明确标为 `mode=external`、`splitter.status=unavailable`，不会伪造内部状态。

## 4. 验证

- 新增 A001 单测 5 项：行为等价、parser failure 与空 draft 区分、零 claims 空因、gold 字段拒绝、KeyFact/F006 路径接线；
- 全项目：`1646 passed, 20 skipped`；
- A001 相关源码 Ruff：PASS；
- A001 相关四个源码文件 mypy strict：PASS；
- JSON schema 由 `scripts/full_flow_a001_trace_schema.py` 从 Pydantic 模型生成。

本阶段没有启动 GPU、没有执行新数据集生成，也没有查看 system held-out 分数。
