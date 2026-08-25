# 03 — Gated Corroboration / Graph 1.0

**时间：** 2026-07-20 至 2026-07-29

**状态：** `COMPLETE / SAFETY GATE FAIL / RETIRED`

## 零基础解释

这条路线第一次真正加入“删除门”：如果某条证据的答案与池内多数互证答案冲突，而且它本身支持票很少，就把它删除。

它确实让 harmful-in-context 下降约 11.2 个百分点，但 required-evidence recall 同时下降约 4.8 个百分点，超过只允许下降 1 个百分点的保护线。lenient 字符串等价修复回收约 1.2 个百分点，仍无法闭合安全门。

所以它不是“完全没用”，而是删除能力真实存在、保护正确证据的能力不够。S1–S6 的诊断进一步发现，错误主要来自 false conflict、missed conflict 和无法可靠弃权。

## 与 Graph 2.0 的关系

Graph 2.0 试图把这里的字符串冲突规则换成 NLI 语义关系，是新的实施与验证路线，因此单列为 [第 04 文件夹](../04_graph2_relation_layer_2026-07-30/README.md)。

## 文件入口

- [计划导航](PLAN.md)
- [状态记录](TRACKER.md)
- [证据索引](EVIDENCE_INDEX.md)
- [来源清单](SOURCE_MANIFEST.json)
