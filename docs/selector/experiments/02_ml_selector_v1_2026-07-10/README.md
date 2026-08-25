# 02 — ML Evidence Selector V1

**时间：** 2026-07-10 至 2026-07-13

**状态：** `COMPLETE / GATE FAIL / STOPPED`

## 零基础解释

这条路线使用 LightGBM/Logistic 等学习器，把相关性、互证和其他特征组合成一个重新排序模型。它在部分排序指标和正确证据召回上有改善，但没有稳定降低 harmful evidence；RAMDocs、FinanceBench 与 ContractNLI 也没有形成可靠的跨域收益。

因此 Gate 1、Gate 2 失败，完整 RAG 答案实验按规则没有运行。FinanceBench 已被用作诊断，不能再冒充新的盲测集。

## 文档版本关系

- `snapshots/` 保存 V1 最终计划、tracker 和合并后的实验记录。
- `snapshots/superseded/` 保存当时提出但未开始运行的 Selector V2 计划；它不是一次已经失败的实验。
- 后来的 Graph/Gated 路线吸收了 V1 的失败教训，但属于新的实验路线。

## 文件入口

- [计划导航](PLAN.md)
- [状态记录](TRACKER.md)
- [证据索引](EVIDENCE_INDEX.md)
- [来源清单](SOURCE_MANIFEST.json)
