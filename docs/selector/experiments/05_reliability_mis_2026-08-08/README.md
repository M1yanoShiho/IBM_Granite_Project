# 05 — ReliabilityRAG-inspired MIS

**时间：** 2026-08-08 至 2026-08-09

**状态：** `COMPLETE / FAIL / RETIRED`

## 零基础解释

这条路线先让每条 passage 独立回答问题，再用 DeBERTa 判断答案之间是否矛盾，最后用最大无冲突集合保留相互不矛盾的一组证据。

它把 harmful evidence 保留率大幅降低，但 required-evidence recall 同时下降 37.17 个百分点；2Wiki supporting recall 下降 11.19 个百分点。它能发现矛盾，却不能稳定判断冲突双方谁正确，因此不能替换 TopK。

## 一个必须如实保留的文档缺口

当时留下的是“代码实施计划”，文件本身明确说它不是完整实验方案。正式运行约束主要保存在冻结配置、execution notes 和最终报告中。归档不会事后伪造一份当时不存在的预注册实验计划。

## 文件入口

- [计划导航](PLAN.md)
- [状态记录](TRACKER.md)
- [证据索引](EVIDENCE_INDEX.md)
- [来源清单](SOURCE_MANIFEST.json)
