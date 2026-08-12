# 04 — Graph 2.0 relation layer

**时间：** 2026-07-30 至 2026-08-11

**状态：** `PARTIAL / NO SELECTOR-LEVEL PASS / RETIRED`

## 零基础解释

Graph 1.0 把“两个答案字符串不同”近似当作冲突。Graph 2.0 改为让 NLI 模型输出 `SUPPORTS / REFUTES / UNKNOWN`，希望同时减少误判冲突和漏判冲突。

零训练关系模型的 R012 Gate 没有通过。后来 seed13 的训练路径完成，训练指标也满足当时三项要求，但真实 Top20 probe 暴露出严重外推问题：其他候选几乎全部被强制判为 SUPPORTS/REFUTES，`UNKNOWN` 约为 0.005%，平均每题约产生 13.34 个 false conflicts。

这说明“训练收敛”不等于“能作为安全 Selector”。该路线没有完成三 seed、没有完成 Selector 级 sealed C1，也没有进入生产。

## 状态边界

这里不能写成一次完整正式测试的 `FAIL`：它在最终 Selector/sealed 阶段之前就停止了。准确表述是 `PARTIAL / NO SELECTOR-LEVEL PASS / RETIRED`。

## 文件入口

- [计划导航](PLAN.md)
- [状态记录](TRACKER.md)
- [证据索引](EVIDENCE_INDEX.md)
- [来源清单](SOURCE_MANIFEST.json)
