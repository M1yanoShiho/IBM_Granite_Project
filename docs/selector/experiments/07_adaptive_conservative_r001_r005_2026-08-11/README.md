# 07 — Adaptive Conservative Selector R001–R005

**时间：** 2026-08-11 至 2026-08-12

**状态：** `R001–R004 PASS / R005 TRAINING-GATE FAIL / R006–R015 CUT`

## 零基础解释

这条路线把“拿不准就保留”写进 0–cap 删除策略，并计划使用 protect/harm 双头 scorer、component-aware 风险控制和 sealed/heldout 验证。

实际执行到：

- R001 候选池恢复与冻结：PASS；
- R002 component/样本协议：PASS；
- R003 TopK 和等量删除基线：PASS；
- R004 标签与资源 preflight：PASS；
- R005 双头 scorer 训练资格门：FAIL；
- R006–R015：按预注册停止规则 CUT，没有运行。

这不是“前面都没用”。R001–R004 的候选池、组件隔离、TopK 基线、标签与资源检查继续构成后续设计证据；失败的是 R005 的具体 scorer/训练资格。

## 文件入口

- [计划导航](PLAN.md)
- [状态记录](TRACKER.md)
- [证据索引](EVIDENCE_INDEX.md)
- [来源清单](SOURCE_MANIFEST.json)
