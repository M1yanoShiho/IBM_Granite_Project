# Generator 修复与 Selector 分阶段协同

**日期：** 2026-08-18
**状态：** `REVISED PLAN / WAITING FOR USER CONFIRMATION / NO RUN AUTHORIZED`
**详细计划：** [PLAN.md](PLAN.md)
**执行跟踪：** [TRACKER.md](TRACKER.md)
**原始方案快照：** [snapshots/PLAN_v1_generator_only_2026-08-18.md](snapshots/PLAN_v1_generator_only_2026-08-18.md)

## 当前处在哪里

三个模块都有可运行代码，但新的最终方法还没有确定：

- Retriever 保持当前 frozen Hybrid RRF/default；
- Legacy Lean Selector 使用 NIAH 和 2Wiki 训练，但 harmful 监督只来自 NIAH synthetic counterfactual；
- F005 的 74/82、90.24% deletion precision 只证明 NIAH 风险代理上的高精度删除；
- Legacy Selector 在 L003 的 1,000 个 2Wiki 问题上删除 0 条，不能据此声称跨数据 harmful filtering；
- G230 的 GC/GM 提高回答率并减少空答案，但 citation precision/recall 下降约 6–8pp；
- 因此最终 Selector、最终 Generator 和完整新系统目前都不存在。

## 修订后的核心方法

Selector 与 Generator 有联系，但不同时自由修改：

```text
1. 修复 Generator
2. 冻结一个教师 Generator GQ
3. 用 GQ 测量删除每条证据对答案和引用的影响
4. 训练 Utility 候选 SU，并冻结职责合格 Selector SQ
5. 冻结 Retriever + SQ + GQ 做完整系统开发验证
6. 只有 SystemF 冻结后，才一次性运行最终三个数据集
```

如果新 Generator 没有通过模块职责门，不会因此完全阻断 Selector：可靠基线 G0 可以作为冻结教师，但不能声称 Generator 修复成功。

主目标是 `SQ=SU`。只有 SU 失败、而旧 Legacy Selector 在同一 GQ 下确有正的端到端作用时，才允许 `SQ=SL` 作为系统 fallback；这时不能声称 Utility Selector 方法成功。

## 三种不同的“通过”

1. **模块职责通过：** 组件足够稳定，可以进入下一阶段；不要求每个 slice 的 CI 都显著。
2. **强统计结论通过：** 预注册主要指标的 CI 支持“优于基线”。
3. **完整系统通过：** Utility Selector 对同一个 GQ 有净作用，完整系统主要联合结果正向且没有实际不可接受的答案、引用或安全退化。

CI 跨 0 不再自动淘汰职责合格组件，但也不能写成统计显著提升。

## 数据角色

- NIAH train：抗 harmful/benign、位置、unsupported、utility；
- 2Wiki train：多证据链和冻结 GQ 下的 evidence utility；
- NIAH/2Wiki dev：锁定资格，不作最终结论；
- ASQA/QAMPARI：已揭示 citation catastrophe guard；
- sealed600：退休，只读历史；
- HotpotQA、MuSiQue-Full、RGB：SystemF 冻结后一次性最终测试，不参与方法选择。

## 当前不会执行

- 不启动 GPU 训练；
- 不生成 utility labels；
- 不修改 Retriever、Legacy Selector 或历史结果；
- 不使用 sealed600；
- 不读取/评分 system held-out；
- 不在用户确认前冻结 G000 protocol。
