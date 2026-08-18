# Generator 修复与 Selector 分阶段协同

**日期：** 2026-08-18
**状态：** `G200R2 COMPLETE / PRE-AUDIT PASS / G210R2 READY / NO TRAINING STARTED`
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
- G110 独立审计通过，主要断点判为 TRUE routing/attachment；G130 已完成一次共享 deterministic runtime 修复；
- G200 已完成数据预物化：NIAH train 515、新 NIAH model-val 307、2Wiki train 1,075、2Wiki model-val 136、unsupported update ratio 10.17%，但 TRUE/minimal-support/manual/length 审计仍未完成；
- G210 structural audit 通过，但 TRUE 过滤后 2Wiki model-val 只剩 76 个 answerable groups，低于最低门 100；
- G215 已把这个失败解释为“当前数据不能冻结，但路线有积极诊断信号，可回到 G200R/G210R 做受控数据修订”，不是 Generator/Selector 路线失败；
- G200R 已把 2Wiki target construction 改为 support-sentence-aligned，并重新物化为 2,665 个 train cases、443 个 validation cases，pre-audit gate 全部通过；
- G210R-v1 structural audit 发现 25 个 2Wiki answerable case 没保留 answer alias，因此 TRUE 没有启动，必须回到 G200R2 过滤这些 case；
- G200R2 已过滤 answer alias 不保留的 2Wiki support-sentence targets，剩余 2Wiki train/model-val 为 1,053/133，pre-audit gate 全部通过；
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
- 不在原 G210 failure 数据上启动 G300 或任何 Generator 训练。

## 下一步

下一步不是训练，而是执行 G210R2 目标审计：

```text
G210R2 revised target audit
-> 只有 G210R2 通过后才进入 G300
```

这次修订只允许处理 2Wiki target construction/audit mismatch；不得降低 TRUE 阈值、改最终测试集边界、读取 held-out，或按已看到的结果临时改门凑通过。
