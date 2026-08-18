# Generator grounding repair（Generator 证据使用与引用修复）

**日期：** 2026-08-18
**状态：** `PLANNED / NO NEW RUN AUTHORIZED`
**详细计划：** [PLAN.md](PLAN.md)
**执行跟踪：** [TRACKER.md](TRACKER.md)

## 当前处在哪里

上一条 full-flow 路线已经完成 G230：

- 新 draft LoRA 把 draft empty 降到 0%，把 final empty 从 G0 的 12.04% 降到约 3.79%–6.09%；
- GM 的 answer match 点估计比 G0 高约 3.52–6.36pp；
- 但 mixed-context 稳定性没有在全部预注册 stress 条件上成立；
- GC/GM 的独立 MiniCheck citation precision/recall 相对 G0 下降约 6–8pp；
- 因此 G230 正式结论是 `NO CANDIDATE`，不能把任何新 LoRA 冻结为 `G*`。

这说明现在不是重新做 Retriever 或重新证明 Selector 会删除 harmful evidence。当前最直接的问题是：

> Generator 已经更愿意回答，也有更高的答案点估计，但还不能稳定地把每个事实和真正支持它的证据对应起来。

## 本路线要完成什么

本路线只做四件事：

1. 从已有 G230 逐题 trace 判断 citation 损失发生在 draft、claim splitter、TRUE routing 还是最终组装；
2. 使用 NIAH train、2Wiki train 和训练集内部构造的 unsupported contexts 建立更接近真实目标的训练数据；
3. 受控比较“从 G230 继续修复”与“从 frozen Granite base 重新联合训练”，只选择一个配方进入三 seed 正式确认；
4. 在 NIAH、2Wiki 和已揭示 citation regression 数据上同时通过 answer、coverage、context robustness 和 citation 门，才冻结新的 Generator `G*`。

## 本路线明确不做什么

- 不修改当前 Retriever 方法或默认配置；
- 不重新训练或重新调 F005 Selector threshold/cap；
- 不覆盖 F005、G230 或其他失败/完成结果；
- 不使用已经退休的 sealed600；
- 不读取或反复运行 HotpotQA、MuSiQue-Full、RGB system held-out；
- 不同时训练 draft、splitter、TRUE 和 Selector；
- 不保证新方法一定取得正结果，也不在开发结果失败后继续换指标或挑 seed。

如果本路线产生通过资格门的 `G*`，才解锁第 02 路线中尚未执行的 Generator-aware Selector 阶段。完整系统冻结后，HotpotQA、MuSiQue-Full 和 RGB 仍只运行一次。
