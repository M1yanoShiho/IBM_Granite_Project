# L001 最小实现报告

**日期：** 2026-08-12
**状态：** `PASS / READY FOR L002 / FINAL UNOPENED`

## 零基础结论

这一阶段还没有声称 Selector 已经优于 TopK10。它完成的是实验工具本身：现在程序可以在完全相同的 TopK10 中，先判断每条证据“多需要保护”和“多可能有害”，再只删除同时满足“高 harmful、低 protect”的证据；不确定时保留。

开发阶段会固定比较 4 个保守阈值和“最多删 1 条/2 条”两个上限。每题实际可以删 0 条，不会为了满足上限而强行删除，也不会用 TopK11–20 补位。程序会同时计算：

- 有害证据减少了多少；
- 正确证据 recall 和完整证据链损失了多少；
- 是否优于“随机删同样数量”和“从排名末尾删同样数量”。

这使 L002 可以直接回答真正的问题：哪一个删除规则在开发集上找到了“删错得少、删坏得多”的平衡。

## 已完成

- 新增 NLI-aware 双头 scorer，恢复预训练 DeBERTa NLI encoder、pooler 和 classifier 信息；protect/harm 两个 head 独立学习。
- 实现 NLI-base 主训练和仅在基础版失败时启用的 NLI-pair loss。
- 复用已有 delete-only `RiskControlledSelector`，保持 TopK10、0–cap 删除和异常时整题保留。
- 补齐 train-fit 阈值、development 4×2 策略评估、count-matched controls、策略选择和冻结入口。
- development 与 final 使用同一 evaluation projection 语义，但 final 的 `decision-dev` 内容仍未打开。

## 验证证据

- 本地：Selector 相关回归测试全部通过；新增相关测试 28 项通过；ruff 和 mypy 通过。
- 服务器：真实 pinned DeBERTa NLI 模型加载、forward、单步反向训练和 checkpoint reload 通过。
- 服务器：L001 三个相关测试文件共 38 项全部通过。
- 边界：没有运行正式 NLI-base 训练，没有读取 `decision-dev` Selector 效果，没有改变生产默认 TopK10。

## 下一步

进入 L002：在服务器分别训练 seed13 和 seed42，然后只在普通 development（`crc-calibration`）上选择唯一的阈值与删除上限。如果没有任何规则同时满足“有实际删除、有害证据下降、正确证据损失受控、优于等量删除对照”，就停止并保持 TopK10，不打开最终盲测。
