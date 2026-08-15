# B110 瓶颈路由决定

**日期：** 2026-08-15
**状态：** `COMPLETE`
**决策代码：** `1374199fb0d42eb520c00e07ee8101ffc134e60f`
**机器产物：** [`artifacts/B110/B110_DECISION.json`](artifacts/B110/B110_DECISION.json)

## 1. 决定

当前主路线明确路由到：

```text
Generator evidence utilization + context robustness
```

这不是说 Retriever 和 Selector 已经没有问题，而是 B100 显示当前最先限制端到端答案收益的环节是 Generator 如何读取、选择和稳定使用已经可见的支持证据。

## 2. 三模块责任

### Retriever

全量 NIAH decision-dev 739 题中，727 题的 TopK10 至少包含一个官方相关文档，12 题不包含。12/739 = 1.62% 记录为独立 Retriever visibility bottleneck；它不是 218 题 B100 support-visible 诊断样本的主失败来源。

### Generator

- O support-only 仍失败 68/218，其中 21 个空输出、47 个非空不匹配；
- 218/218 的 O 证据文本都直接包含规范化 reference 字符串；
- O 相对 K 提高 +8.72pp，CI [+3.10,+14.69]pp；
- O 正确的 150 题中，23 题在加入 benign 或 harmful 噪声后至少一次从对变错；
- O/K、O/O+B、O/O+H 和 K/O-P 都出现大量答案文本变化。

因此，Generator 的基础证据利用和上下文鲁棒性同时成立为主要诊断责任。

### Selector

Legacy Selector 在 109 个 changed 题上 S-K=-2.75pp，CI 跨 0。7 个 K 对/S 错题中删除的 8 条证据全部是 harmful，未删除 support。

因此不能把主要问题写成“Selector 不会找 harmful”或“这 7 题都是 Selector 误删正确证据”。Legacy Selector 继续保留为 evidence-risk baseline，但它当前的风险删除目标没有转化为答案 utility。

## 3. Splitter 分支

O 的 68 个失败中，12 个是 `splitter_no_claims`，只有 2/218 使用 degraded fallback。零 claims 是真实问题，但只占 O 失败的 17.65%，不是多数瓶颈。

因此 B110 不启动 G210 作为主分支，也不把 splitter fallback 与第一轮 Generator 训练同时改动。G210 保持未激活；后续若新的 trace 证明它成为独立主瓶颈，再单独开启。

## 4. 路线门状态

- `G200/G220` 实际 draft 路径的数据构造与 Generator 训练：解除 B110 阻塞；
- `G210` splitter fallback：本轮不激活；
- `S300/S310/S320` Generator-aware Utility Selector：继续阻塞，直到新 Generator 通过 G230；
- Retriever 12 个 visibility failure：单独记录，不与 Generator 训练混为一个改动。

## 5. 解释边界

B110 只决定下一实验责任应放在哪个模块，不证明计划中的新 Generator 训练一定有效。新方法是否保留，仍必须经过 G230 的完整开发集、coverage、citation、stress slice 和多 seed 门。
