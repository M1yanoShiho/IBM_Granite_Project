# B100 受控上下文矩阵结果

**日期：** 2026-08-15
**状态：** `COMPLETE / DIAGNOSTIC INTEGRITY PASS`
**生成代码：** `f3bac13d5481bcfcd13b3465bc6dadc596be52c0`
**评分代码：** `c5cc3ca69bfc46956a1d800fa385412d0b2e3972`
**机器产物：** [`artifacts/B100/`](artifacts/B100/)

## 1. 执行与样本完整性

- 样本共 218 题：109 个 Selector changed + 109 个一对一 matched unchanged；
- 问题类型 109/109 精确匹配，reference shape 106/109、support bucket 104/109 精确匹配；
- 两组的 harmful/benign visibility、chain eligibility 完全对齐；212/218 题可进入严格等量 O+B/O+H 对照；
- 六臂各运行 218 次，共 1308 次 Generator 调用；错误均为 0，trace 缺失均为 0；
- 六臂在同一进程共享 Granite 4.1-3B 与 TRUE；生成时未加载 gold；
- K/S 在 218/218 题上与 A002 的答案、完整 `GenerationResult` 和 trace 全部一致；
- `generations.jsonl` SHA256 为 `cae76fd...08f20`。

因此，B100 的 K/S 不是新的漂移基线，后续差异来自预注册的上下文构造。

## 2. 六臂主结果

| 臂 | Answer match | Coverage | Gold-doc citation precision | Visible-support citation recall |
|---|---:|---:|---:|---:|
| K TopK10 | 131/218 = 60.09% | 196/218 = 89.91% | 82.57% | 31.76% |
| S Legacy Selector | 128/218 = 58.72% | 192/218 = 88.07% | 87.72% | 33.41% |
| O support-only | 150/218 = 68.81% | 197/218 = 90.37% | 100.00% | 40.86% |
| O+B matched benign | 147/218 = 67.43% | 196/218 = 89.91% | 97.43% | 39.85% |
| O+H matched harmful | 142/218 = 65.14% | 193/218 = 88.53% | 98.20% | 38.45% |
| O-P same TopK, support last | 138/218 = 63.30% | 195/218 = 89.45% | 79.60% | 32.96% |

Citation 指标是生成后 gold-document overlap 诊断，不是 MiniCheck 句级裁判；无 citation 的题不进入 precision 均值。

## 3. 配对比较

| 比较 | n | Delta | 95% CI | 错→对 / 对→错 | Answer 文本相同 |
|---|---:|---:|---:|---:|---:|
| S-K | 218 | -1.38pp | [-4.31,+1.45]pp | 4 / 7 | 176/218 |
| O-K | 218 | +8.72pp | [+3.10,+14.69]pp | 30 / 11 | 91/218 |
| O+B-O | 218 | -1.38pp | [-5.68,+2.83]pp | 11 / 14 | 137/218 |
| O+H-O | 218 | -3.67pp | [-8.37,+0.88]pp | 9 / 17 | 126/218 |
| O+H-(O+B) | 212 | -2.36pp | [-7.41,+2.53]pp | 11 / 16 | 112/212 |
| O-P-K | 218 | +3.21pp | [-2.15,+8.70]pp | 21 / 14 | 98/218 |

O-K 是本阶段唯一 CI 不跨 0 的答案差异：去掉全部干扰后，当前 Granite 的 answer match 提高 8.72pp。Coverage 只提高 0.46pp，说明主要变化不是简单减少空答，而是非空答案内容更容易与 reference 对齐。

O+B、O+H 和 O+H-(O+B) 的点估计方向均显示加入噪声后下降、harmful 比 benign 更低，但三个 CI 都跨 0，不能宣称已证明 harmful 的净影响显著大于 benign。

O-P 没有降低准确率，点估计反而为正且 CI 跨 0。因此不能写“把支持证据放到末尾导致准确率下降”。但 K/O-P 只有 98/218 个答案文本相同，并发生 35 个正确性翻转，说明顺序和重新编号确实会明显改变输出，只是净方向未确定。

## 4. support-only 仍失败在哪里

218 个 O 上下文都能直接找到规范化后的 reference 字符串，但仍有 68/218 = 31.19% 未通过当前 answer-match：

- 47 题生成非空但答案未匹配；
- 12 题 draft 非空但 splitter 输出零 claims；
- 9 题全部 claims 被 faithfulness 路径跳过；
- O 中没有 `empty_draft`；splitter 216 structured、2 degraded fallback。

这 68 题不能全部解释为同一种错误。部分非空答案可能是别名、简称或粒度不符合 benchmark reference；逐题判断应以 `B100_cases.jsonl` 为准。但“reference 字符串在证据中直接可见、Generator 仍有 21 个空输出和 47 个非空不匹配”足以说明 Generator evidence utilization 仍是主要问题之一。

## 5. 上下文鲁棒性

O 正确的 150 题中：

- 加入等量 benign 后有 14 题从对变错；
- 加入等量 harmful 后有 17 题从对变错；
- 两类噪声的并集为 23 题；
- O/O+B 答案文本仅 137/218 相同，O/O+H 仅 126/218 相同。

因此，Generator 对上下文组成敏感成立；“harmful 比 benign 的净准确率损害更大”只得到方向性证据，尚未得到排除 0 的统计确认。

## 6. Selector 现象的重新定位

在 109 个 changed 题上：K=58.72%，S=55.96%，S-K=-2.75pp，95% CI [-8.62,+2.97]pp，4 个错→对、7 个对→错。

逐题审计发现，这 7 个 K 对/S 错题中 Selector 共删除 8 条证据，8 条全部是 harmful，没有删除 support。这说明这些回退不能简单归因为“Selector 删错了正确证据”；更符合实际的描述是：Selector 做了风险上合理的删除，但当前 Generator 对清理后的剩余证据、顺序和编号反应不稳。

## 7. B100 判定

- 运行、gold 隔离、trace 和基线复现：PASS；
- support-only 相对 TopK 的诊断提升：观察到，CI 不跨 0；
- harmful 相对 benign 的独立净损害：方向为负，但证据不足；
- support-last 的净损害：未观察到；
- Legacy Selector 的端到端答案提升：未观察到；
- B100 是已揭示 dev 上的机制诊断，不是新方法效果，也不使用 sealed600 或 system held-out。
