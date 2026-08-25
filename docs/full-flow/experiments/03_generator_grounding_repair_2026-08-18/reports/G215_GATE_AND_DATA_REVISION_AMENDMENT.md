# G215 gate and data revision amendment

**日期：** 2026-08-18  
**状态：** `COMPLETE / POSITIVE-SIGNAL DATA REVISION AUTHORIZED / NO TRAINING STARTED`  
**输入状态：** `G210 FAIL / 2Wiki model-val 76 < 100`

## 用户问题

用户指出：如果实验已经有积极信号，就不应该因为没有达到最开始设定的一个数字而把工作整体中断；但也要避免没有意义地盲目推进。

本修订响应该要求，把两件事分开：

1. 原 G210 数据冻结门是否通过；
2. 当前路线是否有足够积极信号继续做受控修订。

## 对 100 门槛的解释

`2Wiki model-val >=100` 不是任意宣称的成功证明，也不是说 `76` 没有研究价值。它是预先冻结的数据稳定性保护，用于保证：

- 2Wiki 多证据链上有足够 model-val 样本；
- Generator 配方选择不只由 NIAH 驱动；
- citation/chain 退化和 relation 错误有最低可解释规模；
- 后续 GR-F/GR-C maximin 选择不建立在太小的验证集上。

所以，G210 的正式结论仍然是：

```text
G210 did not freeze training data.
G300 cannot start from the failed G210 dataset.
```

但这不等于：

```text
Generator repair and Generator-aware Selector are meaningless.
```

## 已观察到的积极信号

| 检查项 | G210/G210 triage 结果 | 解释 |
|---|---:|---|
| structural audit | 3,108/3,108 pass | 数据结构、引用索引、schema 没有系统性破坏 |
| train/model-val group overlap | 0 | 没有按问题组泄漏 |
| train/model-val component overlap | 0 | 没有按 component 泄漏 |
| NIAH train/model-val | 515 / 215 | NIAH 训练和新 model-val 均高于最低门 |
| 2Wiki train | 683 | 多跳训练规模仍高于最低门 |
| 2Wiki model-val | 76 | 未达 formal freeze 门，但不是 0 或随机崩溃 |
| unsupported update ratio | 12.4855% | 仍在 10%-15% 预算内 |
| prohibited data | not read | sealed600/system held-out/dev 边界保持 |
| training / utility | not started | 没有在失败数据上训练 |
| failure localization | relation templates | 失败集中在 `country`、`publication date`、`country of citizenship` 等 target construction/audit mismatch |

这些信号支持“继续修数据”，不支持“直接训练”。

## 修订后的决策

G215 将路线状态从：

```text
G210 FAIL -> project blocked until new authorization
```

改为：

```text
G210 FAIL -> positive-signal data revision authorized -> G200R/G210R
```

重要边界：

- 不把 `76/100` 改写为通过；
- 不降低 TRUE threshold；
- 不替换 TRUE checkpoint；
- 不读 sealed600、HotpotQA、MuSiQue-Full、RGB 或 2Wiki official dev；
- 不在原 G210 failure 数据上启动 Generator training；
- 不生成 utility labels；
- 不推进 S/I/H。

## G200R/G210R 恢复路径

下一步执行新的数据修订阶段：

```text
G200R revised data materialization
-> G210R revised target audit
-> G300 only if G210R passes
```

G200R 允许：

- 根据 G210 triage 修订 2Wiki relation 模板；
- 在 TRUE audit 前固定 support-sentence 对齐规则；
- 在 TRUE audit 前固定 yes/no target 规则；
- 在 2Wiki official train 内扩大预审计候选池；
- 保持 component isolation、denylist 和 ordered IDs/SHA256。

G210R 必须重新产出：

- structural audit rows；
- TRUE audit rows；
- minimal support / citation / unsupported absence checks；
- manual audit sampling report；
- length/truncation audit；
- machine-readable manifest；
- ordered IDs and SHA256；
- server runtime identity。

只有 G210R 通过正式数据冻结门后，G300 才能开始。

## 若 G210R 仍失败

如果 G210R 仍无法形成足够 2Wiki model-val，不能继续假装正式 Generator 数据已经合格。届时只允许两种结论：

1. 记录数据修订失败，停止正式 G 阶段；
2. 由用户另行授权一个明确标为 exploratory 的小 pilot，但该 pilot 不能冻结 GQ，不能生成正式 utility labels，不能支持 SystemF。

## 本阶段产物

| 产物 | 用途 |
|---|---|
| `G215_GATE_AND_DATA_REVISION_AMENDMENT.md` | human-readable amendment |
| `artifacts/G215/gate_amendment_manifest.json` | machine-readable amendment manifest |

Manifest SHA256:

```text
274dff7714a7ee001bf737f1c85f1275a34bfed1fbdc9e7839b2beefc9719110  artifacts/G215/gate_amendment_manifest.json
```

## 阶段判定

G215 通过。当前路线可以继续到 G200R/G210R 数据修订，但仍不能启动训练或 held-out。
