# R002 — 指标、连通分量与 CRC 样本量协议报告

**Run：** `R002 — metric-and-crc-unit-protocol`

**状态：** `COMPLETE / SAMPLE-SIZE GO`

**下一步：** `R003 — topk-and-count-controls`

**代码：** `refactor/three-module-baseline@ffe2d27411e6c4848debcb2887405c045cea0541`

**机器可读证据：** [`R002_VALIDATION_REPORT.json`](R002_VALIDATION_REPORT.json)

## 1. 零基础版结论

R002 完成的不是“训练出了一个能安全删除证据的 Selector”，而是先把以后判断它是否安全的**尺子、分组和考场座位**固定下来：

1. 同一道问题在 TopK10 和 Selector 两边必须一一对应，不能悄悄丢掉难例；
2. 正确证据的保留与多跳链完整性只按 `document_id` 判断；
3. 有共同来源、family 或 official supporting document 的问题不能冒充相互独立，而要归入同一 component；
4. CRC 校准时每个 component 最多贡献一个预先冻结的代表问题；
5. calibration 与 decision-dev 之间没有 component 穿越；
6. 四项必需风险的 calibration component 数都不少于 `99`，因此 `α=1%` 的非零策略在数学上**有可能**通过，允许继续 R003。

这里的 `GO` 仅表示“样本量和协议足够继续”，绝不表示某个 Selector、阈值或删除策略已经通过 CRC。当前没有 scorer、没有真实策略 loss，也没有读取 sealed/heldout 上的 Selector 效果。

## 2. 为什么要按 component 分组

如果两个问题共享同一个来源文档或同一个 synthetic family，它们的错误往往会一起发生。把它们当成两个完全独立样本，会让不确定性看起来虚假地变小。

R002 使用 allowed-key 连通分量：只要两个问题通过允许的 key 直接或间接相连，就归为同一 component。

- NIAH 只使用 `query + assignment 标注的 required/needle source parent + synthetic family`；
- 2Wiki 只使用 `query + official supporting/gold document-title parent`；
- 2Wiki 的 Top20 distractor candidate parent **绝不建边**，否则 train/dev/heldout 每个 split 都会错误塌成一个巨型 component；
- component ID 是该 component 内排序后 query IDs 的稳定 SHA-256；
- component 按“问题数从大到小”确定顺序，再分配到当前问题数最少的 fold；平局规则固定，因此输入顺序不会改变结果。

## 3. 数据角色冻结结果

| 数据 | 可用问题 | Components | 派生角色 | 角色问题数 | 角色 component 数 | Role crossing |
|---|---:|---:|---|---:|---:|---:|
| NIAH train | 1,023 | 908 | train-fit / train-modelval | 920 / 103 | 817 / 91 | 0 |
| NIAH dev | 1,479 | 1,057 | crc-calibration / decision-dev | 740 / 739 | 523 / 534 | 0 |
| 2Wiki train | 3,000 | 2,324 | train-fit / train-modelval | 2,700 / 300 | 2,094 / 230 | 0 |
| 2Wiki dev | 2,000 | 1,732 | crc-calibration / decision-dev | 1,000 / 1,000 | 866 / 866 | 0 |

四套 artifact 的 `components_across_folds=0` 且 `components_across_roles=0`。train 的 representative 文件按协议为空：train-modelval 可以做诊断，但不能冒充 CRC calibration。只有 dev 的 `crc-calibration` 角色生成 CRC representatives；未来 decision-dev 推理使用其全部合格问题，component 只作为 cluster bootstrap 单位，不抽成一题。

## 4. 四项 CRC 样本量门

CRC 使用修正式：

```text
corrected_risk = (观测 loss 总和 + 1) / (component 代表数 n + 1)
```

在还没有任何观测 loss 时，它的最低可能值是 `1/(n+1)`。当目标 `α=1%` 时，至少需要 `n≥99`，否则即使一个错误都没有，非零删除策略也不可能合格。

| 未来风险 | Calibration representatives `n` | 零 loss 修正下界 | 二元 loss 下最多可容忍的观测 loss 数 | 样本量判定 |
|---|---:|---:|---:|---|
| NIAH required recall loss | 523 | `1/524 ≈ 0.1908%` | 4 | GO |
| NIAH conditional chain loss | 402 | `1/403 ≈ 0.2481%` | 3 | GO |
| 2Wiki supporting recall loss | 866 | `1/867 ≈ 0.1153%` | 7 | GO |
| 2Wiki conditional chain loss | 468 | `1/469 ≈ 0.2132%` | 3 | GO |

这些“最多可容忍数”只是解释样本容量：以后某个非零策略仍必须用真实 loss 逐项计算，而且四项风险各自先找最强合格策略，最终只能取四者中最保守的一个。P0 是结构性 TopK10 回退，不会被写成“通过 CRC 的删除策略”。

## 5. 冻结的指标语义

- `harm_reduction = Harm(TopK10) − Harm(Selector)`；正数才是改善；
- `recall_loss = Recall(TopK10) − Recall(Selector)`；正数代表损失；
- recall 与完整链只比较 `document_id` 集合；Selector 必须是 TopK10 的删除后子集，不能偷偷补入新文档；
- conditional chain loss 只在 TopK10 原本包含完整 gold chain 的问题上计算；不合格问题返回显式 `None`，不能用 0 稀释分母；
- deletion precision 没有发生删除时返回显式 `None`，不能伪装成 0；
- paired 比较严格要求两边 query key 完全一致；
- bootstrap 以 component 为整体重采样，sign-flip 也以 component 为单位；点估计仍按 query 加权；
- Monte Carlo p-value 使用 plus-one 修正 `(extreme+1)/(iterations+1)`，不会报告不可能的 `p=0.0`；
- P0–P6 必须满足 selected set 逐级嵌套，且同一代表问题上的 loss 逐级不下降。

## 6. CRC 和 95% 置信区间不是同一件事

CRC expected-risk 是**校准阶段的策略选择规则**：它用独立 calibration 代表问题和修正式，决定以后可以采用多强的删除策略。

95% cluster-bootstrap CI 是**结果阶段的不确定性描述**：它会在 decision-dev 或正式评估中回答 harmful improvement/recall loss 的统计波动范围。

因此：CRC 合格不能替代最终 CI；最终 CI 好看也不能倒过来重调 CRC 阈值。R002 只证明两套机制已经分开实现和测试，没有声称任何真实策略已经通过其中任何一关。

## 7. 2Wiki 官方 split 的如实披露

R001 的只读重算由 R002 冻结披露：official-support component 数为 train `2,324`（最大 20）、dev `1,732`（最大 8）、heldout `1,692`（最大 13）。三个官方 split 的 query overlap 都是 `0`，但 supporting-parent overlap 不是零：train–dev `449`、train–heldout `422`、dev–heldout `434`。

heldout 成员没有为追求 overlap=0 而重分，也没有读取 heldout 的 Selector 效果。跨官方 split 的 parent 重叠将作为后续 parent-seen/unseen 敏感性变量，而不能被隐藏成“完全独立”。

## 8. 可复现性与验证

- 四套 component artifact 均完成 write-once freeze，并用独立 `--verify-only` 再验证：`4/4 PASS`；
- 本地与服务器共 24 个冻结文件逐文件 SHA-256 相同：`24/24 MATCH`；
- 预审独立计算的 component、role 与 dev representative canonical projection fingerprints 全部命中：`10/10 MATCH`；
- 服务器相关测试：`64 passed`；
- 本地全仓测试：`1237 passed`；
- 本地 Ruff 与 mypy：`PASS`；
- candidate pools 全部由 R001 的 `SelectorCandidatePoolManifestV2` 再验证通过；默认 TopK 生产行为未改变。

## 9. R002 判定

**R002 = COMPLETE；sample-size gate = GO；下一步 = R003。**

这次 GO 的准确含义是：指标、独立性分组、数据角色、代表抽样和样本量前置条件已经足够严格，可以继续建立 TopK 数量基线和等量删除对照协议。它不允许跳过 R003–R008，也不允许提前宣称 Selector 已经实现安全删除。
