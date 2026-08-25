# R003 — TopK 数量基线与等量删除协议报告

**Run：** `R003 — topk-and-count-controls`

**状态：** `COMPLETE / BASELINE-PROTOCOL PASS`

**下一步：** `R004 — label-audit-and-200q-preflight`

**代码：** `refactor/three-module-baseline@c23c4df71ba61c95b1acb72411684e6a62ff3e77`

**机器可读证据：** [`R003_VALIDATION_REPORT.json`](R003_VALIDATION_REPORT.json)

## 1. 零基础版结论

R003 回答了一个关键问题：**如果完全不判断证据对错，只是把每个问题的最后几条证据机械删掉，会发生什么？**

结果非常明确。最轻的固定删除方案 `TopK9`，也就是每个问题都从默认 `TopK10` 中固定删掉排名最后的 1 条，在 NIAH dev 上：

- harmful exposure 改善约 **2.04 个百分点**；
- 但正确证据 recall 损失约 **1.90 个百分点**；
- 原本完整的多跳证据链损失约 **4.47 个百分点**；
- 删除的 1,479 条候选中，只有 27 条是已标注 harmful，删除精度约 **1.83%**。

所以，“每题最多删一条”不是这里的问题；真正的问题是“**强制每题都删一条**”。有些问题没有一条证据值得冒险删除。后续方案必须允许：有把握时删 1–3 条，没有把握时删 0 条并完整回退到 TopK10。

R003 的 `PASS` 只表示数量基线、指标分母、统计比较和等量删除生成协议已经完整、可复算。它**不表示 Selector 成功**，因为此时没有训练 scorer、没有真实 Selector 决策，也没有任何非零策略通过 CRC。

## 2. 为什么固定 TopK9 仍不够保守

以 NIAH dev 的 1,479 个合格问题为例：

| 系统 | 每题固定删除数 | Top20 pool-conditional harmful 改善 | 正确证据 recall 损失 | TopK10 完整链条件损失 |
|---|---:|---:|---:|---:|
| TopK10 | 0 | 0.00 pp | 0.00 pp | 0.00 pp |
| TopK9 | 1 | 2.04 pp | 1.90 pp | 4.47 pp |
| TopK8 | 2 | 5.14 pp | 4.18 pp | 8.84 pp |
| TopK7 | 3 | 8.62 pp | 6.88 pp | 13.42 pp |

表格展示的是很典型的“删得越多，错证据少得越多，但正确证据也丢得越多”。这不是我们想要的 Selector 作用，因为纯粹减少上下文数量就能得到这种现象。

TopK9 的 harmful 改善不是随机波动的漂亮点值：component-cluster 95% CI 为约 `[1.37, 2.77] pp`，没有跨过 0。可是它的 recall 损失 CI 也约为 `[1.51, 2.29] pp`，完整链损失 CI 约为 `[3.17, 5.85] pp`。也就是说，它可信地删掉了一点 harmful，也可信地伤害了正确证据；后两项超过本项目“recall 点损失尽量不超过 1 pp、链损失同样优先压低”的保守目标。

这给下一阶段的直接要求是：模型不能只学会“删”；它必须学会**哪些题应该不删、哪些候选才值得删**。因此后续的 0–cap1/2/3 是“每题最多能删多少”的保险丝，不是“每题必须删满多少”的命令。

## 3. 2Wiki 也证明数量缩减会损伤证据链

2Wiki 没有可合法使用的 harmful 标签，所以这里不报告或猜测 harmful reduction。它只检查 official supporting documents 是否被保留。

在 2Wiki dev 上，TopK9 相对 TopK10 的 supporting-document recall 损失为 `0.375 pp`，完整链条件损失为约 `1.50 pp`。继续缩到 TopK8/7 后，完整链损失增至约 `2.68/6.22 pp`。这说明哪怕单条 supporting document 的平均 recall 看起来只小幅变化，多跳问题需要的“整条链”仍可能更快断裂，所以 chain risk 必须继续作为独立安全门。

## 4. 三种 harmful 分母为什么都保存

NIAH 的同一个数可以因分母不同而回答不同问题。R003 同时保存：

1. **Top20 pool-conditional：** 只看 Top20 原本含有 harmful 的问题，回答“有东西可删时，暴露减少多少”；
2. **TopK10 baseline-exposed：** 只看 harmful 已进入默认 TopK10 的问题，回答“面对当前基线已经暴露的风险，删掉了多少”；
3. **Unconditional：** 看全部合格注入问题，回答“放回真实问题总体后，平均改善多少”。

不能只挑最大的那个数字。主张 C1 使用预先约定的 Top20 pool-conditional 指标，同时报告另外两种口径和各自真实分母。NIAH dev 的三个 TopK9 点改善分别为约 `2.04 pp`、`2.32 pp` 和 `1.83 pp`。

## 5. 公平的等量删除对照已经冻结，但结果尚未产生

以后真实 Selector 可能对某题删 0 条、另一题删 2 条。为了判断改善来自“真的找对了 harmful”，还是仅仅来自“少给了两条证据”，随机删除和删最低排名必须对每道题复制 Selector 的实际删除数量。

R003 已冻结：

- master seed `20260811`；
- 100 个随机重复，索引 `0..99`；
- 删除数只允许 `0/1/2/3`；
- random-priority 和 bottom-rank 的稳定排序规则；
- 不允许从 rank 11–20 补位；
- 同一题同一重复中，删 1 条的集合必须是删 2 条集合的子集。

但 R003 没有真实 Selector trace，不知道每道题实际会删几条。因此现在只有可复现的**生成协议**，count-matched 数值结果明确标为 `DEFERRED`。提前生成一个虚假的随机结果会破坏公平性。

## 6. 冻结范围与完整性

正式产物覆盖四个 source split：

| 数据 | 合格问题数 | R002 components | TopK10/9/8/7 trace rows |
|---|---:|---:|---:|
| NIAH train | 1,023 | 908 | 4,092 |
| NIAH dev | 1,479 | 1,057 | 5,916 |
| 2Wiki train | 3,000 | 2,324 | 12,000 |
| 2Wiki dev | 2,000 | 1,732 | 8,000 |

- 四个 TopK 任务和一个 count-matched protocol 均正式生成成功：`5/5`；
- 用相同冻结输入执行独立 `--verify-only`：`5/5 PASS`；
- 临时目录只有预期的 14 个原始文件，随后整体原子移动到正式 R003；
- 复制回本地后，本地与服务器逐文件 SHA-256：`14/14 MATCH`；
- 服务器直接重算四套 TopK manifest 绑定的全部上游输入：`54/54 input pins MATCH`；
- 独立解析并复算全部 `30,008` 行 trace：四个 K 完整性、前缀嵌套、删除后缀、R002 role/component/chain 绑定均为 `0` 异常，2Wiki harmful 字段数为 `0`；
- 服务器相关测试：`60 passed`；本地新增定向测试：`10 passed`；本地全仓：`1268 passed`；Ruff 与 mypy：`PASS`；
- [`CHECKSUMS.sha256`](CHECKSUMS.sha256) 只覆盖这 14 个原始生成文件。随后编写的报告、配置、run manifest 和日志没有加入该文件，以避免封存说明相互自引用；它们由 Git 提交本身追踪。

本轮只读取 train/dev 的冻结 pool 与 R002 component artifact，没有读取 sealed/heldout 上的 Selector 效果，也没有改变生产默认 `top-k`。

## 7. R003 判定

**R003 = COMPLETE；Gate 1 = PASS；下一步 = R004。**

准确含义是：我们现在有了一把公平的“少给几条证据”尺子，并且已经知道固定 TopK9 虽能减少一点 harmful，却不能满足当前保守目标。R004 可以开始做标签与 200-query 资源预检；它仍不能把 R003 写成 Selector 已经有效，也不能跳过 scorer、CRC calibration 和 decision-dev 的后续关卡。
