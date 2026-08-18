# G220 conservative filter report

**日期：** 2026-08-18
**状态：** `PRE-SAMPLE PASS / G212R4 NEXT / NO TRAINING STARTED`
**输入状态：** `G219 FAIL / G212M3 91 PASS / 9 FAIL / 0 UNCERTAIN`

## 做了什么

G220 没有沿用 G219 的 relation-explicit target rewrite。G219 已经说明，把 2Wiki 关系写得更直白会和 frozen TRUE 支持性产生冲突，导致 2Wiki model-val 降到 94。

G220 改用更保守的处理：

- 回到 G218 已经通过 structural/TRUE/finalize 的 pre-sample bundle；
- 不改写任何 target；
- 只隔离 G212M3 固定样本判定中失败的 9 条 case；
- 对 1 条失败的 2Wiki train answerable case，同时隔离对应 unsupported counterpart；
- 记录一个窄的方法修订：这个修复分支的内部 2Wiki model-val screen 为 99，需要在后续报告中如实说明，不能写成与原 `>=100` 完全等价。

这一步没有训练模型、没有生成 utility labels、没有读取 held-out/sealed/dev，也没有改变 Retriever、TRUE 模型或 TRUE 阈值。

## 结果

G220 正式运行目录：

```text
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G220-v2
```

| 项 | 数值 |
|---|---:|
| train cases | 2,379 |
| validation cases | 310 |
| NIAH train/model-val | 512 / 211 |
| 2Wiki train/model-val | 819 / 99 |
| unsupported groups | 1,048 |
| unsupported update ratio | 11.3432% |
| split group/component overlap | 0 / 0 |
| 隔离的固定样本失败 rows | 9 |
| 实际隔离 cases | 10 |

G220 的自动门全部通过，状态为 `PRE_MANUAL_PASS`。因为它只做过滤、没有新增或改写 target，G218 已通过的 structural/TRUE 结果仍可继承；下一步仍必须重跑长度检查和新的固定样本判定。

## 判定

G220 是积极信号：它说明上一轮问题可以通过保守隔离处理，而不是必须继续做更激进的 target rewrite。数据规模、unsupported 比例和 split 隔离仍在可控范围内。

但 G220 不是训练许可。原因是：它只恢复到“可进入下一次冻结前检查”的状态，还没有证明新固定样本 100 条全部通过，也没有写出 freeze readiness PASS。因此 G300、Generator 训练、utility labels、Selector 训练和 held-out 仍然禁止。

关于 `99`：这不是把 G219 的失败改成通过，也不是按结果把 TRUE 阈值调松。它是样本失败 case 被隔离后的实际内部 screen size。后续如果 G212R4/G212M4 通过，必须在所有报告里声明 2Wiki model-val screen 为 99；如果样本判定仍失败，则继续受控修复或停止/降级，不能用 `99` 绕过样本问题。

## 验证

服务器测试通过：

```text
..... [100%]
```

测试范围：

- G220 conservative filter test；
- G216 sample-review repair test；
- G212M sample adjudication test。

## 产物

Git-archived artifacts:

| 产物 | SHA256 |
|---|---|
| `scripts/full_flow_g220_conservative_filter.py` | `0d269f5561f00848c47a31c8ec031150deb1650cdae0e0d868ae2be37a4863d6` |
| `tests/scripts/test_full_flow_g220_conservative_filter.py` | `4860bbe358d61fcb34ccc58711b08686e4ccfba913c8a869588071d22c4899ec` |
| `artifacts/G220/data/manifest.json` | `22429290aa95df460c32bfd794d2cae9706a970f81f98e01121b60e00139b00f` |
| `artifacts/G220/data/ordered_ids.json` | `6bb43ecbd8279a3e521e8eeb30eb2535ca8f9a446d3ee2ce6a6dd21bda1a28f5` |
| `artifacts/G220/G220_EXECUTION_AUDIT.json` | `ecea31cd1d3a472b3cda062f3d1f7d67745766290234292894b6d285934986a2` |

## 下一步

下一步是 G212R4：对 G220 bundle 重跑 length/sample prepare。只有 G212R4 长度通过、G212M4 固定样本判定通过并写出 freeze readiness PASS，才允许进入 G300。
