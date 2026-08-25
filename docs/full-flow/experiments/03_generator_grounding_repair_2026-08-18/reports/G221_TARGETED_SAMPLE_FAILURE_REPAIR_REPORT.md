# G221 targeted sample-failure repair report

**日期：** 2026-08-18
**状态：** `PRE-SAMPLE PASS / G212R5 NEXT / NO TRAINING STARTED`
**输入状态：** `G212M4 FAIL / 90 PASS / 10 FAIL / 0 UNCERTAIN`

## 做了什么

G221 只处理 G212M4 暴露的 10 条失败样本。它没有改写 target，而是采用保守隔离：

- 从 G220 bundle 中隔离 G212M4 失败的 10 条 case；
- 对 2 条失败的 2Wiki train answerable case，同步隔离对应 unsupported counterpart；
- 重新写 train/validation cases、manifest 和 ordered IDs；
- 记录 2Wiki model-val screen 从 99 缩到 96 的方法学限制。

这一步没有训练模型、没有生成 utility labels、没有读取 held-out/sealed/dev，也没有改变 Retriever、TRUE 模型或 TRUE 阈值。

## 结果

Runtime:

```text
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G221-v1
```

| 项 | 数值 |
|---|---:|
| train cases | 2,374 |
| validation cases | 303 |
| NIAH train/model-val | 511 / 207 |
| 2Wiki train/model-val | 817 / 96 |
| unsupported groups | 1,046 |
| unsupported update ratio | 11.3461% |
| split group/component overlap | 0 / 0 |
| 隔离的固定样本失败 rows | 10 |
| 实际隔离 cases | 12 |

G221 的自动门全部通过，状态为 `PRE_MANUAL_PASS`。因为它只做过滤、没有新增或改写 target，已通过的 structural/TRUE 结果仍可继承；下一步仍必须重跑长度检查和新的固定样本判定。

## 判定

G221 是积极信号：失败行可以被局部隔离，train 规模、NIAH model-val、unsupported ratio 和 split 隔离仍在可控范围内。

但 G221 不是训练许可。2Wiki model-val screen 已缩到 96，后续报告必须如实说明这个内部验证规模较原 `>=100` 保护更弱。只有 G212R5 长度通过、G212M5 固定样本判定通过并写出 freeze readiness PASS，G300 才能开始。

## 验证

服务器测试通过：

```text
... [100%]
```

测试范围：

- G221 targeted sample-failure repair test；
- G220 conservative filter test；
- G216 sample-review repair test。

## 产物

Git-archived artifacts:

| 产物 | SHA256 |
|---|---|
| `scripts/full_flow_g221_targeted_sample_failure_repair.py` | `38eebba936f6f1836aabd145e018fb45cac4df7b687d3b4dd8f7101c1436d9b7` |
| `tests/scripts/test_full_flow_g221_targeted_sample_failure_repair.py` | `9ebb48bce3b0366661778f386f43769b144cafe58ad46db78f98b2cbdfe212c6` |
| `artifacts/G221/data/manifest.json` | `6a9ab588a16f9c46ff784b4e74aa0b39fafab8f98b266599c4af2b1b94ee5efc` |
| `artifacts/G221/data/ordered_ids.json` | `377dea044dd66eb262644ef2a8ac0876e682eb6aa47a4ed875e91dc3e31b0726` |
| `artifacts/G221/G221_EXECUTION_AUDIT.json` | `d923813e1e0ffb11988385a7f19c58f1ecbbb95c8151f2dd000635a10dfe1005` |

## 下一步

下一步是 G212R5：对 G221 bundle 重跑 length/sample prepare。G212R5 只能生成长度报告和固定样本包；G212M5 判定通过前仍不能训练。
