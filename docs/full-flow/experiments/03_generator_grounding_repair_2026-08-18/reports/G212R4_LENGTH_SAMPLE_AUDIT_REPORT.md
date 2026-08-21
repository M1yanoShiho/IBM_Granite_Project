# G212R4 length/sample audit report

**日期：** 2026-08-18
**状态：** `LENGTH PASS / SAMPLE PENDING / NO TRAINING STARTED`
**输入状态：** `G220 PRE-SAMPLE PASS`

## 做了什么

G212R4 对 G220 conservative filtered bundle 重新执行训练前长度检查，并生成新的固定 100 条样本包。

本阶段只做两件事：

- 用 frozen Granite 4.1-3B tokenizer 统计所有 prompt/target example 长度；
- 按固定 seed `G212R4-v1-fixed-stratified-sample` 从 5 个 stratum 各取 20 条样本。

这一步没有训练模型、没有生成 utility labels、没有读取 held-out/sealed/dev，也没有改变 Retriever、TRUE、Selector 或任何 target。

## 长度结果

| 项 | 数值 |
|---|---:|
| cases | 2,689 |
| examples | 11,211 |
| max length limit | 2,304 |
| max observed length | 2,120 |
| over max length | 0 |
| truncation rate | 0.0000 |
| p50 / p90 / p95 / p99 | 1,348 / 1,841 / 1,882 / 1,966 |

分层最大长度：

| stratum | examples | max | over max |
|---|---:|---:|---:|
| 2Wiki train answerable | 4,095 | 2,031 | 0 |
| 2Wiki train unsupported | 1,048 | 1,667 | 0 |
| 2Wiki model-val answerable | 495 | 1,949 | 0 |
| NIAH train answerable | 4,096 | 2,120 | 0 |
| NIAH model-val answerable | 1,477 | 2,030 | 0 |

## 样本包

固定样本已经生成，但还没有判定：

| stratum | rows |
|---|---:|
| 2Wiki train answerable | 20 |
| 2Wiki train unsupported | 20 |
| 2Wiki model-val answerable | 20 |
| NIAH train answerable | 20 |
| NIAH model-val answerable | 20 |

所有 rows 当前都是 `PENDING`，所以 G212R4 不能解锁 G300。

## 判定

G212R4 通过长度门，说明 G220 后的数据不会因为输入过长而阻断训练前准备。这是积极信号，可以继续到 G212M4 固定样本判定。

但 G212R4 不是数据冻结，也不是训练许可。只有 G212M4 固定样本判定通过并写出 freeze readiness PASS，G300 才能开始。

## 验证

服务器测试通过：

```text
4 passed in 0.02s
```

测试范围：

- G212 prepare 写出 length pass 和 pending sample；
- length failure 不启动训练；
- G216 manifest 兼容；
- G220 conservative filter manifest 兼容。

## 产物

Runtime:

```text
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G212R4-v1
```

Git-archived artifacts:

| 产物 | SHA256 |
|---|---|
| `scripts/full_flow_g212_manual_length_audit.py` | `e87fb401298b054fa239293b4ec4c2f049d36e718f9d313dbacbc3ca67dd568b` |
| `tests/scripts/test_full_flow_g212_manual_length_audit.py` | `e4ce6049618c43edbb0544e5be57272dbc69c582cba593ce04bcaad896f3bb80` |
| `artifacts/G212R4/length_audit.json` | `73c9e83ea49e45c384c7b39784d19abb90d4189ef8a346b5799f733070371406` |
| `artifacts/G212R4/length_rows.jsonl` | `fa466b9f29e5f1775fa6d68a2cbe4d8282204bb691e5fa1aa9193342297afece` |
| `artifacts/G212R4/manual_sample.jsonl` | `6f51e289368c7f266f6458e0255c3e047c420b6fe030e238a025d90e8cee6669` |
| `artifacts/G212R4/manual_sample_summary.json` | `f29f194d1bd377f955382d81f138cd3035d3976fe6422b0352fdf515930b0bc4` |
| `artifacts/G212R4/prepare_manifest.json` | `890181593f46f0c5ae75ab6703d89bab408cdd185d5eb4c2cecda3666e42cf39` |
| `artifacts/G212R4/G212R4_EXECUTION_AUDIT.json` | `67d688358628fdd58341af069468a62cebaaeb701524a09addd51eac0c49f43c` |

## 下一步

下一步是 G212M4：对 `artifacts/G212R4/manual_sample.jsonl` 的 100 条固定样本给出判定记录，并写出 freeze readiness manifest。
