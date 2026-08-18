# G212R5 length/sample audit report

**日期：** 2026-08-18
**状态：** `LENGTH PASS / SAMPLE PENDING / NO TRAINING STARTED`
**输入状态：** `G221 PRE-SAMPLE PASS`

## 做了什么

G212R5 对 G221 targeted filtered bundle 重新执行训练前长度检查，并生成新的固定 100 条样本包。

本阶段只做两件事：

- 用 frozen Granite 4.1-3B tokenizer 统计所有 prompt/target example 长度；
- 按固定 seed `G212R5-v1-fixed-stratified-sample` 从 5 个 stratum 各取 20 条样本。

这一步没有训练模型、没有生成 utility labels、没有读取 held-out/sealed/dev，也没有改变 Retriever、TRUE、Selector 或任何 target。

## 长度结果

| 项 | 数值 |
|---|---:|
| cases | 2,677 |
| examples | 11,148 |
| max length limit | 2,304 |
| max observed length | 2,120 |
| over max length | 0 |
| truncation rate | 0.0000 |
| p50 / p90 / p95 / p99 | 1,348 / 1,842 / 1,882 / 1,966 |

分层最大长度：

| stratum | examples | max | over max |
|---|---:|---:|---:|
| 2Wiki train answerable | 4,085 | 2,031 | 0 |
| 2Wiki train unsupported | 1,046 | 1,667 | 0 |
| 2Wiki model-val answerable | 480 | 1,949 | 0 |
| NIAH train answerable | 4,088 | 2,120 | 0 |
| NIAH model-val answerable | 1,449 | 2,030 | 0 |

## 样本包

固定样本已经生成，但还没有判定：

| stratum | rows |
|---|---:|
| 2Wiki train answerable | 20 |
| 2Wiki train unsupported | 20 |
| 2Wiki model-val answerable | 20 |
| NIAH train answerable | 20 |
| NIAH model-val answerable | 20 |

所有 rows 当前都是 `PENDING`，所以 G212R5 不能解锁 G300。

## 判定

G212R5 通过长度门，说明 G221 后的数据不会因为输入过长而阻断训练前准备。这是积极信号，可以继续到 G212M5 固定样本判定。

但 G212R5 不是数据冻结，也不是训练许可。只有 G212M5 固定样本判定通过并写出 freeze readiness PASS，G300 才能开始。

## 验证

服务器测试通过：

```text
5 passed in 0.03s
```

测试范围：

- G212 prepare 写出 length pass 和 pending sample；
- length failure 不启动训练；
- G216 manifest 兼容；
- G220 conservative filter manifest 兼容；
- G221 targeted sample-failure repair manifest 兼容。

## 产物

Runtime:

```text
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G212R5-v1
```

Git-archived artifacts:

| 产物 | SHA256 |
|---|---|
| `scripts/full_flow_g212_manual_length_audit.py` | `c485ee52d607f8254fe0969ff984e5f16334a08fc4c3ede429bbeeefa8e5f63f` |
| `tests/scripts/test_full_flow_g212_manual_length_audit.py` | `e12f56037635c4f88a111ac63c49517a67dec93a5dfb0ae81ae451170cf9e26f` |
| `artifacts/G212R5/length_audit.json` | `df7ac5f568c583298210a4123a7cd734a161d9e05d62a07905451c9925e1a978` |
| `artifacts/G212R5/length_rows.jsonl` | `3d89b9178d5d4f1295e461d904a54b0989b0445550e301112e079a11bca5ed5a` |
| `artifacts/G212R5/manual_sample.jsonl` | `488483b1d6752d014b82b792e12925dbaac0ac9f6af1375aced963b2a7fc426f` |
| `artifacts/G212R5/manual_sample_summary.json` | `8c31e550c6ef15d962a7e008c4df9a010901159a1c0e8b137b0cf7011d5c4152` |
| `artifacts/G212R5/prepare_manifest.json` | `7f0c81e8a04e77592a825cd6459eff843d194fc5a22d414d228510887e22b6bb` |
| `artifacts/G212R5/G212R5_EXECUTION_AUDIT.json` | `939da8ce725164e05884737b54a51e343d5bcf1124b49736c2eee33058aed323` |

## 下一步

下一步是 G212M5：对 `artifacts/G212R5/manual_sample.jsonl` 的 100 条固定样本给出判定记录，并写出 freeze readiness manifest。
