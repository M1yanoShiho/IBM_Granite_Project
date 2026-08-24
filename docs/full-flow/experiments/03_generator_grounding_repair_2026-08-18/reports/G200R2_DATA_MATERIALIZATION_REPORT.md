# G200R2 数据修订物化报告

**日期：** 2026-08-18  
**状态：** `COMPLETE / PRE-AUDIT PASS / G210R2 READY / NO TRAINING STARTED`  
**服务器 runtime：** `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G200R-v2`

## 本阶段做了什么

G200R2 修复 G210R-v1 structural audit 暴露的问题：G200R-v1 中有 25 个 2Wiki support-sentence targets 没有保留 official answer alias。

G200R2 在 2Wiki target materialization 时新增过滤：

```text
如果 answer 不是 yes/no，semantic_target 必须包含 official answer；
如果 answer 是 yes/no，仍要求至少两条 support sentence 构成 evidence chain。
```

这不是降低门槛，而是在 G200R2 物化阶段提前执行 G210R-v1 已经证明必要的结构约束。

## 边界

- `training_started=false`；
- `sealed_or_heldout_read=false`；
- `dev_read=false`；
- gold/reference/provenance 只用于离线 target construction；
- TRUE checkpoint、TRUE threshold 和 judge 角色未改变；
- 没有运行 TRUE audit；
- 没有生成 utility labels；
- G210R2 通过前不允许启动 G300。

## 结果

| 项目 | 数量 |
|---|---:|
| NIAH train answerable groups | 515 |
| NIAH 新 model-val groups | 307 |
| 2Wiki train answerable groups | 1,053 |
| 2Wiki model-val answerable groups | 133 |
| 2Wiki train unsupported groups | 1,053 |
| train cases | 2,621 |
| validation cases | 440 |
| answerable train updates | 9,385 |
| unsupported updates | 1,053 |
| unsupported update ratio | 10.0881% |
| split group overlap | 0 |
| split component overlap | 0 |
| 2Wiki target construction | `support_sentence_aligned_v1` + answer-alias preservation |

G200R2 预物化硬门全部通过：NIAH train >=400、NIAH 新 model-val >=100、2Wiki train >=400、2Wiki model-val >=100、unsupported update ratio 在 10%-15%、split leakage 为 0。

相比 G200R-v1，2Wiki answerable train 从 1,075 降到 1,053，model-val 从 136 降到 133；仍高于正式最低门。

## 关键 SHA256

| 产物 | SHA256 |
|---|---|
| runtime `train_cases.jsonl` | `131fe0daacccc5c71a81db2f2eec55ecf56f6661263317299a40c991f19da1ec` |
| runtime `validation_cases.jsonl` | `87a47d81dc87ba311f4819e6f872671d2eb1dde7c1b315ecc13f9ed775a3d035` |
| `artifacts/G200R2/data/manifest.json` | `84e8fc6f7701e91e289d51df3916c29beccd760391fa4f67c0e6a8d93a84cff5` |
| `artifacts/G200R2/data/ordered_ids.json` | `8c7490cd7a7d4a842137e8d72acf39ad76041c23ea57a5fa1ec551e9389c28b1` |
| `artifacts/G200R2/G200R2_EXECUTION_AUDIT.json` | `84de66bf176864c1c76d62d18515a31a024b0ad3e1db67d8e6267d6ee3cfd61b` |
| runtime code `full_flow_g200_v2.py` | `b1c844c32d58c6f15a9db9dd05f4cd85da894c7a1a72ea4e61bfa0828b57f0f4` |

## 验证

- 本地语法检查：`scripts/full_flow_g200_v2.py`、`scripts/full_flow_g210_audit_v2.py` pass；
- 本地单元测试：G200/G210 相关测试 5 passed；
- 服务器 materialization exit 0；
- runtime train/validation cases 留在服务器，Git 只归档 manifest、ordered IDs 和 execution audit。

## 阶段判定

G200R2 通过 pre-audit，可以进入 G210R2。G210R2 必须重新执行 structural audit、TRUE audit、finalize、人工样本、minimal support 和 length/truncation 审计。G210R2 通过前，仍不允许启动 G300、Generator 训练、utility labels、S/I/H 或 held-out。
