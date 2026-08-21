# G210R-v1 structural failure report

**日期：** 2026-08-18  
**状态：** `FAIL / STRUCTURAL ANSWER_ALIAS / TRUE NOT STARTED / NO TRAINING STARTED`  
**服务器 runtime：** `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G210R-v1`

## 本阶段做了什么

G210R-v1 对 G200R-v1 的数据执行 structural audit。该步骤只检查数据结构、citation remap、answer alias、split leakage，并生成 TRUE worklist；不运行 TRUE 大模型，不训练 Generator，不生成 utility labels，也不读取 held-out。

## 结果

Structural audit 没有通过：

| 项目 | 数量 |
|---|---:|
| total cases checked | 3,108 |
| NIAH train pass | 515 |
| NIAH validation pass | 307 |
| 2Wiki train answerable pass | 1,053 |
| 2Wiki train answerable fail | 22 |
| 2Wiki validation answerable pass | 133 |
| 2Wiki validation answerable fail | 3 |
| 2Wiki unsupported pass | 1,075 |
| split group overlap | 0 |
| split component overlap | 0 |

失败原因全部相同：

```text
answer_alias_mode = literal_answer_missing
```

也就是说，G200R-v1 用 support sentence 作为 target 的方向是对的，但其中 25 个 2Wiki answerable case 的 support sentence 链里没有包含 official answer。citation remap 没失败，split 没泄漏。

## 阶段影响

- TRUE audit 未启动；
- pre-manual finalize 未启动；
- 人工审计未启动；
- G300 仍 blocked；
- 正确恢复方式是回到 G200R2，在物化时过滤掉 answer alias 不保留的 2Wiki support-sentence targets。

## 关键 SHA256

| 产物 | SHA256 |
|---|---|
| `artifacts/G210R-v1/structural/structural_summary.json` | `e3e7914af5575562aa648b8c439e1468b8bb7f6249f7bdd3fa0990435c992991` |
| `artifacts/G210R-v1/structural/structural_audit_rows.jsonl` | `44b9546b61f6611108d8ddc4ced5a202f4953c672b1f70861c63f36c39d5e762` |
| `artifacts/G210R-v1/structural/true_worklist.jsonl` | `e058e24229561442c5a8752093cacaceece15b972da50b840c009be98e6eea81` |
| `artifacts/G210R-v1/structural_failure_summary.json` | `c756d78976d9978fea9571db847ff38d3ec9d9a5f66eed4a564351bef0bdbda7` |

## 阶段判定

G210R-v1 failed before TRUE. This is a data construction failure, not a Generator training result. The next allowed step is G200R2 data materialization with an explicit answer-alias preservation filter for 2Wiki support-sentence targets.
