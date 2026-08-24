# G212M3 sample review report

**日期：** 2026-08-18
**状态：** `NOT FREEZE READY / 91 PASS / 9 FAIL / NO TRAINING STARTED`
**输入状态：** `G212R3 LENGTH PASS / SAMPLE PENDING`

## 做了什么

G212M3 对 G212R3 生成的固定 100 条 sample 做判定，并写出 review rows、summary、ordered IDs 和 freeze readiness manifest。

这一步没有训练 Generator、没有生成 utility labels、没有读取 held-out/sealed/dev。

## 结果

| 项 | 数值 |
|---|---:|
| fixed sample rows | 100 |
| PASS | 91 |
| FAIL | 9 |
| UNCERTAIN | 0 |
| forced structural failures | 0 |

按分层统计：

| stratum | PASS | FAIL | UNCERTAIN |
|---|---:|---:|---:|
| `2wiki:train-fit:twowiki_evidence_chain:answerable` | 19 | 1 | 0 |
| `2wiki:train-fit:unsupported_support_removed:unsupported` | 20 | 0 | 0 |
| `2wiki:train-modelval:twowiki_evidence_chain:answerable` | 16 | 4 | 0 |
| `niah:train-fit:niah_qa2d_single_claim:answerable` | 17 | 3 | 0 |
| `niah:train-modelval:niah_qa2d_single_claim:answerable` | 19 | 1 | 0 |

失败行：

| sample_index | case_id | failure class |
|---:|---|---|
| 8 | `2wiki::0b03d31b08a511ebbd7dac1f6bf848b6` | film country/origin target not self-contained |
| 42 | `2wiki::f99e50a908c911ebbd92ac1f6bf848b6` | nationality target uses an indirect U.S. Olympic coach cue |
| 45 | `2wiki::d5cff1720bda11eba7f7acde48001122` | birthplace target uses nickname instead of birthplace statement |
| 57 | `2wiki::23b5787a08e911ebbda7ac1f6bf848b6` | film-origin target uses indirect tale/setting cues |
| 59 | `2wiki::7a6f25d2087411ebbd67ac1f6bf848b6` | nationality target uses army service instead of nationality |
| 64 | `niah-old::10517` | QA2D target truncates `Doctor Who` to `doctor` |
| 66 | `niah-old::10048` | QA2D target has word-order corruption around Cuba Gooding |
| 74 | `niah-old::10751` | QA2D target malformed around raised markers and visibility |
| 98 | `niah-new-modelval::1155` | QA2D target malformed around Meat Loaf / Marion Raven duet |

## 判定

G212M3 没有通过 freeze readiness：`freeze_ready=false`，`g300_unlocked=false`。因此不能进入 G300 training。

这不是路线失败。91/100 通过、unsupported 层 20/20 通过、且失败集中在可解释的 target construction/QA2D construction 问题，说明路线仍有积极信号。下一步应执行 G219 controlled target repair，修复这些问题类型后重新跑 structural、TRUE、length 和 sample gates。

## 产物

Runtime:

```text
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G212M3-v1/review
```

Git-archived artifacts:

| 产物 | SHA256 |
|---|---|
| `artifacts/G212M3/review/sample_review_rows.jsonl` | `d47364c456e46fa2d0f826458f69c74da9b796cdfa58aaa64b2e4261299853b3` |
| `artifacts/G212M3/review/sample_review_summary.json` | `937d79254249e1552dbc2164062b904803a44b0b19e21835185c4104771267ec` |
| `artifacts/G212M3/review/ordered_review_ids.json` | `f98a16f78ea3ebc1e6e2369e79ee4b8cd18e1b513eb3691f54d3e775c68a1936` |
| `artifacts/G212M3/review/freeze_readiness_manifest.json` | `d2cadd49b89b94612727fd8bc0cb0d020d6e0eccf8e9e3c0efe27097d1aa0654` |
| `artifacts/G212M3/G212M3_EXECUTION_AUDIT.json` | `d8290910fc658bffcc18bb8f60538095fd26935be6f670b2b9d32b6056769cce` |

## 下一步

G219 只能针对 G212M3 暴露的 2Wiki answerable target self-containment 和 NIAH QA2D target malformation 做受控修复。G219 不能改变 Retriever、TRUE 阈值、held-out/dev 边界、sample 判定结果或训练入口。
