# G212M4 sample review report

**日期：** 2026-08-18
**状态：** `NOT FREEZE READY / 90 PASS / 10 FAIL / NO TRAINING STARTED`
**输入状态：** `G212R4 LENGTH PASS / SAMPLE PENDING`

## 做了什么

G212M4 对 G212R4 固定的 100 条样本写出判定记录、summary、ordered IDs 和 freeze readiness manifest。

这一步没有训练 Generator、没有生成 utility labels、没有读取 held-out/sealed/dev。

## 结果

| 项 | 数值 |
|---|---:|
| fixed sample rows | 100 |
| PASS | 90 |
| FAIL | 10 |
| UNCERTAIN | 0 |
| forced structural failures | 0 |

按分层统计：

| stratum | PASS | FAIL | UNCERTAIN |
|---|---:|---:|---:|
| `2wiki:train-fit:twowiki_evidence_chain:answerable` | 18 | 2 | 0 |
| `2wiki:train-fit:unsupported_support_removed:unsupported` | 20 | 0 | 0 |
| `2wiki:train-modelval:twowiki_evidence_chain:answerable` | 17 | 3 | 0 |
| `niah:train-fit:niah_qa2d_single_claim:answerable` | 19 | 1 | 0 |
| `niah:train-modelval:niah_qa2d_single_claim:answerable` | 16 | 4 | 0 |

失败行：

| sample_index | case_id | failure class |
|---:|---|---|
| 3 | `2wiki::42dd918208ec11ebbda7ac1f6bf848b6` | cited evidence does not support Mike Gottfried identity/nationality target |
| 14 | `2wiki::f72b5c84086b11ebbd60ac1f6bf848b6` | birth-place question not preserved for Kwek Leng Beng |
| 47 | `2wiki::e163e5f9085f11ebbd5dac1f6bf848b6` | birth-place question not preserved for Alexander Nevsky |
| 52 | `2wiki::ae5facbc08d411ebbd96ac1f6bf848b6` | Indian Creek target does not state a country |
| 57 | `2wiki::15c309eb08ed11ebbda7ac1f6bf848b6` | nationality inferred indirectly from Union Army/New Mexico roles |
| 77 | `niah-old::10038` | QA2D says Graham number value is `n` |
| 83 | `niah-new-modelval::10718` | QA2D says hip bone is a synovial joint rather than preserving hip joint relation |
| 86 | `niah-new-modelval::11295` | evidence does not directly support alpacas as the fine-wool animal |
| 90 | `niah-new-modelval::1166` | evidence covers publication/printing timeline, not writing in 1886 |
| 91 | `niah-new-modelval::11044` | target overstates half-dollar production stop; evidence says general-circulation production ceased |

## 判定

G212M4 没有通过 freeze readiness：`freeze_ready=false`，`g300_unlocked=false`。因此不能进入 G300 training。

这仍不是路线失败。长度检查通过、unsupported 样本 20/20 通过、90/100 样本可用，说明 G220 的保守路线有积极信号。失败也不是随机扩散，而是集中在两类可定位问题：

- 2Wiki answerable target 仍有少量出生地/国家/国籍关系没有写成自洽、直接受证据支持的 claim；
- NIAH QA2D 仍有少量问题语义被改窄、改错或证据只间接支持的句子。

下一步应执行 G221 targeted sample-failure repair review。G221 不能训练，不能读取 held-out，也不能把 G212M4 改写为通过；它只能基于这些失败类型做受控修复或过滤，并在修复后重跑 length 和固定样本判定。

## 产物

Runtime:

```text
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G212M4-v1/review
```

Git-archived artifacts:

| 产物 | SHA256 |
|---|---|
| `scripts/full_flow_g212m_sample_adjudication.py` | `654150d2cca2b171757f6e8163ffbc4dc17d032f686ce7a062890f86f83377e3` |
| `tests/scripts/test_full_flow_g212m_sample_adjudication.py` | `986def015da7649f012e69bc103f338199362fa3ba96fd0be20ca7e1bffbf9d9` |
| `artifacts/G212M4/review/sample_review_rows.jsonl` | `35250917fe52363dbd837436dd68d62931e049c02e6ee6a9a4d25789f00cddb7` |
| `artifacts/G212M4/review/sample_review_summary.json` | `8998f7bb48820f3bb9b2c51488e90e2a3063b02543497b2108cff63d5a42b587` |
| `artifacts/G212M4/review/ordered_review_ids.json` | `1e2b9e0cf4824a7722ebc0697cfa3d0ce3d74f822c0d08f3769e0905d5394ad6` |
| `artifacts/G212M4/review/freeze_readiness_manifest.json` | `bf8be8327b66cd56ca607f181cc5867a2ca102f6d3259a9fcbd0d202ab98b96f` |
| `artifacts/G212M4/G212M4_EXECUTION_AUDIT.json` | `7167da19d63970b35a7218d485c6b2b3b2e5da1941d1153cd3b16311dc3f940a` |

## 下一步

G221 应只围绕 G212M4 暴露的 10 条失败类型做修复/过滤决策，并重新生成可审计的 manifest、ordered IDs 和 hash。G221 通过后仍要重跑 length/sample；只有新的 freeze readiness PASS 才能进入 G300。
