# G212M5 sample review report

**日期：** 2026-08-18
**状态：** `NOT FREEZE READY / 97 PASS / 3 FAIL / NO TRAINING STARTED`
**输入状态：** `G212R5 LENGTH PASS / SAMPLE PENDING`

## 做了什么

G212M5 对 G212R5 固定的 100 条样本写出判定记录、summary、ordered IDs 和 freeze readiness manifest。

这一步没有训练 Generator、没有生成 utility labels、没有读取 held-out/sealed/dev。

## 结果

| 项 | 数值 |
|---|---:|
| fixed sample rows | 100 |
| PASS | 97 |
| FAIL | 3 |
| UNCERTAIN | 0 |
| forced structural failures | 0 |

按分层统计：

| stratum | PASS | FAIL | UNCERTAIN |
|---|---:|---:|---:|
| `2wiki:train-fit:twowiki_evidence_chain:answerable` | 18 | 2 | 0 |
| `2wiki:train-fit:unsupported_support_removed:unsupported` | 20 | 0 | 0 |
| `2wiki:train-modelval:twowiki_evidence_chain:answerable` | 19 | 1 | 0 |
| `niah:train-fit:niah_qa2d_single_claim:answerable` | 20 | 0 | 0 |
| `niah:train-modelval:niah_qa2d_single_claim:answerable` | 20 | 0 | 0 |

失败行：

| sample_index | case_id | failure class |
|---:|---|---|
| 14 | `2wiki::7e38489c0bda11eba7f7acde48001122` | evidence states France Gall collaborated with Michel Berger, not that he was her spouse |
| 17 | `2wiki::c083071908cf11ebbd95ac1f6bf848b6` | Jessica Birkel evidence gives nationality and birth date, not birthplace |
| 56 | `2wiki::23257d6e087c11ebbd69ac1f6bf848b6` | Rio Grande Band evidence does not directly state the band origin country |

## 判定

G212M5 没有通过 freeze readiness：`freeze_ready=false`，`g300_unlocked=false`。因此不能进入 G300 training。

这不是路线失败。与 G212M4 的 90/100 相比，G212M5 达到 97/100；unsupported 层 20/20，NIAH train 和 NIAH model-val 也都是 20/20。剩余失败全部集中在 2Wiki answerable 的关系自洽问题。

下一步不应直接训练，也不应把 3 条失败改写成通过。下一步应执行 G222 residual sample-failure continuation amendment：在不读取 held-out、不启动训练的前提下，明确剩余 3 条失败是继续过滤、目标修订，还是需要把冻结条件从“100/100 硬门”改为“局部缺陷可隔离但结论带限制”的规则。

## 产物

Runtime:

```text
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G212M5-v1/review
```

Git-archived artifacts:

| 产物 | SHA256 |
|---|---|
| `scripts/full_flow_g212m_sample_adjudication.py` | `654150d2cca2b171757f6e8163ffbc4dc17d032f686ce7a062890f86f83377e3` |
| `tests/scripts/test_full_flow_g212m_sample_adjudication.py` | `986def015da7649f012e69bc103f338199362fa3ba96fd0be20ca7e1bffbf9d9` |
| `artifacts/G212M5/review/sample_review_rows.jsonl` | `a79756358e00338a9af3e7a97f23b031d84091ee7b90421e46568fb816206872` |
| `artifacts/G212M5/review/sample_review_summary.json` | `e0b0c189cae4bb2ce620f5d55396a043ce7b40ef79accac7108bd0ca25be2e13` |
| `artifacts/G212M5/review/ordered_review_ids.json` | `058ab92a1035612a7ac5dae613fb15a2c80dea93992a9137fd11022c9ac6ffeb` |
| `artifacts/G212M5/review/freeze_readiness_manifest.json` | `8b15f60b184d4fa918d4f3a9682ba95b5585998dcc56193ca89ef68768c06b64` |
| `artifacts/G212M5/G212M5_EXECUTION_AUDIT.json` | `4a0076a7f491b1d1fbe17cdaa67db41716a760e570a5b6ab787e6362397162fe` |

## 下一步

G222 应只围绕 G212M5 暴露的 3 条失败和硬门规则做方法学修订或受控隔离决策。G222 不能训练、不能读取 held-out，也不能把 G212M5 改写为通过。
