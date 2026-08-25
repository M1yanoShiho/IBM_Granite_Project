# G212R3 length/sample audit report

**日期：** 2026-08-18  
**状态：** `LENGTH PASS / SAMPLE PENDING / NO TRAINING STARTED`  
**输入状态：** `G218 PRE-SAMPLE PASS`

## 做了什么

G212R3 对 G218 finalize 后的 pre-sample bundle 重跑训练前长度审计，并生成新的固定样本。

这一步没有训练模型、没有生成 utility labels、没有读取 held-out/dev。

## 结果

| 项 | 数值 |
|---|---:|
| input train cases | 2,384 |
| input validation cases | 315 |
| audited cases | 2,699 |
| audited examples | 11,268 |
| max_length | 2,304 |
| observed max length | 2,120 |
| examples over max_length | 0 |
| truncation rate | 0 |
| fixed sample rows | 100 |

固定样本分层：

| stratum | rows |
|---|---:|
| `2wiki:train-fit:twowiki_evidence_chain:answerable` | 20 |
| `2wiki:train-fit:unsupported_support_removed:unsupported` | 20 |
| `2wiki:train-modelval:twowiki_evidence_chain:answerable` | 20 |
| `niah:train-fit:niah_qa2d_single_claim:answerable` | 20 |
| `niah:train-modelval:niah_qa2d_single_claim:answerable` | 20 |

## 判定

G212R3 length gate 通过。固定样本已生成，但所有 rows 仍是 pending，所以 G212R3 不能解锁 G300。

下一步必须执行 G212M3 sample review/adjudication。

## 产物

Runtime:

```text
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G212R3-v1
```

Git-archived artifacts:

| 产物 | SHA256 |
|---|---|
| `artifacts/G212R3/prepare_manifest.json` | `5dd4d0cdd525fb59bd1563fabecba91a6d155a0ff6e9ea48db8efc5d80aa77d0` |
| `artifacts/G212R3/length_audit.json` | `9647c6301a8c4390ef6535c8420db1eab648ccab58b0667386093383be41ae0b` |
| `artifacts/G212R3/length_rows.jsonl` | `78890d262272d4a6ab1d1c11620db757752a9e945c9ebdacf5aa9ca4fc25ffe0` |
| `artifacts/G212R3/manual_sample_summary.json` | `f7ccf073c07a0486069ddb0720f3639b4ebaa5122443241c029bfa1939f192ac` |
| `artifacts/G212R3/manual_sample.jsonl` | `dbe12b9c7f240d4393b984195b0a20180971291ff249dd76fa070e2b5b89db29` |
| `artifacts/G212R3/G212R3_EXECUTION_AUDIT.json` | `583d6a68d8642354dee4379c0cf468a7e4aedd9295db2b623e0205b4656d5029` |
