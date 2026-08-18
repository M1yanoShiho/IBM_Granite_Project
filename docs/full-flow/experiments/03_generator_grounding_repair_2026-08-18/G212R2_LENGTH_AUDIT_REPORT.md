# G212R2 Length Audit Report

**Status:** `LENGTH_PASS_SAMPLE_REVIEW_PENDING`

G212R2 reran the length/truncation audit and fixed sample packet preparation on the G216 revised bundle. This stage did not train Generator, did not generate utility labels, and did not read sealed, held-out, or official dev data.

## Inputs

- G216 manifest SHA256: `0782877816719e13f2caa1ce53fd38be38c2e2d18756df91ba9095b0a0301df0`
- G216 train cases SHA256: `a8d0cd236b912fefd42cc2f06e43ccc71ed31e6349a2d84642c76784ffb309d7`
- G216 validation cases SHA256: `fdbb9b4003dcfa2984ed42b91cdab55db47bd2dae9e7c8ee89490672359f4c6d`
- Granite tokenizer snapshot: `/scratch/fl25387/IBM_Granite_Project_latest/model-cache/models--ibm-granite--granite-4.1-3b/snapshots/c0650403e44e78ec0262dab1c90914c65b196c4e`

## Length Result

| Metric | Value |
|---|---:|
| Cases | 2,704 |
| Examples | 11,295 |
| Max length | 2,304 |
| Observed max | 2,120 |
| Over max length | 0 |
| Truncation rate | 0.0 |
| p50 | 1,345 |
| p90 | 1,841 |
| p95 | 1,882 |
| p99 | 1,966 |

By stratum:

| Stratum | Examples | p95 | Max | Over max |
|---|---:|---:|---:|---:|
| 2Wiki train answerable | 4,120 | 1,645 | 2,012 | 0 |
| 2Wiki train unsupported | 1,049 | 1,359 | 1,667 | 0 |
| 2Wiki model-val answerable | 515 | 1,619 | 1,943 | 0 |
| NIAH train answerable | 4,120 | 1,906 | 2,120 | 0 |
| NIAH model-val answerable | 1,491 | 1,918 | 2,030 | 0 |

## Fixed Sample Packet

G212R2 prepared a new fixed 100-row sample:

- 20 rows from 2Wiki train answerable;
- 20 rows from 2Wiki train unsupported;
- 20 rows from 2Wiki model-val answerable;
- 20 rows from NIAH train answerable;
- 20 rows from NIAH model-val answerable.

All rows are still `PENDING`; this report does not unlock G300.

## Artifacts

- Execution audit: `artifacts/G212R2/G212R2_EXECUTION_AUDIT.json`, SHA256 `f28264408d39cdda98925e200f2a458a4a629c769fd7078c556b3109b794021c`
- Prepare manifest: `artifacts/G212R2/prepare/prepare_manifest.json`, SHA256 `36437e47b4fed13a9ca852e6c5e2f1e780c2d67f5caff8a82238cd1701b2a464`
- Length audit: `artifacts/G212R2/prepare/length_audit.json`, SHA256 `6d3b1c04e2b9069139b78788ac637598cb5ffec1e272862e465700b49bd22d00`
- Length rows: `artifacts/G212R2/prepare/length_rows.jsonl`, SHA256 `41391b163eb2d18c5a7ee8e5a7bd1532ce8a2c81f12fa2d2667b80ee9bf502b7`
- Sample summary: `artifacts/G212R2/prepare/manual_sample_summary.json`, SHA256 `33e291adcd223a856af50de199a0b7c6305e9f8fecb28afe7de11e08302f5a51`
- Sample rows: `artifacts/G212R2/prepare/manual_sample.jsonl`, SHA256 `d920e49067ab95dd4a288aebd9a08fdd51c89d5b0ec024978bb9cb804d2a3bd1`

## Next Step

Run G212M2 sample review/adjudication on the new fixed sample. Only if it passes can the data be frozen for G300.
