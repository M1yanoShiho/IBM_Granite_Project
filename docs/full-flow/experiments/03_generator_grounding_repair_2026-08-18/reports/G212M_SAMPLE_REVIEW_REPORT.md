# G212M Sample Review Report

**Status:** `NOT_FREEZE_READY_SAMPLE_REVIEW_FAILED`

G212M completed the fixed 100-row sample review/adjudication packet prepared by G212R. This stage did not train Generator, did not generate utility labels, and did not read sealed, held-out, or official dev data.

## Inputs

- G212R manual sample: `9434da2434ea10dad5e1fe7548ab54745e11820b4f325f5b1728bf0b4921bd58`
- G212R prepare manifest: `fa683cd0e25b1f190b9197a2743de630085f1bec748053aaa5594cda47a69485`
- G212R length audit: `0946f082118dbb5d8eb12cf2ed3ef3b8f1b51c76cc5f41b1aa303d372b1b7873`

## Review Rule

Each sampled row was checked for:

- cited evidence support;
- citation remap consistency;
- answer or yes/no chain preservation;
- unsupported rows saying `I don't know.` without visible removed support;
- no obvious target or prompt corruption.

Rows with self-contained support failures, ambiguous relation chains, or semantic QA2D mismatches were marked `FAIL`. This packet is intentionally recorded as sample review/adjudication, not as a completed human-review claim.

## Result

| Stratum | PASS | FAIL | UNCERTAIN |
|---|---:|---:|---:|
| 2wiki train answerable | 17 | 3 | 0 |
| 2wiki train unsupported | 20 | 0 | 0 |
| 2wiki model-val answerable | 17 | 3 | 0 |
| NIAH train answerable | 20 | 0 | 0 |
| NIAH model-val answerable | 18 | 2 | 0 |
| **Total** | **92** | **8** | **0** |

The freeze readiness manifest is therefore `NOT_FREEZE_READY_SAMPLE_REVIEW_FAILED`, and G300 remains locked.

## Failed Rows

| Sample | Case | Stratum | Reason |
|---:|---|---|---|
| 3 | `2wiki::f2ff9630084c11ebbd56ac1f6bf848b6` | 2wiki train answerable | Birthplace comparison chain not preserved. |
| 16 | `2wiki::3b46e32e0bb011ebab90acde48001122` | 2wiki train answerable | Paternal-grandfather chain uses an unanchored pronoun. |
| 19 | `2wiki::8c4e731309b311ebbdb0ac1f6bf848b6` | 2wiki train answerable | Mount Babel fact is not self-contained in the target. |
| 42 | `2wiki::37886d7208e311ebbda4ac1f6bf848b6` | 2wiki model-val answerable | Jolliff Spring Branch/country chain is not preserved. |
| 45 | `2wiki::6dee720a0bde11eba7f7acde48001122` | 2wiki model-val answerable | Husband relation starts with an unanchored pronoun. |
| 60 | `2wiki::a8cde8280bda11eba7f7acde48001122` | 2wiki model-val answerable | Father relation starts with an unanchored pronoun. |
| 95 | `niah-new-modelval::1024` | NIAH model-val answerable | Question asks author and topic; target preserves only topic. |
| 97 | `niah-new-modelval::10597` | NIAH model-val answerable | Question asks who the song was written for; target says it was written for `1973`. |

## Interpretation

This is not evidence that the route is meaningless. The signal is still positive: 92/100 sampled rows passed, unsupported rows were clean, and NIAH train rows passed. The failures are concentrated in target self-containment and a small number of NIAH model-val QA2D mismatches.

The correct next step is a controlled repair stage, not direct training:

1. repair or filter 2Wiki targets whose selected support sentence is not self-contained enough for the relation chain;
2. filter or regenerate NIAH model-val QA2D targets with clear semantic mismatches;
3. rerun structural, TRUE, length, and sample review gates before any G300 training.

## Artifacts

- Execution audit: `artifacts/G212M/G212M_EXECUTION_AUDIT.json`, SHA256 `48fbd78183abb6b1c64ba921452e6f6dcf9394486244ff6b37f2f58ae64c6aa9`
- Review rows: `artifacts/G212M/review/sample_review_rows.jsonl`, SHA256 `00f3f979f8ca0364226505da0dadb51f1a3d14c71c20724f88a9d0e803b91d6d`
- Review summary: `artifacts/G212M/review/sample_review_summary.json`, SHA256 `3893f2cdcbfa5923df6cc0b9fa089b8e523d57ecf29cf6d95d5026568c0326e8`
- Ordered review IDs: `artifacts/G212M/review/ordered_review_ids.json`, SHA256 `90b6914d280d12d020564a3d4fabec97fc03ba6a410c3a47557ecbbf4340eb0b`
- Freeze readiness: `artifacts/G212M/review/freeze_readiness_manifest.json`, SHA256 `49385a8800abb4f8b20c8543e90e9e68edb2f9479a83732d04e0113bfc6124b5`
