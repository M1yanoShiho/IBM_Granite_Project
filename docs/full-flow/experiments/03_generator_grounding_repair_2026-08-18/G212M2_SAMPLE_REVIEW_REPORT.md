# G212M2 Sample Review Report

**Status:** `NOT_FREEZE_READY_SAMPLE_REVIEW_FAILED`

G212M2 completed sample review/adjudication on the new fixed 100-row G212R2 packet. This stage did not train Generator, did not generate utility labels, and did not read sealed, held-out, or official dev data.

## Inputs

- G212R2 sample SHA256: `d920e49067ab95dd4a288aebd9a08fdd51c89d5b0ec024978bb9cb804d2a3bd1`
- G212R2 prepare manifest SHA256: `36437e47b4fed13a9ca852e6c5e2f1e780c2d67f5caff8a82238cd1701b2a464`
- G212R2 length audit SHA256: `6d3b1c04e2b9069139b78788ac637598cb5ffec1e272862e465700b49bd22d00`

## Result

| Stratum | PASS | FAIL | UNCERTAIN |
|---|---:|---:|---:|
| 2Wiki train answerable | 16 | 4 | 0 |
| 2Wiki train unsupported | 20 | 0 | 0 |
| 2Wiki model-val answerable | 17 | 3 | 0 |
| NIAH train answerable | 20 | 0 | 0 |
| NIAH model-val answerable | 19 | 1 | 0 |
| **Total** | **92** | **8** | **0** |

Freeze readiness is therefore `NOT_FREEZE_READY_SAMPLE_REVIEW_FAILED`, and G300 remains locked.

## Failed Rows

| Sample | Case | Stratum | Reason |
|---:|---|---|---|
| 4 | `2wiki::5d4ad9d60baf11ebab90acde48001122` | 2Wiki train answerable | Father-in-law chain uses unanchored pronouns. |
| 13 | `2wiki::76af8d460bdb11eba7f7acde48001122` | 2Wiki train answerable | Father birthplace chain is ambiguous because Robert F. Kennedy is not anchored in the target sentence. |
| 15 | `2wiki::4c50f2f40bdd11eba7f7acde48001122` | 2Wiki train answerable | Director relation starts with unanchored `It` and omits the film title. |
| 17 | `2wiki::2413b227085211ebbd58ac1f6bf848b6` | 2Wiki train answerable | Same-country movie chain omits `The Blue 9`. |
| 44 | `2wiki::e163e5f9085f11ebbd5dac1f6bf848b6` | 2Wiki model-val answerable | Birthplace comparison lacks Alexander Nevsky birthplace evidence in the target. |
| 50 | `2wiki::013e9f1808c811ebbd91ac1f6bf848b6` | 2Wiki model-val answerable | First film title is truncated, corrupting the comparison target. |
| 57 | `2wiki::5f6a05d4096f11ebbdb0ac1f6bf848b6` | 2Wiki model-val answerable | Same-country chain lacks Cove name and country. |
| 94 | `niah-new-modelval::1015` | NIAH model-val answerable | QA2D target truncates `My Name Is Earl` to `my name`. |

## Interpretation

The route still has a positive signal: 92/100 rows passed again, unsupported rows were clean, and NIAH train rows were clean. However, the repeated 2Wiki self-containment failures after G216 show this is not just a small set of isolated bad cases. Continuing to remove only the newly sampled failures would risk wasting time and overfitting the sample.

The next stage should be a systematic target repair, not training and not another narrow sample-only removal:

1. make 2Wiki targets self-contained for relation chains, especially when support sentences start with pronouns, omit the entity title, or omit the relation-bearing attribute;
2. add stricter NIAH QA2D semantic filters for title/entity truncation;
3. rematerialize or filter under the existing data boundaries;
4. rerun structural/TRUE as needed, then length and sample review again.

## Artifacts

- Execution audit: `artifacts/G212M2/G212M2_EXECUTION_AUDIT.json`, SHA256 `6c6d47a9b76efa256c213601feb440c6b66618612eacb85682f898916954e07d`
- Review rows: `artifacts/G212M2/review/sample_review_rows.jsonl`, SHA256 `630f21acd3dfa2e89126bfb9dabf1a191a505b0b240ce0638ad11919bbc759ce`
- Review summary: `artifacts/G212M2/review/sample_review_summary.json`, SHA256 `ff46b678a27834a723e2f172bedaeaaa7444a1d0ab1aea4df5881661b366cdbf`
- Ordered review IDs: `artifacts/G212M2/review/ordered_review_ids.json`, SHA256 `de137df7d9675d8664450a5992386776ed3ecd0c256384b2cb1a00409125683e`
- Freeze readiness: `artifacts/G212M2/review/freeze_readiness_manifest.json`, SHA256 `091715a899c39121442014d698b4486314c4c1c19455baad56d640e4275c1056`
