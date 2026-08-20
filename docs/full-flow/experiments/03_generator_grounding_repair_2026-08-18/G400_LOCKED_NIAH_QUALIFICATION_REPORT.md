# G400 Locked NIAH Qualification Report

**Stage:** G400 formal locked NIAH qualification
**Date:** 2026-08-20
**Status:** `G400_NIAH_RESPONSIBILITY_PASS`
**Candidate:** frozen `GR-C` seeds 13/42/73
**Held-out:** not read

## Result

G400 formal generation and scoring completed for all three locked `GR-C` seeds.

This stage gives a positive Generator qualification signal on the constructed NIAH full/stress set. It does not freeze `GQ`, does not start Selector utility labels, and does not authorize held-out evaluation. Cross-data G410 and statistical G420 checks are still pending.

## Runtime Completion

Runtime root:

```text
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G400-v1
```

| Seed | Runtime dir | Rows | Status | Errors | Missing trace | Runtime gold/reference |
|---:|---|---:|---|---:|---:|---|
| 13 | `formal-seed13-5a79b75` | 2873 | COMPLETE | 0 | 0 | false |
| 42 | `formal-seed42-5a79b75` | 2873 | COMPLETE | 0 | 0 | false |
| 73 | `formal-seed73-5a79b75` | 2873 | COMPLETE | 0 | 0 | false |

All three run manifests report `sealed_or_heldout_read=false`, `gold_loaded_at_runtime=false`, `reference_answers_loaded_at_runtime=false`, and `utility_labels_started=false`.

## Scoring Scope

Formal scoring covered 2,873 tasks:

| Task group | Count |
|---|---:|
| NIAH answerable full/stress tasks | 1829 |
| Unsupported safety tasks | 1044 |
| Total | 2873 |

Scoring uses NIAH dev gold only after generation. The `run` command has no gold/reference argument.

## Family Deltas vs Fixed G0

Positive values mean the three-seed `GR-C` family improved over fixed G0 on the same inputs.

| Context | correct+cited | answer | coverage | citation precision | citation recall | unsupported assertion |
|---|---:|---:|---:|---:|---:|---:|
| K_topk | +6.13pp | +6.27pp | +7.13pp | +1.27pp | +1.98pp | 0.00pp |
| S_legacy_selected | +10.70pp | +8.87pp | +7.95pp | +5.12pp | +5.28pp | 0.00pp |
| O_support_only | +8.10pp | +8.56pp | +5.20pp | +2.75pp | +2.98pp | 0.00pp |
| OB_support_benign | +5.35pp | +8.26pp | +5.35pp | +0.54pp | +1.15pp | 0.00pp |
| OH_support_harmful | +11.93pp | +11.77pp | +7.19pp | +8.03pp | +8.56pp | 0.00pp |
| OP_support_last | +10.86pp | +14.83pp | +6.12pp | +2.83pp | +3.29pp | 0.00pp |
| train_unsupported_safety | 0.00pp | 0.00pp | -29.95pp | 0.00pp | 0.00pp | -29.95pp |

The unsupported safety row is directionally positive for safety: unsupported ungrounded assertions dropped from 30.08% under G0 to near zero across the three candidates.

## Gate

G400 technical checks passed:

- runtime error = 0;
- missing trace = 0;
- invalid citation index = 0 in the scored report;
- no runtime gold/reference read by generation;
- no sealed or held-out input read;
- no utility labels started.

G400 responsibility checks passed:

- NIAH `correct_and_cited` delta > 0;
- answer, coverage, citation precision, and citation recall non-inferiority checks passed;
- unsupported assertion delta did not increase;
- at least 2/3 seeds had nonnegative `correct_and_cited` direction;
- no seed-level citation precision or recall drop exceeded the tripwire.

No G400 tripwire fired.

## Boundary

G400 is still a constructed NIAH qualification stage. It is important evidence against the earlier NIAH regression risk, but it is not the final Generator freeze.

This stage did not:

- run G410 cross-data qualification;
- run G420 statistical gate;
- freeze `GQ`;
- generate Selector utility labels;
- modify Retriever or Selector;
- read sealed600 or system held-out datasets.

## Next Stage

G400 unlocks G410 cross-data qualification under the existing plan. G410 must test the same frozen `GR-C` family on the registered cross-data checks before G420 decides whether to freeze a new Generator or fall back to G0.
