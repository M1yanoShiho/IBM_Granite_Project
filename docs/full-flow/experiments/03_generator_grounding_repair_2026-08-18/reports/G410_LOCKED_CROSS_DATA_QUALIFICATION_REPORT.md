# G410 Locked Cross-Data Qualification Report

**Stage:** G410 formal 2Wiki cross-data qualification
**Date:** 2026-08-20
**Status:** `G410_CROSS_DATA_RESPONSIBILITY_PASS`
**Formal task subset:** false
**Held-out:** not read
**GQ freeze:** not performed

## What Ran

G410 used the locked `GR-C` Generator candidates from seeds 13/42/73 and the fixed G310 G0 baseline.

The formal 2Wiki cross-data task packet was prepared from G223 validation 2Wiki cases:

- 95 2Wiki case groups;
- 5 context variants per group;
- 475 generation tasks total;
- no reference/gold fields in the generation task packet;
- references kept in a separated packet and used only after generation for scoring.

Runtime root:

```text
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G410-v1
```

Formal runtime directories:

- `formal-prepare-739b155`
- `formal-seed13-739b155`
- `formal-seed42-739b155`
- `formal-seed73-739b155`
- `formal-score-739b155`

## Completion

All three seeds completed the full formal packet:

| Seed | Rows | Status | Runtime gold/reference | Held-out/sealed |
|---:|---:|---|---|---|
| 13 | 475 | COMPLETE | false | false |
| 42 | 475 | COMPLETE | false | false |
| 73 | 475 | COMPLETE | false | false |

No utility labels were started in this stage.

## Main Result

Formal scoring returned:

```text
G410_CROSS_DATA_RESPONSIBILITY_PASS
```

Relative to fixed G0, the family deltas were:

| Context | correct+cited | answer | coverage | citation precision | citation recall |
|---|---:|---:|---:|---:|---:|
| all_2wiki | +39.65pp | +6.95pp | +2.53pp | +31.37pp | +35.47pp |
| topk | +34.04pp | -1.75pp | -1.75pp | +26.32pp | +31.23pp |
| support_only | +38.60pp | +16.14pp | +2.81pp | +23.33pp | +27.19pp |
| support_first | +36.14pp | -3.86pp | -2.81pp | +28.95pp | +33.68pp |
| support_middle | +46.67pp | +14.39pp | +7.72pp | +36.49pp | +40.76pp |
| support_last | +42.81pp | +9.82pp | +6.67pp | +41.75pp | +44.50pp |

The gate recorded:

- `technical_pass=true`;
- `responsibility_pass=true`;
- `positive_signal=true`;
- no tripwires;
- `strong_claim=PENDING_G420_STATISTICS`;
- `citation_regression_guard=PENDING_G420_REVEALED_ASQA_QAMPARI_BOUNDARY_CONFIRMATION`.

## Interpretation

This is a strong cross-data signal: the Generator candidate that passed G400 NIAH qualification also improves 2Wiki citation-grounded answer quality over fixed G0 on the registered internal cross-data screen.

The result supports continuing to G420 combined Generator gate. It does not by itself freeze GQ, start utility label generation, authorize held-out evaluation, or justify changing thresholds, seeds, datasets, or final-test boundaries.

## Boundary Check

This stage did not:

- freeze a teacher Generator;
- run G420;
- generate utility labels;
- train or qualify a Selector;
- modify Retriever, Selector, seeds, gates, or scoring rules;
- read sealed600;
- read HotpotQA, MuSiQue-Full, RGB, or any SystemF held-out data.

## Archived Artifacts

Git-archived artifacts:

- `artifacts/G410/G410_FORMAL_PREPARE_MANIFEST.json`
- `artifacts/G410/G410_FORMAL_SEED13_RUN_MANIFEST.json`
- `artifacts/G410/G410_FORMAL_SEED42_RUN_MANIFEST.json`
- `artifacts/G410/G410_FORMAL_SEED73_RUN_MANIFEST.json`
- `artifacts/G410/G410_FORMAL_SCORE_REPORT.json`
- `artifacts/G410/G410_FORMAL_RUNTIME_SCORE_REPORT.md`
- `artifacts/G410/G410_FORMAL_QUALIFICATION_MANIFEST.json`

The next stage is G420 combined Generator gate/statistics. G420 must combine G400 NIAH and G410 2Wiki evidence, keep the existing frozen rules, and decide whether to freeze new `GR-C` as GQ or use G0 fallback.
