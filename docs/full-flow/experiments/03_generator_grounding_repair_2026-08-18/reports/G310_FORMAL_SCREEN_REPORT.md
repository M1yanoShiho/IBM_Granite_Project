# G310 Formal Seed13 Screen Report

**Date:** 2026-08-19
**Status:** `COMPLETE / RECIPE_SELECTED`
**Selected recipe:** `GR-C`

## Decision

G310 completed the predeclared seed13 screen over the same fixed G223 controlled-continuation input, with the same frozen Retriever context, frozen Granite base splitter, frozen TRUE runtime routing, and MiniCheck post-generation citation scoring.

The selected recipe is `GR-C`. `GR-C` survived all G310 screen exclusions and had the best maximin `correct_and_cited` delta across the NIAH and 2Wiki validation slices. `GR-F` was excluded by the predeclared answer regression rule.

This is a model-validation recipe screen. It is not final Generator qualification, not a clean data freeze claim, and not a held-out result.

## Formal Runs

Runtime root:

```text
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G310-v1
```

Training jobs:

| Run | Status | Recipe | Seed | Train groups | Examples | Steps | Mean group loss | Validation group loss | Reload |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `formal-grf-seed13` | COMPLETE | `gr-f` | 13 | 2370 | 9207 | 297 | 0.076403 | 0.254666 | PASS |
| `formal-grc-seed13` | COMPLETE | `gr-c` | 13 | 2370 | 9207 | 297 | 0.044435 | 0.258522 | PASS |

Training boundaries:

- `dev_read=false`
- `sealed_or_heldout_read=false`
- `utility_labels_started=false`
- `clean_freeze_ready=false`
- G223 input status was `CONTROLLED_CONTINUATION_READY`, with 2Wiki model-val screen size 95.

## Screen Runtime

Formal screen output:

```text
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G310-v1/screen-formal-f79697c
```

The run produced 2968 generation rows and 2968 scored rows.

| Artifact | SHA256 |
|---|---|
| `run_spec.json` | `6e8b0e93fba7280e593bdc6601a702a9997037c83b0f058a7f83ba176ce12acf` |
| `generations.jsonl` | `a640f4c044548f16b5200eed4a80c3c8da1f7cd7003e2d69cdb31a09ac034aa4` |
| `run_manifest.json` | `d8890c971e065df38e9d1e77884533c1165c2fa2e9d78751efd659fe6a7cb15f` |
| `score_report.json` | `e76c5b0e2b55653269dcad50e09c20f8acec28760d0aecc8fd24c603cf03f5b8` |
| `scored_rows.jsonl` | `610ca878ee367855c3d229a8092a2c35d545c6f15be72b4363aa9639fc6a9996` |
| `SCREEN_REPORT.md` server runtime | `7e05c78a7fc6bc51bff0d08a4204f10bcfce34ba69760cd0f9739eb9816c4a9b` |
| `artifacts/G310/G310_FORMAL_SCREEN_RUNTIME_REPORT.md` archived copy | `febcbed0649c629581e6055e0e6f5ab08d8da879560dd50fcfabdbbb8473c496` |

Small audit copies are archived in `artifacts/G310/`. The large per-question generation and scored rows remain on the server runtime path above and are identified by SHA256.

## Aggregate Metrics

`correct_and_cited` is answer match plus complete MiniCheck-supported sentence citations.

| Slice | Arm | Groups | Answer | Coverage | Correct and cited | MiniCheck precision | MiniCheck recall |
|---|---|---:|---:|---:|---:|---:|---:|
| validation answerable | G0 | 302 | 0.6260 | 0.9100 | 0.4725 | 0.7358 | 0.7207 |
| validation answerable | GR-F | 302 | 0.5954 | 0.9380 | 0.5490 | 0.8409 | 0.8409 |
| validation answerable | GR-C | 302 | 0.6188 | 0.9594 | 0.5674 | 0.8499 | 0.8493 |
| NIAH validation | G0 | 207 | 0.6901 | 0.9151 | 0.6197 | 0.8015 | 0.7995 |
| NIAH validation | GR-F | 207 | 0.6135 | 0.9434 | 0.5459 | 0.8099 | 0.8099 |
| NIAH validation | GR-C | 207 | 0.6391 | 0.9620 | 0.5659 | 0.8133 | 0.8133 |
| 2Wiki validation | G0 | 95 | 0.4863 | 0.8989 | 0.1516 | 0.5926 | 0.5491 |
| 2Wiki validation | GR-F | 95 | 0.5558 | 0.9263 | 0.5558 | 0.9084 | 0.9084 |
| 2Wiki validation | GR-C | 95 | 0.5747 | 0.9537 | 0.5705 | 0.9295 | 0.9277 |
| train unsupported safety | G0 | 1044 | 0.0000 | 0.3008 | 0.0000 | 0.0000 | 0.0000 |
| train unsupported safety | GR-F | 1044 | 0.0000 | 0.0019 | 0.0000 | 0.0000 | 0.0000 |
| train unsupported safety | GR-C | 1044 | 0.0000 | 0.0010 | 0.0000 | 0.0000 | 0.0000 |

Decision deltas:

| Recipe | Survives | Overall `correct_and_cited` delta | NIAH delta | 2Wiki delta | Failures |
|---|---|---:|---:|---:|---|
| GR-C | yes | +0.0949 | -0.0538 | +0.4189 | none |
| GR-F | no | +0.0765 | -0.0738 | +0.4042 | `answer_regression_gt_2pp` |

## Boundary Check

- Runtime generation did not read gold/reference answers.
- References were used only after generation for scoring.
- TRUE was used for runtime routing, not as the citation judge.
- MiniCheck was used only after generation; unique judge calls were 2028.
- No held-out, sealed final data, or Selector utility labels were read or generated.
- Server log scan found no traceback, OOM, killed process, or error keyword.

## Result

`G310 COMPLETE / RECIPE_SELECTED`: use `GR-C` as the only recipe candidate for G320 recipe freeze.

This result unlocks G320 only. It does not freeze a teacher Generator, does not start utility label generation, and does not authorize held-out evaluation.
