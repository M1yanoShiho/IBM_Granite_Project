# Current State And Artifact Index

**Date:** 2026-08-21

**Status:** `CURRENT_STATE_INDEX_COMPLETE`

This file is the first place to read before continuing the project.  It separates the current usable route from paused work, historical evidence, and unrelated dirty files.

## One-Sentence Current State

The current frozen candidate is `SystemF-fast-S0-GQ-2026-08-21`: frozen Hybrid RRF Retriever + `S0` TopK keep-all Selector + frozen GQ Generator.  Final held-out H has not been run and still needs separate user authorization.

## What Is Paused

S110 utility materialization on the teacher server has been paused by user request.

| Item | State |
|---|---|
| Runtime root | `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/S110-v1/formal-1a54bbe` |
| Pause marker | `S110_PAUSED_BY_USER_REQUEST.json` |
| Screen status after pause | no active S110 screens |
| GPU status after pause | both RTX A4000 GPUs released |
| Preserved rows | seed13: 1452; seed42: 1230; seed73: not started |
| Resume rule | do not resume unless the user explicitly approves a revised plan |

## Files To Read First

These are the current source-of-truth files:

| Purpose | File |
|---|---|
| Human-readable route overview | `README.md` |
| Full plan and boundaries | `PLAN.md` |
| Stage tracker | `TRACKER.md` |
| Current state index | `CURRENT_STATE_AND_ARTIFACT_INDEX_2026-08-21.md` |
| Frozen fast-path system report | `I220_SYSTEMF_FAST_FREEZE_REPORT.md` |
| Frozen fast-path system manifest | `artifacts/I220/systemf_fast_s0_gq_freeze_manifest.json` |

## Current Main Results

These are development/qualification results, not held-out results.

| Evidence | Main comparison | Result |
|---|---|---:|
| G400 NIAH TopK | `TopK+GQ - TopK+G0` correct_and_cited | +6.13pp |
| G410 2Wiki internal screen | `TopK+GQ - TopK+G0` correct_and_cited | +39.65pp |
| G400 unsupported safety | unsupported ungrounded assertion | 30.08% -> 0.13% |
| S090 matched NIAH diagnostic | `Legacy SL+GQ - TopK+GQ` correct_and_cited | -0.46pp |

Interpretation:

- GQ has a positive end-to-end signal under frozen development inputs.
- Existing Selector filtering ability should not be described as useless.
- The precise limitation is: learned/Legacy Selector end-to-end gain over `TopK+GQ` has not been established yet.
- Therefore Selector work is optional/diagnostic until we decide whether it is needed after a clean full-flow review.

## Current Commit Chain

| Commit | Stage | Meaning |
|---|---|---|
| `9d6991d` | S090 | early full-flow triage |
| `b12d75e` | I090 | fast-path system candidate selected |
| `85dda3d` | I100 | fast-path input reuse accepted |
| `a083b6d` | I200 | development comparison packaged |
| `a05983d` | I210 | responsibility gate passed |
| `cae3702` | I220 | SystemF-fast frozen |

Local, GitHub, and server repo were synchronized through `cae3702` before this index was written.

## Canonical Evidence Buckets

### Main Path / Keep Close

These files are directly relevant to the next discussion and next run decision:

- `artifacts/G400/G400_FORMAL_SCORE_REPORT.json`
- `artifacts/G410/G410_FORMAL_SCORE_REPORT.json`
- `artifacts/G420/G420_FORMAL_SCORE_REPORT.json`
- `artifacts/G430/G430_GQ_FREEZE_MANIFEST.json`
- `artifacts/S090/early_full_flow_triage_manifest.json`
- `artifacts/I090/fast_path_system_decision_manifest.json`
- `artifacts/I100/fast_path_input_reuse_manifest.json`
- `artifacts/I200/fast_path_dev_comparison_manifest.json`
- `artifacts/I210/fast_path_responsibility_gate_manifest.json`
- `artifacts/I220/systemf_fast_s0_gq_freeze_manifest.json`

### Useful But Not Blocking

These are useful for diagnosing Selector, data quality, or Generator history, but they should not block the next planning discussion:

- `S100_FORMAL_UTILITY_PILOT_REPORT.md`
- `artifacts/S100/`
- paused S110 partial runtime under `/scratch/.../S110-v1/formal-1a54bbe`
- G300/G310/G320/G330 training implementation and adapter evidence
- L003/Legacy Selector traces referenced by S090

### Historical / Superseded

These files explain how the route got here, but they should not be read first when deciding the next experiment:

- old G200/G200R/G200R2 materialization reports;
- failed or superseded audit/repair reports from G210 through G223;
- G219 failure and G220/G221 repair attempts;
- older F001/F004/F005/F006 baseline/diagnostic runtime directories.

They are not garbage in the audit sense; they are historical evidence.  They are just not the current route.

## Server Runtime Data To Keep Separate

Runtime outputs live under `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow`.  They should not be copied wholesale into the Git repository.

Largest observed runtime directories:

| Runtime directory | Approx size | Current role |
|---|---:|---|
| `G220-v1` | 688M | historical/superseded data repair route |
| `F006` | 285M | historical diagnostic/baseline route |
| `G310-v1` | 257M | Generator screen/adapters; keep for GQ provenance |
| `S110-v1` | 170M | paused optional Selector utility route |
| `G330-v1` | 138M | GQ adapter provenance; keep |
| `G218-v1` / `G219-v1` | 133M / 130M | historical repair/failure evidence |

No server runtime data was deleted during this cleanup index step.

## Local Dirty Files Outside This Route

The local worktree still contains unrelated dirty/untracked files.  They were not staged or committed for this route:

- deleted presentation files under `docs/presentations/`;
- untracked presentation folders under `docs/presentations/2026-*`;
- untracked `.aris/traces/novelty-check/`;
- untracked `docs/selector/SELECTOR_BALANCE_REVIEW_2026-08-10.md`;
- untracked `findings.md`, `progress.md`, `task_plan.md`, `$STATUS`.

These need a separate cleanup decision.  They should not be mixed into full-flow experiment commits.

## Recommended Next Discussion

Before any new experiment is launched, decide the final development plan in this order:

1. Define the exact full-flow comparison we still need on development data.
2. Decide whether existing Legacy Selector should be tested once more with frozen GQ as a diagnostic, or whether I200 already covers enough.
3. Decide the final held-out authorization package: datasets, baselines, ablations, expected runtime, and stop rules.
4. Only after that, launch H100 or any extra development run.

Do not restart S110 unless the discussion concludes that Selector retraining is necessary and worth the time.

