# Current State And Artifact Index

**Date:** 2026-08-21

**Status:** `CURRENT_STATE_INDEX_COMPLETE / THREE_MODULE_PLAN_ADDED`

This file is the first place to read before continuing the project.  It separates the current usable route from paused work, historical evidence, server runtime outputs, and the cleaned local workspace.

## One-Sentence Current State

The historical deadline-prioritized freeze is `SystemF-fast-S0-GQ-2026-08-21`.  After that freeze, the user directed the next system candidate to use the already-frozen Lean harm/protect Selector instead of `S0`; that three-module candidate is now connected and has passed a one-query wiring smoke, but it has not yet run the overall development experiment or any held-out evaluation.

## Three-Module Wiring Update

The new wiring-only candidate is:

```text
frozen Hybrid RRF Retriever
-> frozen NLI harm/protect Selector (seed 13, q=.99, cap=2)
-> frozen GQ GR-C Generator (seed-13 smoke member) + frozen TRUE verifier
-> one answer
```

The single configuration entry point is `configs/experiments/systemf_three_module_smoke_seed13.toml`.  It binds the frozen Selector checkpoint, Generator adapter, Granite base snapshot and TRUE snapshot to their recorded SHA-256 values before loading them.

The smoke used one synthetic development-only question and ten fixture documents.  All three real server modules loaded in one process; the Retriever produced ten candidates, the Selector emitted ten live harm/protect score rows, and the Generator returned `IBM acquired Red Hat in 2019.` with a verified citation.  The Selector deleted zero rows on this particular fixture, which is a valid conservative-policy outcome rather than a skipped module.

This smoke establishes wiring only.  It is not evidence of overall quality improvement, does not supersede the historical I220 results, and does not authorize reading or scoring HotpotQA, MuSiQue-Full, RGB, or any other held-out data.  The separate overall experiment design is now frozen in `../04_frozen_three_module_system_evaluation_2026-08-21/PLAN.md`; no held-out run has started.

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
| Detailed stage reports | `reports/` |
| Frozen fast-path system report | `reports/I220_SYSTEMF_FAST_FREEZE_REPORT.md` |
| Frozen fast-path system manifest | `artifacts/I220/systemf_fast_s0_gq_freeze_manifest.json` |
| Current three-module final experiment plan | `../04_frozen_three_module_system_evaluation_2026-08-21/PLAN.md` |

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
| `143bc1d` | I225 | current state/artifact index written |

Local, GitHub, and server repo were synchronized through `143bc1d` before the workspace cleanup update.

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

- `reports/S100_FORMAL_UTILITY_PILOT_REPORT.md`
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

## Local Workspace Cleanup

After user clarification, presentation materials are treated as team records and must be retained.  The cleanup scope is limited to experiment-side scratch files and route bookkeeping:

- tracked presentation files under `docs/presentations/` were restored and retained;
- untracked experiment scratch files were removed from the local repo workspace;
- route-critical lightweight evidence remains in this experiment directory and its `artifacts/` manifests;
- server runtime outputs were not deleted or copied into Git.

The machine-readable cleanup record is `artifacts/I226/workspace_cleanup_manifest.json`.

The experiment root is now intentionally small:

- `README.md`
- `PLAN.md`
- `TRACKER.md`
- `CURRENT_STATE_AND_ARTIFACT_INDEX_2026-08-21.md`

Detailed stage reports live in `reports/`.  Machine-readable artifacts stay in `artifacts/`.

## Recommended Next Action

The datasets, four baselines plus Ours, three ablations, statistics, runtime budget and artifact layout are now fixed in experiment 04 v4. The roadmap is split into five separately authorized goals. Only Goal 1 is currently open: prepare per-query data isolation and the scorer on synthetic or already-revealed development fixtures. Do not connect baselines, run the 10-arm smoke check, generate or score held-out answers, or begin Goal 2 until Goal 1 has passed and the user separately starts the next goal.

Do not restart S110 and do not retrain or retune Selector unless the user explicitly opens a new route.
