# Experiment 04 Goal 4 — Module ablation results

**Decision:** `PASS`  
**Formal attempt:** `attempt-20260822a`  
**Completed:** 2026-08-22

## Outcome

Goal 4 is technically valid and complete. The three fresh dataset jobs finished `COMPLETED
0:0`, produced exactly 3,300 new answers, and reused the 1,100 frozen Goal 3 Full seed13
rows without regenerating Full. All three generation manifests and all three score manifests
are `PASS`.

The scientific result does not show a positive RAR contribution from the current GR-C
seed13 Generator. Replacing it with the frozen Direct Generator increases RAR on HotpotQA
and MuSiQue; the Retriever and Selector substitutions leave RAR unchanged on all three
datasets. This is a result of the frozen experiment, not a reason to change the method after
held-out evaluation.

## Table 2

Values are percentages.

| Dataset | Configuration | Ret. | Sel. | Ans. | Cit. | RAR |
|---|---|---:|---:|---:|---:|---:|
| HotpotQA | Full | 100.00 | 99.88 | 13.45 | 63.17 | 0.25 |
| HotpotQA | w/ Dense Retriever | 100.00 | 99.88 | 13.92 | 66.47 | 0.25 |
| HotpotQA | w/ Top-10 | 100.00 | 100.00 | 13.53 | 63.17 | 0.25 |
| HotpotQA | w/ Direct Generator | 100.00 | 99.88 | 25.76 | 45.29 | 3.50 |
| MuSiQue answerable | Full | 84.92 | 84.92 | 6.60 | 41.53 | 0.00 |
| MuSiQue answerable | w/ Dense Retriever | 87.98 | 87.90 | 7.05 | 41.34 | 0.00 |
| MuSiQue answerable | w/ Top-10 | 84.92 | 84.92 | 6.60 | 41.53 | 0.00 |
| MuSiQue answerable | w/ Direct Generator | 84.92 | 84.92 | 15.81 | 27.83 | 1.00 |
| RGB noise | Full | 47.85 | 46.25 | 0.00 | 90.56 | 0.00 |
| RGB noise | w/ Dense Retriever | 46.67 | 46.22 | 0.00 | 89.47 | 0.00 |
| RGB noise | w/ Top-10 | 47.85 | 46.27 | 0.00 | 90.56 | 0.00 |
| RGB noise | w/ Direct Generator | 47.85 | 46.25 | 0.67 | 79.11 | 0.67 |

## Paired RAR inference

Differences are `Full − ablation`, in percentage points. CIs use the frozen paired
component-cluster percentile bootstrap with 10,000 resamples and seed 13.

| Dataset | Ablation | ΔRAR | 95% CI |
|---|---|---:|---:|
| HotpotQA | w/ Dense Retriever | 0.00 | [0.00, 0.00] |
| HotpotQA | w/ Top-10 | 0.00 | [0.00, 0.00] |
| HotpotQA | w/ Direct Generator | -3.25 | [-5.25, -1.50] |
| MuSiQue answerable | w/ Dense Retriever | 0.00 | [0.00, 0.00] |
| MuSiQue answerable | w/ Top-10 | 0.00 | [0.00, 0.00] |
| MuSiQue answerable | w/ Direct Generator | -1.00 | [-2.00, -0.25] |
| RGB noise | w/ Dense Retriever | 0.00 | [0.00, 0.00] |
| RGB noise | w/ Top-10 | 0.00 | [0.00, 0.00] |
| RGB noise | w/ Direct Generator | -0.67 | [-1.67, 0.00] |

On HotpotQA and MuSiQue, the negative CIs exclude zero: Full has lower RAR than the
Direct Generator ablation. RGB has the same direction but its interval includes zero.
Higher Full citation scores therefore do not compensate for its lower answer performance
under the joint RAR criterion.

## Isolation and run validity

- Runtime inputs contained no scorer-only gold fields. The scorer sidecar path was removed
  from the generation environment and supplied only after the three new generation files
  were complete and hash-frozen.
- Goal 3 `prepared.jsonl`, Full generation, Full score, runtime hash, ordered IDs, and model
  manifest fingerprints were verified before every dataset job.
- `Full` had no generation command in the Goal 4 runner and `full_regenerated=false` is
  asserted by generation, score, summary, and final audit artifacts.
- Every non-Full row passed the single-module-substitution validator. There was no
  cross-job or cross-attempt mixing.
- One HotpotQA Dense-Retriever generation failure remains in the common denominator
  (`1/400 = 0.25%`); every other dataset-arm cell has zero runtime errors. All cells satisfy
  the preregistered `≤1%` validity guard.

## Verification evidence

- 52 Experiment 04 regression tests: `PASS`.
- Goal 4 Ruff, mypy contract check, and Slurm shell syntax: `PASS`.
- Independent audit: 4,400 rows, 4,400 unique dataset-arm-query keys, 12 aggregate cells,
  nine bootstrap cells, metric bounds, aggregate recomputation, manifest contracts, and
  artifact hashes: `PASS`.
- All three Slurm jobs: `COMPLETED 0:0`.

The release-facing machine evidence is in
[`final_audit.json`](../../results/experiment04/final_audit.json),
[`table2.json`](../../results/experiment04/table2.json), and
[`final_results.json`](../../results/experiment04/final_results.json), whose `paired_bootstrap.goal4`
field contains the frozen intervals. The original Goal 4 pass manifest and standalone bootstrap
file remain recoverable from `research-archive-2026-08-25`.

## Handoff

Goal 4 is `PASS`. Goal 5 may start automatically for the frozen final statistics and
cross-artifact audit. No method change, new generation, or held-out-driven tuning is
authorized by this handoff.
