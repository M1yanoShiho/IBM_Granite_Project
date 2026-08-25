# S090 Early Full-Flow Triage

**Date:** 2026-08-21
**Status:** `COMPLETE / FAST_GQ_SYSTEM_READY`
**Held-out:** not read
**S110:** continues in background, not used for this decision

## Why This Stage Exists

The original S path made Utility Selector materialization and training the next blocking step. That is scientifically clean, but too slow for the 2026-09-05 deadline if the main method can already be decided from existing frozen evidence.

S090 therefore reuses already completed, frozen, non-held-out results to answer a narrower question:

> Is it worth waiting for a newly trained Utility Selector before starting full-system development, or is the current positive signal mainly from the frozen Generator GQ?

This stage does not change GQ, does not train or freeze a new Selector, does not read held-out data, and does not make a final held-out claim.

## Evidence Reused

- G400 formal NIAH scored rows: `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G400-v1/formal-score-5a79b75/scored_rows.jsonl`
- G400 formal score report: `artifacts/G400/G400_FORMAL_SCORE_REPORT.json`
- G410 formal 2Wiki score report: `artifacts/G410/G410_FORMAL_SCORE_REPORT.json`
- L003 frozen Legacy Selector traces: `/scratch/fl25387/IBM_Granite_Project_latest/runs/selector-lean-v3/L003/final/decision_trace_seed13.jsonl` and `decision_trace_seed42.jsonl`

The key re-aggregation is the matched 218-query NIAH subset where both `K_topk` and `S_legacy_selected` contexts exist in G400. For GQ, seeds 13/42/73 are averaged as the frozen GQ family; no best-seed selection is used.

## Main Result

On the matched 218 NIAH questions:

| Metric | A: TopK+G0 | B: TopK+GQ | C: Legacy SL+GQ | B-A | C-B | C-A |
|---|---:|---:|---:|---:|---:|---:|
| correct_and_cited | 47.71% | 57.03% | 56.57% | +9.33pp | -0.46pp | +8.87pp |
| answer_match | 60.09% | 68.65% | 67.58% | +8.56pp | -1.07pp | +7.49pp |
| coverage | 89.91% | 95.72% | 96.02% | +5.81pp | +0.31pp | +6.12pp |
| citation precision | 70.72% | 73.17% | 73.32% | +2.45pp | +0.15pp | +2.60pp |
| citation recall | 70.34% | 73.17% | 73.32% | +2.83pp | +0.15pp | +2.98pp |

For `correct_and_cited`, the transition counts are:

| Comparison | Wrong to right | Right to wrong | Net |
|---|---:|---:|---:|
| B-A | 46 | 25 | +21 |
| C-B | 2 | 3 | -1 |
| C-A | 45 | 25 | +20 |

The 2Wiki side already has a strong TopK+GQ signal from G410: `all_2wiki correct_and_cited` is +39.65pp versus fixed G0. The old L003 Legacy Selector deleted 0 evidence items on 1,000 2Wiki questions for both seed13 and seed42 traces, so it currently provides no observed 2Wiki filtering effect to test as an added selector contribution.

## Decision

S090 changes the execution priority:

1. Keep S110 running in the background because it may still produce a useful Utility Selector dataset.
2. Do not let S110/S200/SU block the deadline-critical main experiment path.
3. Promote the fastest defensible development candidate to `Retriever + S0 TopK + GQ`.
4. Treat Utility Selector as an optional enhancement: it can enter the final method only if it later shows positive same-GQ end-to-end effect over `TopK+GQ`.
5. Do not claim Legacy Selector or Utility Selector success from the current evidence.

In plain terms: the method is not failing. The positive signal is real, but it is mainly Generator-side at this point. The safe deadline plan is to push the GQ-based full system and baseline comparisons now, while the Utility Selector route continues only as time allows.

## Boundary

S090 did not:

- start or stop S110;
- train S200/SU;
- modify Retriever, Selector, GQ, seeds, thresholds, metrics, or data splits;
- use sealed600;
- read HotpotQA, MuSiQue-Full, RGB, or any final held-out data.

Next stage: start fast-path full-system development with `SQ=S0` unless S110/SU finishes early and proves a positive same-GQ Selector effect.
