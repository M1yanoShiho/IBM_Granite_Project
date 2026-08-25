# I090 Fast-Path System Decision

**Date:** 2026-08-21
**Status:** `COMPLETE / FAST_PATH_SYSTEM_CANDIDATE_SELECTED`
**System candidate:** `SystemF-fast-S0-GQ-2026-08-21`
**Held-out:** not read; still requires separate user authorization
**S110:** continues in background as optional Utility Selector evidence

## Decision

The deadline-safe system candidate is:

```text
Retriever: frozen Hybrid RRF/default
Selector:  S0 TopK keep-all
Generator: frozen GQ = GR-C three-seed family
```

This is a valid `Retriever -> Selector -> Generator -> one answer` system because `S0 TopK` is the frozen keep-all Selector module. It is not a claim that Selector learning improved the system. The current evidence supports a Generator/GQ-centered method first.

## Why This Is the Right Fast Path

G420 already established that GQ improves over the fixed G0 baseline:

| Development scope | Main metric | Delta vs G0 | 95% CI |
|---|---|---:|---:|
| G400 NIAH `K_topk` | correct_and_cited | +6.13pp | [+2.63pp, +9.90pp] |
| G410 2Wiki `all_2wiki` | correct_and_cited | +39.65pp | [+29.97pp, +49.35pp] |

S090 then checked whether the old Legacy Selector adds useful same-GQ effect. On the matched 218 NIAH questions:

| Comparison | correct_and_cited | answer_match |
|---|---:|---:|
| TopK+GQ minus TopK+G0 | +9.33pp | +8.56pp |
| Legacy SL+GQ minus TopK+GQ | -0.46pp | -1.07pp |
| Legacy SL+GQ minus TopK+G0 | +8.87pp | +7.49pp |

So `Legacy SL+GQ` still benefits from GQ, but it does not show extra value over `TopK+GQ`. On 2Wiki, the old L003 Legacy Selector deleted 0 items in 1,000 questions, so it has no observed cross-data filtering contribution there.

## What This Means for the Project

The method is worth continuing. The positive signal is not weak: the Generator repair has already shown strong non-held-out development gains, especially on citation-grounded 2Wiki.

The part we should not overclaim is Selector learning. Under the deadline, the main project story should be:

- the Retriever is fixed and reports support visibility;
- the Selector module for the fast system is TopK keep-all;
- the main improvement comes from a grounded Generator GQ;
- Utility Selector remains an optional enhancement only if S110/S200/S300 later show same-GQ end-to-end benefit.

This avoids spending several days waiting for Utility Selector data before we can write the main result tables.

## Baseline Interpretation

For fast-path reporting:

| Arm | Meaning | Status |
|---|---|---|
| A = TopK+G0 | old/default Generator baseline | available from G400/G410 |
| B = TopK+GQ | Generator-repaired fast system | selected as `D` because `SQ=S0` |
| C = Legacy SL+GQ | old risk-filter baseline | diagnostic only; no positive same-GQ contribution |
| D = SQ+GQ | final fast-path candidate | equals B when `SQ=S0` |

The main comparison is therefore `B-A` / `D-A`. `D-B` is not applicable because there is no learned Selector contribution in the fast path.

## Boundary

I090 did not:

- change Retriever, Selector, Generator, seeds, thresholds, metrics, or data splits;
- start S200/SU training;
- stop S110;
- use sealed600;
- read HotpotQA, MuSiQue-Full, RGB, or any final held-out data.

Next action is to prepare/run only the remaining development/baseline packaging needed for the report. Final held-out H remains blocked until explicit user authorization.
