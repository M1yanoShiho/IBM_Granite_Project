# I200 Fast-Path Development Comparison Package

**Date:** 2026-08-21

**Status:** `COMPLETE / FAST_PATH_DEV_COMPARISON_PACKAGED`

**System candidate:** `SystemF-fast-S0-GQ-2026-08-21`

## Plain-Language Answer

Yes, there is a positive signal worth continuing.  The current evidence says the useful improvement is mainly from the new Generator family GQ, not from retraining or waiting for a new Selector.

The fast-path system is therefore:

```text
frozen Retriever -> S0 TopK keep-all Selector -> frozen GQ Generator -> one answer
```

This lets the project move to full development reporting and baseline comparison now, while S110/SQ remains a background optional enhancement.

## Locked Arms

| Arm | Selector | Generator | Role |
|---|---|---|---|
| A | S0 TopK keep-all | G0 | previous/default baseline |
| B | S0 TopK keep-all | GQ | Generator-only fast-path comparison |
| C | Legacy SL | GQ | diagnostic risk/safety baseline |
| D | S0 TopK keep-all | GQ | current SystemF-fast candidate |

Because `SQ=S0` for the fast path, `D = B`.  I200 therefore supports the main comparison `D-A = TopK+GQ - TopK+G0`.  It does not claim a learned Selector gain.

## Main Development Results

| Development evidence | Queries | Metric | A: TopK+G0 | D/B: TopK+GQ | Delta | 95% CI for Delta |
|---|---:|---|---:|---:|---:|---:|
| G400 NIAH TopK | 739 | correct_and_cited | 47.90% | 54.04% | +6.13pp | [+2.63, +9.90] |
| G400 NIAH TopK | 739 | answer_match | 64.01% | 70.28% | +6.27pp | [+2.86, +9.92] |
| G400 NIAH TopK | 739 | coverage | 87.96% | 95.08% | +7.13pp | [+4.73, +9.66] |
| G410 2Wiki all_2wiki | 95 | correct_and_cited | 15.16% | 54.81% | +39.65pp | [+29.97, +49.35] |
| G410 2Wiki all_2wiki | 95 | answer_match | 48.63% | 55.58% | +6.95pp | [-2.34, +16.15] |
| G410 2Wiki all_2wiki | 95 | coverage | 89.89% | 92.42% | +2.53pp | [-1.72, +6.74] |
| G400 unsupported safety | 1044 | unsupported_ungrounded_assertion | 30.08% | 0.13% | -29.95pp | [-32.73, -27.11] |

The strongest paper-facing signal is `correct_and_cited`: it improves on both NIAH and 2Wiki, with CI lower bounds above zero.  The safety signal is also strong: unsupported ungrounded assertions drop sharply.

## Selector Diagnostic

On the 218 matched NIAH subset used for the fast triage:

| Comparison | correct_and_cited | answer_match | coverage | Interpretation |
|---|---:|---:|---:|---|
| B - A: TopK+GQ vs TopK+G0 | +9.33pp | +8.56pp | +5.81pp | positive Generator signal |
| C - B: Legacy SL+GQ vs TopK+GQ | -0.46pp | -1.07pp | +0.31pp | no useful same-GQ Selector gain |
| C - A: Legacy SL+GQ vs TopK+G0 | +8.87pp | +7.49pp | +6.12pp | mostly inherits GQ improvement |

This is why waiting several days for Utility Selector materialization/training is not the right main route under the deadline.  It remains useful as an optional enhancement, not as a blocker.

## I200 Decision

`I200_FAST_PATH_DEV_COMPARISON_PASS`

The current system candidate can proceed to I210 fast-path responsibility gate:

- D-A `correct_and_cited` is positive on both available development checks.
- NIAH main CI lower bound is above zero.
- 2Wiki `correct_and_cited` CI lower bound is above zero.
- Unsupported ungrounded assertions decrease strongly.
- There is no held-out read and no boundary change.

## Limitations

- This is still development/qualification evidence, not final held-out evidence.
- D equals B because `SQ=S0`; therefore no learned Selector contribution should be claimed.
- 2Wiki is an internal screen from G223/G410, not HotpotQA/MuSiQue/RGB held-out.
- Final held-out H still requires separate user authorization after SystemF freeze.

