# I210 Fast-Path Responsibility Gate

**Date:** 2026-08-21

**Status:** `PASS / SYSTEMF_FAST_FREEZE_READY`

**System candidate:** `SystemF-fast-S0-GQ-2026-08-21`

## Gate Result

I210 passes for the fast-path system candidate.

The candidate can be frozen as a deadline-prioritized SystemF variant because its main development evidence is positive, the safety signal does not regress, and no data/runtime boundary was changed.

## What Passed

| Check | Result | Evidence |
|---|---|---|
| D-A main metric positive | PASS | NIAH `correct_and_cited` +6.13pp; 2Wiki `correct_and_cited` +39.65pp |
| Main metric CI lower bound positive | PASS | NIAH CI low +2.63pp; 2Wiki CI low +29.97pp |
| Answer does not regress by more than 2pp | PASS | NIAH +6.27pp; 2Wiki +6.95pp |
| Coverage does not regress by more than 2pp | PASS | NIAH +7.13pp; 2Wiki +2.53pp |
| Citation precision/recall do not regress by more than 3pp | PASS | NIAH precision +1.27pp / recall +1.98pp; 2Wiki precision +31.37pp / recall +35.47pp |
| Unsupported ungrounded assertions do not increase | PASS | 30.08% -> 0.13%, delta -29.95pp |
| Runtime errors / missing traces | PASS | G400/G410 formal run manifests report complete runs and no blocking tripwires |
| Held-out boundary | PASS | No HotpotQA/MuSiQue/RGB held-out was read |
| Selector claim discipline | PASS | `SQ=S0`, so no learned Selector gain is claimed |

## Important Interpretation

This is a pass for the fast-path system:

```text
Retriever + S0 TopK keep-all Selector + GQ Generator
```

It is not a pass for Utility Selector training.  Since `D=B`, there is no same-GQ Selector improvement to claim.  The valid claim is:

> Under frozen development inputs, replacing G0 with GQ improves answer+citation grounding and strongly reduces unsupported ungrounded answers.

## Next Step

Proceed to I220 and freeze `SystemF-fast-S0-GQ-2026-08-21` as the deadline-prioritized SystemF candidate.

After I220, final HotpotQA/MuSiQue/RGB held-out H still requires separate user authorization.

