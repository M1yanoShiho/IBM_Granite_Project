# I220 SystemF-Fast Freeze

**Date:** 2026-08-21

**Status:** `COMPLETE / SYSTEMF_FAST_FROZEN`

**Frozen system:** `SystemF-fast-S0-GQ-2026-08-21`

## Frozen System

```text
frozen Hybrid RRF Retriever
-> S0 TopK keep-all Selector
-> frozen GQ Generator
-> one answer
```

This is the deadline-prioritized SystemF candidate.  It is a valid full-flow system because it still follows `Retriever -> Selector -> Generator -> one answer`.  Its Selector is deliberately simple (`S0` TopK keep-all), so the result should be described as a Generator-first system improvement, not as a learned Selector improvement.

## Why This System Is Frozen

The I210 responsibility gate passed:

- NIAH `correct_and_cited`: +6.13pp over `TopK+G0`;
- 2Wiki `correct_and_cited`: +39.65pp over `TopK+G0`;
- unsupported ungrounded answers: 30.08% -> 0.13%;
- no held-out read;
- no Retriever/Selector/Generator/scoring rule changed during the fast-path decision.

## Allowed Claims

- The frozen GQ Generator improves the fast-path full-flow system on development evidence.
- The fast-path system strongly reduces unsupported ungrounded answers on the checked unsupported safety slice.
- The system is ready for one-time held-out evaluation if the user separately authorizes H100.

## Not Allowed Claims

- Do not claim Utility Selector training succeeded.
- Do not claim Legacy Selector or Utility Selector improved the same GQ end-to-end.
- Do not claim HotpotQA/MuSiQue/RGB performance until H100 is explicitly authorized and run.

## Next Step

The next stage is H100 one-time held-out evaluation.  It must remain blocked until the user separately authorizes running final held-out datasets.

S110 continues in the background as an optional enhancement route; it does not block this frozen fast-path SystemF.

