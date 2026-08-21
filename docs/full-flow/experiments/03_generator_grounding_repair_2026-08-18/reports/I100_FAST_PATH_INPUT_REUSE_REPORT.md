# I100 Fast-Path Input Freeze / Reuse

**Date:** 2026-08-21

**Status:** `COMPLETE / FAST_PATH_INPUT_REUSE_ACCEPTED`

**System candidate:** `SystemF-fast-S0-GQ-2026-08-21`

## Decision

I100 does not launch a new Retriever run for the fast-path development package.  It freezes the already completed development inputs that were used by the verified G400/G410/G420/G430 chain:

- NIAH development input: frozen F000/A002/B100 candidate/context artifacts.
- 2Wiki development screen: frozen G410 internal model-val task packet.
- Generator: frozen GQ from G430.
- Selector for the fast path: `S0` TopK keep-all.

This is enough for I200 development/baseline packaging because the immediate claim is not "a learned Selector improves retrieval."  The current claim is narrower:

> With the same frozen development inputs and TopK Selector, GQ improves the system compared with the previous G0 Generator.

## Why Not Rerun Retriever Now

The deadline-critical question is whether the method direction is worth carrying into full reporting.  Existing checked artifacts already bind the relevant development inputs:

- F000/A002 use the same NIAH runtime input hash: `bad751bde256d01e204cbae85de2d43ce8e865689b4a08a9fb83500843ea3805`.
- B100 freezes the 218-question matched NIAH context packet used by the S090 triage.
- G400 validates GQ on NIAH with three seeds, 2873 tasks/seed, 0 runtime errors, and 0 missing traces.
- G410 validates GQ on a 95-case 2Wiki internal screen with three seeds, 475 tasks/seed, 0 blocking tripwires.
- G420 combines G400/G410 and qualifies GQ for freezing.

Rerunning Retriever now would spend time without changing the main fast-path comparison.  It is deferred until it is needed for final SystemF/H held-out execution.

## What This Supports

This I100 freeze/reuse supports:

- I200 development result packaging for `TopK+G0` vs `TopK+GQ`;
- reporting the Legacy Selector diagnostic as non-primary evidence;
- freezing the current fast-path candidate as `Retriever + S0 TopK + GQ` after I210 if responsibility checks remain positive.

It does not support:

- a claim that Utility Selector training has succeeded;
- a claim that Legacy Selector improves the same GQ end-to-end;
- final HotpotQA/MuSiQue/RGB held-out conclusions.

## Boundary Check

- No held-out data was read.
- No sealed600 data was used.
- No Generator, Selector, Retriever, seed, threshold, or scoring rule was changed.
- S110 continues in the background as optional Utility Selector materialization.
- I200 can proceed immediately using the frozen development evidence.

