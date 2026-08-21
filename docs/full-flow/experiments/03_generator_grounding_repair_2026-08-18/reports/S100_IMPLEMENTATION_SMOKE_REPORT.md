# S100 Implementation Smoke Report

**Stage:** S100 utility pilot implementation/smoke
**Date:** 2026-08-20
**Status:** `IMPLEMENTATION_SMOKE_PASS / FORMAL_S100_READY`
**Formal S100:** not started
**Held-out:** not read

## What Was Added

Added `scripts/full_flow_s100_utility_pilot.py` with three commands:

- `prepare`: fixes the S100 pilot question set and writes a generation-only task packet;
- `run`: runs frozen GQ for one seed without loading references or support provenance;
- `score`: scores only after generation and writes scored rows, utility labels, a report, and a manifest.

The generation task packet deliberately excludes `answer`, `official_answer`, `reference_answers`, `gold`, and `support_evidence_ids`.  References and support provenance are stored only in the scoring packet.

## Local Verification

Local tests passed:

```text
tests/full_flow/test_full_flow_g400_niah_qualification.py
tests/full_flow/test_full_flow_g410_cross_data_qualification.py
tests/full_flow/test_full_flow_g420_generator_gate.py
tests/full_flow/test_full_flow_s100_utility_pilot.py

8 passed
```

Source hashes:

- `scripts/full_flow_s100_utility_pilot.py`: `f8c8e755b7215db609e8757eac6c66cb724d8e2d88585af2129f23ba4d7b5cc2`
- `tests/full_flow/test_full_flow_s100_utility_pilot.py`: `d49097581cf25aecc82ee8607afcf4adb39eee5f23cf18216bb874cb8d20413a`

## Server Verification

Server repo head:

```text
4a3ebbb7fa6ea5f3e8fd952fcedc73a2e365577f
```

Server tests passed:

```text
8 passed
```

Runtime smoke root:

```text
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/S100-v1/smoke-4a3ebbb
```

Prepare smoke used 1 NIAH train question and 1 2Wiki train question, producing 22 generation tasks.  The prepare manifest confirms:

- `questions=2`;
- `tasks=22`;
- generation task packet contains no reference answers;
- generation task packet contains no support provenance;
- `sealed_or_heldout_read=false`.

Generation smoke ran all three frozen GQ seeds with `limit_tasks=2`:

| Seed | Status | Tasks | Errors | Missing trace |
|---:|---|---:|---:|---:|
| 13 | `COMPLETE` | 2 | 0 | 0 |
| 42 | `COMPLETE` | 2 | 0 | 0 |
| 73 | `COMPLETE` | 2 | 0 | 0 |

Scoring smoke completed and wrote report/rows/manifest.  Because the smoke intentionally ran only 2 tasks per seed, it did not contain complete full plus all leave-one-out groups, so it produced `labels=0` and `s110_recommendation=NO_DECISION_INCOMPLETE_QUESTIONS`.  This is expected for smoke and is not a formal S100 result.

## Boundary Check

This stage did not:

- run the formal 100-question S100 pilot;
- create formal utility labels;
- train Selector;
- run full-system I;
- read sealed600;
- read HotpotQA, MuSiQue-Full, RGB, or RGB-counterfactual;
- modify Retriever, Selector, GQ adapters, GQ recipe, GQ seeds, TRUE, or scoring rules.

## Next Step

S100 implementation and smoke are ready.  The next step is formal S100: fixed 50 NIAH train plus 50 2Wiki train questions, full context plus ten leave-one-out variants, run under seeds 13/42/73, then score and decide whether S110 materialization has enough stable utility signal.
