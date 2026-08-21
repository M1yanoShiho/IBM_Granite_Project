# G410 Implementation And Smoke Report

**Stage:** G410 implementation/smoke
**Date:** 2026-08-20
**Status:** `IMPLEMENTATION PASS / FORMAL G410 READY`
**Formal G410:** not run in this stage
**Held-out:** not read

## What Was Added

Added `scripts/full_flow_g410_cross_data_qualification.py` and focused tests for the G410 2Wiki cross-data qualification path.

The runner has three commands:

- `prepare`: reads G223 validation cases offline and writes a no-answer 2Wiki task packet plus a separate reference packet.
- `run`: generates one locked `GR-C` seed from the no-answer task packet. This command has no reference/gold input.
- `score`: after generation, joins fixed G310 G0, three locked `GR-C` seed outputs, the separated references, and MiniCheck citation scoring.

The current implementation covers the registered G223 2Wiki internal cross-data screen: 95 case groups x 5 context variants = 475 tasks. ASQA/QAMPARI revealed citation guard remains explicitly pending for G420 boundary confirmation and is not treated as a new held-out result here.

## What Was Verified

Local verification:

- `py_compile` passed for the G410 runner and tests.
- `pytest tests/full_flow/test_full_flow_g410_cross_data_qualification.py -q` passed: 2 tests.

Server verification:

- Server repo synced to `34736908c612bbd4647460f601a5a10a063ca32d`.
- Server `py_compile` passed.
- Server G410 tests passed: 2 tests.
- Real prepare smoke produced 475 no-answer tasks and a separated reference packet from G223 validation 2Wiki.
- Three real generation smoke runs completed for seeds 13/42/73, each with 1 task, 0 runtime errors, and no runtime gold/reference read.
- Score smoke used `--allow-subset` only for the 1-task smoke subset and returned `G410_CROSS_DATA_RESPONSIBILITY_PASS`.

The score smoke status is not an effect conclusion because it covers only 1 task.

## Runtime Smoke Artifacts

Server runtime root:

```text
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G410-v1
```

Smoke directories:

- `smoke-prepare-ae6ca76`
- `smoke-seed13-ae6ca76`
- `smoke-seed42-ae6ca76`
- `smoke-seed73-ae6ca76`
- `smoke-score-3473690`

## Boundary Check

This stage did not:

- run formal G410;
- run G420;
- freeze a teacher Generator;
- create utility labels;
- read sealed600;
- read system held-out;
- modify Retriever, Selector, G330 recipe, seeds, metrics, or gates.

## Result

G410 implementation is ready for formal execution. The next step is to launch full G410 2Wiki cross-data generation for seeds 13/42/73 on the 475 prepared tasks, then score the completed outputs with fixed G310 G0.
