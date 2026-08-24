# G400 Implementation And Smoke Report

**Stage:** G400 implementation/smoke  
**Date:** 2026-08-19  
**Status:** `IMPLEMENTATION PASS / FORMAL G400 READY`  
**Formal G400:** not run in this stage  
**Held-out:** not read

## What Was Added

Added `scripts/full_flow_g400_niah_qualification.py` and targeted tests for the locked G400 NIAH qualification path.

The runner has two separate commands:

- `run`: generates one locked `GR-C` candidate seed at a time. This command has no gold/reference argument.
- `score`: after generation, joins fixed G0 from A002/B100, candidate generations, NIAH dev gold, and MiniCheck citation scoring.

The implementation keeps G0 fixed instead of regenerating it. This preserves the comparison boundary and reduces GPU cost.

## What Was Verified

Local verification:

- `py_compile` passed for the G400 runner and test file.
- `pytest tests/full_flow/test_full_flow_g400_niah_qualification.py -q` passed: 2 tests.
- Adjacent regression tests passed: 11 tests across G230, G230 citation, and G400.

Server verification:

- Server repo synced to `82a8144b80c340fe4f81ded3144bbbb53b3e59e8`.
- Server `py_compile` passed.
- Server adjacent regression tests passed: 11 tests.
- Real runtime smoke generated 1 NIAH answerable task for each locked seed: 13, 42, 73.
- All three smoke runs completed with 0 runtime errors and 0 missing traces.
- Run manifests recorded `gold_loaded_at_runtime=false`, `reference_answers_loaded_at_runtime=false`, `sealed_or_heldout_read=false`, and `utility_labels_started=false`.
- A score smoke successfully combined fixed A002/B100 G0, the three seed smoke outputs, NIAH dev gold, and MiniCheck.

The score smoke status was `CONTROLLED_CONTINUATION_SIGNAL`, but this is from a single task and is not an effect conclusion.

## Runtime Smoke Artifacts

Server runtime root:

`/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G400-v1`

Smoke directories:

- `smoke-seed13-474e99d`
- `smoke-seed42-474e99d`
- `smoke-seed73-474e99d`
- `smoke-score-82a8144`

## Boundary Check

This stage did not:

- run formal G400;
- run G410 or G420;
- freeze a teacher Generator;
- create utility labels;
- read sealed600;
- read system held-out;
- change Retriever, Selector, G330 recipe, seeds, metrics, or gates.

## Result

G400 implementation is ready for formal execution. The next step is to launch full locked G400 generation for seeds 13/42/73 using the same runner and frozen inputs, then score the completed outputs.
