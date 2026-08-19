# G310 Screen Implementation Report

**Date:** 2026-08-18
**Status:** `IMPLEMENTATION TEST PASS / SUPERSEDED BY FORMAL SCREEN REPORT`

## What Changed

G310 now has a dedicated seed13 screen runner and scorer in `scripts/full_flow_g310_seed13_screen.py`.

It compares only the predeclared arms:

- `G0`: frozen base Generator path;
- `GR-F`: fresh draft LoRA from frozen Granite base;
- `GR-C`: continuation from the frozen GM13 adapter.

The runner uses the full Generator path for the screen: adapter-enabled draft call, frozen-base claim splitter, and frozen TRUE routing. The scorer is post-generation only: it reads references after generation for answer scoring and uses MiniCheck/ALCE-style sentence-citation scoring for citation precision, citation recall, and `correct_and_cited`.

## Boundary Status

- Runtime generation does not read gold/reference answers.
- TRUE remains the runtime verifier, not the citation judge.
- MiniCheck is used only after generation for citation scoring.
- Held-out/sealed/dev final test data was not read.
- Selector utility labels were not generated.
- No recipe was selected in this implementation step.

## Tests

Local syntax check:

```text
python3 -m py_compile scripts/full_flow_g310_seed13_screen.py tests/scripts/test_full_flow_g310_seed13_screen.py
PASS
```

Local pytest was unavailable in the local Python installation:

```text
python3 -m pytest tests/scripts/test_full_flow_g310_seed13_screen.py
No module named pytest
```

Server syntax check and unit test were run in the project experiment environment:

```text
python -m py_compile scripts/full_flow_g310_seed13_screen.py tests/scripts/test_full_flow_g310_seed13_screen.py
PASS

python -m pytest tests/scripts/test_full_flow_g310_seed13_screen.py
3 passed in 0.12s
```

## Current Runtime State

This implementation report records the code/test readiness checkpoint. The later formal G310 screen completed and selected `GR-C`; see [G310_FORMAL_SCREEN_REPORT.md](G310_FORMAL_SCREEN_REPORT.md).

The formal GR-F and GR-C seed13 adapter training jobs used:

- `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G310-v1/formal-grf-seed13`
- `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G310-v1/formal-grc-seed13`

This report only freezes the G310 screen implementation. The formal recipe decision is archived separately.

## SHA256

| Artifact | SHA256 |
|---|---|
| `scripts/full_flow_g310_seed13_screen.py` | `1aae31c5eed5d0b073bbd1be96c57a0b118bbe1435f4f670374c84ed53c505d5` |
| `tests/scripts/test_full_flow_g310_seed13_screen.py` | `792cbe08708b6036f33dd17d8b79d749645efcca73f74b2fe20990d888493769` |

## Result

`IMPLEMENTATION TEST PASS`: the fixed G310 screen code was ready and was used by the later formal seed13 screen.

This implementation checkpoint alone is not a Generator repair success claim.
