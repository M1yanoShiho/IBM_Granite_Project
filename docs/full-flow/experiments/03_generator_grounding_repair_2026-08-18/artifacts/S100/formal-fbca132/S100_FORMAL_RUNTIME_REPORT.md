# S100 Utility Pilot Report

**Status:** `S100_PILOT_COMPLETE`
**S110 recommendation:** `S110_READY`
**Questions:** 100
**Generation tasks:** 1100
**Labels:** 1000

## Label Summary

| label | count |
|---|---:|
| MUST_KEEP | 90 |
| SAFE_DROP | 815 |
| NEUTRAL | 32 |
| UNCERTAIN | 63 |

- stable label rate: 0.9370
- utility label rate: 0.9050
- uncertain rate: 0.0630
- datasets with utility: 2wiki, niah

## Boundary Check

- generation runtime did not load reference answers or support provenance;
- scoring used references only after generation;
- sealed and held-out data were not read;
- Selector training and full-system evaluation were not started.
