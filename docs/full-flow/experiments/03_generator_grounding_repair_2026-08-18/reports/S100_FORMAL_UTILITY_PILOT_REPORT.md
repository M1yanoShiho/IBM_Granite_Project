# S100 Formal Utility Pilot Report

**Stage:** S100 formal utility pilot
**Date:** 2026-08-21
**Status:** `S100_PILOT_COMPLETE / S110_READY`
**Held-out:** not read

## Purpose

S100 used the frozen three-seed GQ teacher family to measure how each candidate evidence item affects the Generator.  For each fixed pilot question, GQ was run on the full context and on leave-one-out contexts where one evidence item was removed.  References and support identities were used only after generation for scoring and label construction.

This stage checks whether the utility signal is stable and dense enough to justify S110 materialization.  It does not train a Selector, does not run the full system, and does not read held-out data.

## Runtime Identity

- Runtime root: `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/S100-v1/formal-fbca132`
- Server repo head before formal S100: `fbca132b4b8aaafc28c26ea9685ccd19f1c84c8d`
- GQ: `GQ-GR-C-THREE-SEED-FAMILY-2026-08-20`
- Sample seed: `S100-v1-fixed-pilot-2026-08-20`
- Questions: 100 total, split as 50 NIAH train and 50 2Wiki train
- Evidence per question: 10
- Task identities: 1,100 per seed, covering full context plus ten leave-one-out variants
- Generated rows: 3,300 total across seeds 13, 42, and 73

## Generation Verification

| Seed | Status | Tasks | Errors | Missing trace | Runtime reference loaded | Runtime support provenance loaded | Held-out read |
|---:|---|---:|---:|---:|---|---|---|
| 13 | `COMPLETE` | 1,100 | 0 | 0 | false | false | false |
| 42 | `COMPLETE` | 1,100 | 0 | 0 | false | false | false |
| 73 | `COMPLETE` | 1,100 | 0 | 0 | false | false | false |

Generation hashes:

- seed13 generations: `bf95a4caccdc404f5bd9429a7c507fc14600917a3f0f69a4223ff284982348e3`
- seed42 generations: `be9fecb31f33e5c90d3f566ab9b523c5f8f47cabf5af7b92f058e16cfa2bf015`
- seed73 generations: `eb86ee61e6bca4d63d7266008b9e647f55293a658c1850546fc5cf94920f0ef9`

## Label Results

S100 produced 1,000 evidence-level labels from 100 questions.

| Label | Count |
|---|---:|
| MUST_KEEP | 90 |
| SAFE_DROP | 815 |
| NEUTRAL | 32 |
| UNCERTAIN | 63 |

Summary rates:

- stable label rate: 0.9370
- utility label rate: 0.9050
- uncertain rate: 0.0630
- utility labels: 905
- datasets with utility: 2Wiki and NIAH

By dataset:

| Dataset | MUST_KEEP | SAFE_DROP | NEUTRAL | UNCERTAIN | Total |
|---|---:|---:|---:|---:|---:|
| 2Wiki | 66 | 363 | 8 | 63 | 500 |
| NIAH | 24 | 452 | 24 | 0 | 500 |

By evidence role:

| Role | MUST_KEEP | SAFE_DROP | NEUTRAL | UNCERTAIN | Total |
|---|---:|---:|---:|---:|---:|
| Support | 90 | 2 | 32 | 26 | 150 |
| Distractor | 0 | 813 | 0 | 37 | 850 |

The pilot therefore produced dense utility signal on both datasets.  UNCERTAIN items remain conservative: they are kept by default or excluded from utility loss, rather than being forced into SAFE_DROP.

## Boundary Check

- generation task packet contains no reference answers;
- generation task packet contains no support provenance;
- scoring used references only after all generation completed;
- sealed600 and final held-out data were not read;
- Selector training was not started;
- full-system evaluation was not started;
- GQ was not changed after utility generation began.

## Archived Artifacts

Small, auditable artifacts were copied into `artifacts/S100/formal-fbca132/`.  Large generation files remain in the server runtime and are identified by SHA256 in the run manifests.

Key archived hashes:

- ordered IDs: `a7b80c29ceeddd7290c8e47995650484f29b7f643274f200683f94f4d4c09d51`
- prepare manifest: `4e2ab1e1e4fdbf02813ca771a9ae86ee73c495d4928f4ee8e1a34cd7fdfb1b02`
- score report: `eb2cffac78492b87f00d00075df243d64d091ced2c40debe9df4fc5a3449fa35`
- score manifest: `1fc9abb9be55230d380f2ab2058fa43f9543005a1ad47fe9a83dd83c985a66e7`
- scored rows: `82db7b1d30b55e5f489ec2240d2e68032b83a6dfcf7ab4732c6848499441e8dc`
- utility labels: `28abc423e113107dd67ff2045e0c4a825fb06b5f78df3dce3c7fb571e782dcab`

## Decision

S100 is `S100_PILOT_COMPLETE` and recommends `S110_READY`.

This is a positive utility-signal result, not a Selector or full-system result.  The next stage may proceed to S110 utility materialization under the same frozen GQ and the existing held-out restrictions.
