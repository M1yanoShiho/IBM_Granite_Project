# G330 Three-Seed Fit Report

**Date:** 2026-08-19
**Status:** `COMPLETE / THREE_SEED_FIT`
**Recipe:** `GR-C`

## Result

G330 completed the frozen `GR-C` three-seed fit for seeds `13`, `42`, and `73`.

Seed13 was the formal `GR-C` seed13 adapter trained during G310 and revalidated against the G320 freeze. Seeds 42 and 73 were newly trained under the same frozen G320 recipe. All three training manifests report `COMPLETE`, `reload_check.status=PASS`, `sealed_or_heldout_read=false`, and `utility_labels_started=false`.

This stage only produces trained adapters and manifests. It does not qualify the Generator, does not freeze `GQ`, does not generate Selector utility labels, and does not authorize held-out evaluation.

## Runtime

Runtime roots:

```text
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G310-v1/formal-grc-seed13
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G330-v1/formal-grc-seed42
/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G330-v1/formal-grc-seed73
```

Seed42 and seed73 were launched as separate screen jobs on separate GPUs:

- `g330_grc42`
- `g330_grc73`

Both screens exited after writing final manifests, and GPUs were free at the completion check.

## Training Summary

| Seed | Source | Status | Groups | Examples | Steps | Mean group loss | Validation group loss | Final 25-group loss | Reload |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 13 | reused from G310 formal `GR-C` | COMPLETE | 2370 | 9207 | 297 | 0.044435 | 0.258522 | 0.018767 | PASS |
| 42 | G330 formal `GR-C` | COMPLETE | 2370 | 9207 | 297 | 0.041775 | 0.227723 | 0.009626 | PASS |
| 73 | G330 formal `GR-C` | COMPLETE | 2370 | 9207 | 297 | 0.048233 | 0.258000 | 0.009684 | PASS |

Runtime and adapter hashes:

| Seed | Training manifest SHA256 | Adapter weights SHA256 | Adapter config SHA256 |
|---:|---|---|---|
| 13 | `2aadc3fd2ff68873b597612d21396fea2a79ff89ba9bf00d53958d22a4918b99` | `492d336c7acc32785becded707220dbdf1a8acf9a895630147fc4e8cd28d0707` | `1918333d819e6007d3faf17d3497c4f4665f7c0d3cb032f510986cfdb9837942` |
| 42 | `2d78d06f79fb3e334eced1b222c72e3bdcf319009b0380f2efa7240b20d06580` | `96d8087e1dbed831ad795fb7a037677eeb7e2d6986a91df61a897b9c1b6d383c` | `df6c8e4c39f3d7567849a4a32355aea4bb9176c784d089832fe14a8697a2d8f8` |
| 73 | `a4683e1af118c96e658f2f9e2b8c6384b4b2feb471bfc70467f6d59112894aa2` | `d5f90954f3ac2829a213ab7c3b04e6d26b89990c42e95306bc74743d1ac2431c` | `f1659959d9ffe7348c370e9f01a497066710793c9754e9d06c5d8963aaf45dd1` |

Shared identities:

| Item | SHA256 / identity |
|---|---|
| G320 recipe freeze commit | `8cceac7e1ce0616e6e3d6485a96168caf3d29fbd` |
| Training script | `8973ac773c4da18dd1648c88de8ab316378ffb99eb59d080e170ac9767014b77` |
| G223 manifest | `9213eb32f1234290ac5d25c66c51756927adaff58ff4cd4cf2f70ece2312b30b` |
| Train cases | `88e5592ed796e756fb836c9faee22393a0fa56c57b50eae69c449bd78c438635` |
| Validation cases | `f896e8a92fe5e36757249f1a22d3c55f579233879beca229341833137f95cb67` |
| Ordered IDs | `b3bd73290f9753ada6b6f79b7d6758d755b6eec1e42035936b76ceee90b5b283` |
| Granite model config | `9a0e589b69e7d3ad9fb9fb2c844aa7d7156e052cb7ea4211de7de48ab7c8525c` |
| Environment | torch `2.6.0+cu126`, transformers `4.57.6`, peft `0.20.0` |

## Boundary Check

- All three seeds use recipe `gr-c`.
- All three seeds use fixed train groups 2370 and training examples 9207.
- All three seeds use G223 `CONTROLLED_CONTINUATION_READY`; this is still not clean freeze.
- 2Wiki model-val screen size remains 95.
- All three manifests report `dev_read=false`, `sealed_or_heldout_read=false`, and `utility_labels_started=false`.
- Log scan for seed42 and seed73 found no traceback, OOM, killed process, or error keyword.

## Next Stage

G330 unlocks Generator qualification runs:

- G400: locked NIAH qualification;
- G410: cross-data qualification;
- G420: Generator responsibility gate.

G330 by itself is not a success claim. The next evidence is generated and scored only in G400/G410/G420.
