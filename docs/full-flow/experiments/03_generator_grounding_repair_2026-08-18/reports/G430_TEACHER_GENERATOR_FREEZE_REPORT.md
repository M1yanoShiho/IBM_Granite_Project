# G430 Teacher Generator Freeze Report

**Stage:** G430 teacher Generator freeze
**Date:** 2026-08-20
**Status:** `GQ_FROZEN_NEW_GRC`
**GQ:** `GR-C` three-seed frozen teacher family
**Held-out:** not read
**Utility labels:** not started

## What Was Frozen

G420 recommended `FREEZE_NEW_GRC_IN_G430`. G430 therefore freezes the new Generator teacher as the `GR-C` three-seed family from seeds 13, 42, and 73.

This is not best-seed selection. All three qualified `GR-C` adapters remain bound to GQ, and later Selector utility work must not replace, retune, or reselect them.

## Frozen Runtime Identity

Base Generator:

```text
/scratch/fl25387/IBM_Granite_Project_latest/model-cache/models--ibm-granite--granite-4.1-3b/snapshots/c0650403e44e78ec0262dab1c90914c65b196c4e
```

Frozen recipe:

- recipe: `GR-C`
- learning rate: `5e-5`
- epochs: `1`
- max length: `2304`
- LoRA: `r=8`, `alpha=16`, `dropout=0.05`
- runtime decode: greedy
- adapter scope: draft generation call only
- claim splitter: frozen Granite base with adapters disabled
- TRUE verifier: frozen, not trained

Frozen adapters:

| Seed | Runtime adapter | adapter_model.safetensors SHA256 | adapter_config.json SHA256 |
|---:|---|---|---|
| 13 | `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G310-v1/formal-grc-seed13/adapter` | `492d336c7acc32785becded707220dbdf1a8acf9a895630147fc4e8cd28d0707` | `1918333d819e6007d3faf17d3497c4f4665f7c0d3cb032f510986cfdb9837942` |
| 42 | `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G330-v1/formal-grc-seed42/adapter` | `96d8087e1dbed831ad795fb7a037677eeb7e2d6986a91df61a897b9c1b6d383c` | `df6c8e4c39f3d7567849a4a32355aea4bb9176c784d089832fe14a8697a2d8f8` |
| 73 | `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G330-v1/formal-grc-seed73/adapter` | `d5f90954f3ac2829a213ab7c3b04e6d26b89990c42e95306bc74743d1ac2431c` | `f1659959d9ffe7348c370e9f01a497066710793c9754e9d06c5d8963aaf45dd1` |

Server entity verification was performed on 2026-08-20 and matched all archived G330 hashes.

## Qualification Evidence

GQ freeze is supported by:

- G320 recipe freeze: `GR-C`;
- G330 three-seed fit: seeds 13/42/73 complete and reload-pass;
- G400 locked NIAH qualification: `G400_NIAH_RESPONSIBILITY_PASS`;
- G410 locked 2Wiki cross-data qualification: `G410_CROSS_DATA_RESPONSIBILITY_PASS`;
- G420 combined Generator gate: `G420_NEW_GENERATOR_QUALIFIED_G430_READY`.

G420 primary bootstrap summary:

- G400 K_topk correct+cited: +6.13pp, 95% CI [2.63pp, 9.90pp];
- G410 all_2wiki correct+cited: +39.65pp, 95% CI [29.97pp, 49.35pp].

## What G430 Allows

G430 unlocks S100 utility pilot using this frozen GQ.

S100 may measure whether full-context and leave-one-out utility labels are stable enough to train a Utility Selector. If labels are too sparse or unstable, Selector utility training must stop or fall back according to the plan.

## Boundary Check

This stage did not:

- generate utility labels;
- train or qualify a Selector;
- run full system I;
- read sealed600;
- read HotpotQA, MuSiQue-Full, RGB, or RGB-counterfactual;
- modify Retriever, Selector, Generator recipe, adapters, seeds, gates, or scoring rules.

Once S100 starts using GQ, GQ must not be changed without rebuilding utility labels.
