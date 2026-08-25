# G320 Recipe Freeze Report

**Date:** 2026-08-19
**Status:** `COMPLETE / RECIPE_FROZEN`
**Frozen recipe:** `GR-C`

## Decision

G320 freezes `GR-C` as the only Generator training recipe allowed to enter G330.

This is a recipe freeze, not a teacher Generator freeze. It does not create `GQ`, does not start Selector utility labels, and does not authorize held-out evaluation.

## Evidence Used

G320 uses only the completed G310 formal seed13 screen:

- G310 archive commit: `256243ba854bdf1fc0606759439b94071a4b0681`
- G310 formal screen status: `RECIPE_SELECTED`
- G310 selected recipe: `GR-C`
- G310 formal screen manifest SHA256: `667552a35a321268f9674d48cae18e869df3e3d1c99dce993a17da5381b0ad50`
- G310 formal score report SHA256: `e76c5b0e2b55653269dcad50e09c20f8acec28760d0aecc8fd24c603cf03f5b8`

No new training, scoring, utility-label generation, or held-out access was performed in G320.

## Tie-Break Trace

The predeclared G310 selection rule was:

1. discard recipes that hit a hard exclusion;
2. among surviving recipes, choose the higher maximin `correct_and_cited` delta across NIAH and 2Wiki model-val;
3. if still tied, choose `GR-F`.

| Recipe | Survives hard exclusions | Overall delta | NIAH delta | 2Wiki delta | Maximin delta | Failures |
|---|---|---:|---:|---:|---:|---|
| GR-C | yes | +0.0949 | -0.0538 | +0.4189 | -0.0538 | none |
| GR-F | no | +0.0765 | -0.0738 | +0.4042 | -0.0738 | `answer_regression_gt_2pp` |

`GR-C` is selected. The fallback tie-break was not used.

## Frozen Recipe

All G330 Generator training must use this recipe unless a later stage is explicitly failed and separately amended.

| Field | Frozen value |
|---|---|
| Recipe id | `gr-c` |
| Initialization | continue from `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G220-v1/formal/gm-seed13/adapter` |
| Init adapter config SHA256 | `3891557ce282e952949f6deeb0fff99e134113211f30b99a1a05f9d4a36a9171` |
| Learning rate | `5e-05` |
| Epochs | `1` |
| Microbatch size | `1` |
| Gradient accumulation groups | `8` |
| Max length | `2304` |
| LoRA | `r=8`, `alpha=16`, `dropout=0.05`, `bias=none` |
| LoRA target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| Query-group equalization | enabled; one case group is one unit, not one context variant |
| Token weighting | prompt `0`, answer/punctuation `1`, citation bracket/index `4`, eos `1` |
| Decode policy | greedy at runtime |
| Adapter scope | draft generation call only |
| Claim splitter scope | frozen Granite base with adapters disabled |
| TRUE scope | frozen verifier; not trained |

Frozen identities:

| Item | SHA256 / identity |
|---|---|
| G300 training script | `8973ac773c4da18dd1648c88de8ab316378ffb99eb59d080e170ac9767014b77` |
| G310 screen script | `1aae31c5eed5d0b073bbd1be96c57a0b118bbe1435f4f670374c84ed53c505d5` |
| Granite model config | `9a0e589b69e7d3ad9fb9fb2c844aa7d7156e052cb7ea4211de7de48ab7c8525c` |
| G223 manifest | `9213eb32f1234290ac5d25c66c51756927adaff58ff4cd4cf2f70ece2312b30b` |
| Train cases | `88e5592ed796e756fb836c9faee22393a0fa56c57b50eae69c449bd78c438635` |
| Validation cases | `f896e8a92fe5e36757249f1a22d3c55f579233879beca229341833137f95cb67` |
| Ordered IDs | `b3bd73290f9753ada6b6f79b7d6758d755b6eec1e42035936b76ceee90b5b283` |

The existing G310 `GR-C` seed13 adapter is eligible to be reused as the G330 seed13 artifact if G330 revalidates that its manifest, adapter hashes, recipe, and boundaries match this freeze:

| Artifact | SHA256 |
|---|---|
| G310 `GR-C` training manifest | `2aadc3fd2ff68873b597612d21396fea2a79ff89ba9bf00d53958d22a4918b99` |
| G310 `GR-C` adapter weights | `492d336c7acc32785becded707220dbdf1a8acf9a895630147fc4e8cd28d0707` |
| G310 `GR-C` adapter config | `1918333d819e6007d3faf17d3497c4f4665f7c0d3cb032f510986cfdb9837942` |

## G330 Lock

G330 is unlocked with one frozen recipe:

- recipe: `GR-C`
- seeds: `13`, `42`, `73`
- seed choice is fixed and must not be changed after seeing results;
- no `GR-F`, no alternate learning rate, no new threshold, no held-out, no utility labels;
- if any seed fails technically, record the failure and follow the plan rather than changing the recipe.

## Boundary Status

- G223 remains `CONTROLLED_CONTINUATION_READY`, not clean freeze.
- 2Wiki model-val screen size remains 95 and must be reported.
- G320 did not read held-out or sealed final data.
- G320 did not generate Selector utility labels.
- G320 did not change Retriever, Selector, TRUE, MiniCheck, or scoring rules.

## Result

`G320 COMPLETE / RECIPE_FROZEN`: proceed to G330 using `GR-C` only.

This does not establish final Generator qualification. GQ can only be frozen later by G430 after G330/G400/G410/G420 are completed.
