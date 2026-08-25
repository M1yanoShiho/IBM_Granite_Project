# Experiment 04 Goal 3 execution readiness

**State:** `COMPLETE / PASS`

**Formal held-out generation:** complete and frozen  
**Formal held-out scoring:** complete

## Checkpoint recovery and formal launch

After the VPN connection was restored, the original teacher server `it097952` was reachable.
All seven frozen custom files were present at their registered runtime paths. Their source-side
SHA-256 values matched the frozen manifest exactly; the files were streamed to the declared
BluePebble restore directory, rehashed at the destination, and made mode `0600`. No checkpoint
was retrained, substituted, or modified.

All three deterministic preflights then passed without invoking a model or exposing held-out
content: HotpotQA `400`, MuSiQue answerable `400`, and RGB noise `300`. The first one-job
seven-arm bundles were launched as Slurm jobs `18673236`, `18673237`, and `18673238`; their
subsequent invalidation and fresh replacements are recorded below. Every attempt keeps the
scorer-only path out of generation subprocesses and introduces it only after all seven generation
files have been frozen and hashed.

## First-attempt invalidation and repaired rerun

The first jobs (`18673236`, `18673237`, `18673238`) revealed that some exact chat-template
prompts exceeded the frozen `2,304`-token input limit. A machine-only audit established an exact
correspondence between these overflows and the observed generation errors without printing or
inspecting held-out content, answers, or labels. HotpotQA and MuSiQue were cancelled; RGB
correctly froze an `INVALID_REQUIRES_FULL_RERUN` manifest and exited before scoring. No output
from that attempt is eligible for Table 1.

The repaired implementation applies the same rank-preserving maximal whole-evidence prefix to
all seven arms while leaving the frozen token limit and model settings unchanged. Local and
server targeted suites are now `19 passed`; a sealed audit of all 7,700 arm-query contexts found
zero remaining overflows. Fresh attempt directories were created and full one-job seven-arm
reruns were first submitted as attempt B, but all three exited at preflight in six seconds because
the submission passed the Selector directory rather than the required checkpoint file. This was
before runtime data reading or generation. A login-node precheck then showed that attempt C also
passed the runtime directory rather than the dataset file; all three still-pending jobs were
cancelled at zero seconds and never ran. With both runtime and scorer arguments resolved to the
dataset-specific sealed files, all three independent preflights passed (`400/400/300`). Fresh
attempt D was then submitted as HotpotQA `18674672`, MuSiQue `18674673`, and RGB `18674674`.
All three passed in-job preflight and are running on `bp1-gpu035`. Older attempts remain untouched for audit and will not be reused or
mixed. See
`GOAL3_PROMPT_BUDGET_REPAIR.md` and
`artifacts/goal3_prompt_budget_repair_manifest.json`.

## Ready components

- The seven-arm dataset-bundle runner is implemented with one query-local candidate pool,
  a real Hybrid Top-40 input to Granite reranking, one shared Ours upstream preparation,
  ordered-prefix resume checks, explicit machine-readable failures, and the 1% whole-bundle
  invalidation guard.
- Direct Granite keeps the model's declared citation indices. Missing or illegal citations
  are not replaced by fabricated citations and remain scoreable as zero citation outcomes.
- Provence output is aligned back to frozen unit IDs, so `Sel.` is not approximated from a
  surviving source alone.
- Runtime generation and scorer-only postprocessing are separate commands. The Slurm wrapper
  removes the scorer path from the environment before preparation/generation and introduces
  it only after the seven output files have been frozen and hashed.
- MiniCheck scoring, five-metric aggregation, all-three-seed mean/sample-SD reporting,
  10,000-resample paired component-cluster bootstrap (seed 13), Table 1 materialization, and
  the 7,700-row final audit are implemented.
- All six sealed input files were copied to physically separate server directories and their
  SHA-256 values match Goal 1. The scorer-only directory is mode `0700`.
- The public Selector backbone and the full TRUE verifier were downloaded and hash-verified;
  the Goal 2 public model cache was reused. Server-side and local Goal 3 tests are each
  `19 passed`, targeted Ruff is `PASS`; the local full regression suite (excluding the already
  recorded stale Experiment 03 state assertion) is `PASS`, Mypy is `PASS` across 137 source
  files, and the Slurm wrapper passes shell syntax validation. Repository-wide Ruff reports
  26 pre-existing unrelated Experiment 03/script findings, which were not modified.

## Historical blocking evidence

Two hash-only Slurm audits checked the historical `/scratch` paths without reading model
contents:

| Job | Node class | Result |
|---|---|---|
| `18671757` | compute (`bp1-compute049`) | all seven expected custom files missing |
| `18671828` | GPU (`bp1-gpu037`) | all seven expected custom files missing |

The same files were absent from the repository, the local OneDrive workspace, the current
`/user/work/fl25387` and `/user/home/fl25387` trees, Git objects/refs, and available archives.
Three additional GPU-node audits (`18672935`, `18672936`, `18672937`) also found the files
missing from node-local `/scratch`. This blocker was resolved by locating the original teacher
server, not by changing the frozen system.

## Exact restore target

Restore the following files beneath
`/user/work/fl25387/experiment04_goal3_custom_checkpoints`:

| Relative file | Required SHA-256 |
|---|---|
| `selector_seed13/model.safetensors` | `86622bd9ab6391c9eb560133b01b0cf3744c3706638ff8b0bd38925b84bf72bf` |
| `grc_seed13/adapter_model.safetensors` | `492d336c7acc32785becded707220dbdf1a8acf9a895630147fc4e8cd28d0707` |
| `grc_seed13/adapter_config.json` | `1918333d819e6007d3faf17d3497c4f4665f7c0d3cb032f510986cfdb9837942` |
| `grc_seed42/adapter_model.safetensors` | `96d8087e1dbed831ad795fb7a037677eeb7e2d6986a91df61a897b9c1b6d383c` |
| `grc_seed42/adapter_config.json` | `df6c8e4c39f3d7567849a4a32355aea4bb9176c784d089832fe14a8697a2d8f8` |
| `grc_seed73/adapter_model.safetensors` | `d5f90954f3ac2829a213ab7c3b04e6d26b89990c42e95306bc74743d1ac2431c` |
| `grc_seed73/adapter_config.json` | `f1659959d9ffe7348c370e9f01a497066710793c9754e9d06c5d8963aaf45dd1` |

Those hashes pass on the BluePebble restore target. The repaired attempt-D A100 jobs completed all
seven systems for each dataset in one fresh bundle and scored only after generation freeze. All
7,700 outputs, Table 1, paired CI, and the final audit pass. Goal 3 is `COMPLETE / PASS`; Goal 4
has become the unique active goal.

Machine-readable audit: `artifacts/goal3_execution_readiness.json`.
