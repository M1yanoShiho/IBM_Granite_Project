---
name: running-hpc-experiments
description: Use when a task must run on the BluePebble HPC (slurm, sbatch, bp1), when writing code that will execute on the cluster, when preparing a job handoff, or when HPC results are ready to pull back and record. Triggers - 在 HPC 上跑, submit a job, slurm script, pull results back, 拉回结果.
---

# Running HPC Experiments (BluePebble)

## Overview

Every HPC experiment ships as a **triple: code + slurm script + ledger entry**, created together BEFORE submission. A handoff missing any leg is incomplete — "script later" and "record after" are the failure modes this skill exists to prevent.

## Cluster facts (copy these; never re-invent)

| Fact | Value |
|---|---|
| Account / partition / qos | `coms039904` / `gpu` / `normal` |
| GPU | `--gres=gpu:3g.40gb:1` (A100 40GB MIG slice). The sole rtx_3090 node drains forever — do not target it |
| Repo on HPC | `/user/work/$USER/IBM_Granite_Project` (not `~`) |
| Env | `module load languages/python/3.12.3` + `source /user/work/$USER/venv/bin/activate` + offline HF/ir_datasets caches on `/user/work` — copy the env block **verbatim** from `scripts/run_niah_rag.slurm` |
| Scripts / logs | `scripts/run_<name>.slurm` → `logs/%x-%j.out` |
| Models | compute nodes are OFFLINE — `hf download <model>` on the login node first (one-time) |

## The three rules

**1. Code and slurm script are one change.** Code meant for the cluster is incomplete without its `scripts/run_<name>.slurm` in the same commit. Start from the newest similar script (e.g. `scripts/run_niah_rag.slurm`) and keep its skeleton: header comment with submit examples, `#SBATCH` block, `set -euo pipefail`, positional args with defaults, and in-job `echo` of the aggregate CSV + `python -m eval.significance ...` so the `.out` log carries first-read numbers.

**2. Every handoff ends in a copy-paste block** — six parts, real values filled in, no placeholders except `$USER`:

```bash
# local :  git push
# bp1   :  ssh bp1 → cd /user/work/$USER/IBM_Granite_Project && git pull && mkdir -p logs results
# submit:  sbatch scripts/run_<name>.slurm <real args>
# watch :  squeue -u $USER   |   tail -f logs/<job-name>-<jobid>.out   |   sacct -j <jobid>
# record:  git add -f results/<the new files> && git commit -m "results(...): ..." && git push   # on bp1
# local :  git pull   # raw data now version-controlled locally
```

Before writing the submit line, **verify every flag against the tool's actual argparse (or the script's `${1:?usage}` header)** — inventing CLI flags is the #1 observed failure.

**3. No run without a ledger entry** in `docs/hpc-run-log.md`:
- **BEFORE submit:** purpose/hypothesis, expected metric + direction, exact command, git commit hash.
- **AFTER pull:** job id, raw files produced (committed with `git add -f` — `results/` is gitignored), headline numbers + significance output, and the 2-4 sentence finding drafted for `docs/results-summary.md`.

Results that exist only as an `.out` log or an scp'd loose file do not count as recorded.

## Known landmines (all cost real jobs once)

- **Index cache reuse across different corpora** produced an entire invalid certification (fixed by the corpus-fingerprint cache key, f63e905). Reuse `--cache-dir` only for the same task+corpus.
- **Several LLM retrievers in one run** must share one LLM instance (the run_niah/run_rag shared-LLM path) or the job OOMs.
- **torch stays <2.6** (safetensors loading break).
- **Long embedding jobs**: shard + checkpoint to `/user/work` so a walltime kill cannot lose work.
- **GPU contention**: spine experiments first (work-plan rule); background jobs never block them.

## Common mistakes

| Mistake | Fix |
|---|---|
| Inventing CLI flags or dataset names | read the argparse / script usage header first |
| Script under `slurm/`, account left as `<PLACEHOLDER>` | `scripts/`, real `coms039904` |
| scp-only results, raw CSVs never committed | `git add -f results/...` on bp1, pull via git |
| "env: paste from another script yourself" | include the env block verbatim in YOUR script |
| Submitting first, writing the purpose down later | ledger PURPOSE entry first — that is the pre-registration |
