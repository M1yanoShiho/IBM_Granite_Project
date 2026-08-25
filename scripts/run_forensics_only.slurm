#!/bin/bash
# Re-run answer_chunk_forensics against an EXISTING run directory. No retrieval, no
# generation, no index build -- the run directory is reused as it stands, so this is only
# ever used when the analysis changed and the experiment did not.
#
# It exists as a job rather than a login-node command for one reason: at top_k=1000 the
# forensics parses every line of pipeline_runs.jsonl, which is GB-scale, and the login-node
# rule here is seconds-scale checks only. Two minutes of JSON parsing on login03 is exactly
# the shape of thing that got a cluster-wide warning last time.
#
# Submit from the project root:
#   sbatch scripts/run_forensics_only.slurm \
#     runs/chunk-nq-b-c60o10-topk1000 \
#     results/r18-nq-c60o10-topk1000-classes.jsonl
#SBATCH --job-name=forensics-only
#SBATCH --account=coms039904
#SBATCH --partition=compute
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=uz25020@bristol.ac.uk

set -euo pipefail

RUN_DIR="${1:?usage: sbatch scripts/run_forensics_only.slurm RUN_DIR DUMP.jsonl}"
DUMP="${2:?usage: sbatch scripts/run_forensics_only.slurm RUN_DIR DUMP.jsonl}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MODEL_CACHE_DIR=/user/work/$USER/hf_cache/hub
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export LLM_DEVICE=cpu
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src

# Overwriting the dump in place is intentional: it is a derived artefact of this run
# directory, and keeping a stale copy beside a new one is how two versions of a published
# number start circulating. The previous values stay recoverable from git.
echo "=== forensics: $RUN_DIR -> $DUMP ==="
python scripts/answer_chunk_forensics.py \
  --run "$RUN_DIR" \
  --report "$RUN_DIR/evaluation_report.json" \
  --dump "$DUMP"
