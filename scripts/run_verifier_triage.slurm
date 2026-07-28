#!/bin/bash
# Half 1.6 GPU arms: the verifier-triage arms that do not fit the CPU box
# (local/generator-B-verifier-triage-guide-v2.md + local/G1-execution-guide.md).
#
#   arm D-true  : google/t5_xxl_true_nli_mixture -- 11B params, 42.5GB in fp32.
#                 Loaded in bf16 (~21GB).
#   arm E-3b    : Granite-as-judge on granite-4.1-3b  (the currently wired judge).
#   arm E-8b    : Granite-as-judge on granite-4.1-8b  (capacity control -- separates
#                 "LLM-as-a-judge is wrong here" from "3B is too small"). ~16GB bf16.
#   All Granite arms share ONE fixed prompt, never tuned per arm.
#
# Arms base/large/minicheck already ran on CPU; this job does NOT re-run them --
# it scores the SAME shared pair set (same --seed, --n, --multihop), so the cells
# are head-to-head comparable with the local CPU run.
#
# Resource: a FULL a100 (not a MIG slice). granite-4.1-8b needs an a100 in this
# repo (see docs/hpc-run-log.md E1-cascade / E3-2x2, `--gres=gpu:a100:1`), and an
# a100 also holds TRUE (21GB). The three GPU arms load SEQUENTIALLY and each is
# freed before the next (verifier_triage.py `_free_gpu`), so no two large models
# ever co-reside -- one a100 in one job suffices for all three.
#
# One-time on the login node before submitting (compute nodes are HF-offline):
#   export HF_HOME=/user/work/$USER/hf_cache
#   hf download google/t5_xxl_true_nli_mixture
#   hf download ibm-granite/granite-4.1-3b
#   hf download ibm-granite/granite-4.1-8b
#   export TRIAGE_DATA_DIR=/user/work/$USER/triage_data
#   PYTHONPATH=src python -c "import sys; sys.path.insert(0,'scripts'); \
#     import verifier_triage as v; v.ensure_asqa(); v.ensure_2wiki()"
#
# Submit from the project root:
#   mkdir -p logs results && sbatch scripts/run_verifier_triage.slurm \
#     docs/generator/verifier-triage-results-hpc.md results/verifier-triage/scores.jsonl
#SBATCH --job-name=verifier-triage
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
REPORT="${1:?usage: sbatch run_verifier_triage.slurm REPORT DUMP}"
DUMP="${2:?usage: sbatch run_verifier_triage.slurm REPORT DUMP}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MODEL_CACHE_DIR=/user/work/$USER/hf_cache
export TRIAGE_DATA_DIR=/user/work/$USER/triage_data
export PYTHONUNBUFFERED=1
# NOTE: GRANITE_MODEL_ID is deliberately NOT set -- each Granite arm pins its own
# model id (granite3b / granite8b) inside verifier_triage.py, so the arms cannot
# silently collapse onto one ambient model.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

echo "=== Half 1.6 verifier triage: arms true + granite3b + granite8b ==="
# --seed / --n / --multihop MUST match the CPU run or the arms stop being
# comparable: the pair set is constructed deterministically from them.
python scripts/verifier_triage.py \
  --arms true,granite3b,granite8b \
  --n 150 \
  --multihop 60 \
  --seed 13 \
  --output "$REPORT" \
  --dump "$DUMP"

echo "=== report ==="
cat "$REPORT"
