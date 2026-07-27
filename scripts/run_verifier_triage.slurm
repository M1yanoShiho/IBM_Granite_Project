#!/bin/bash
# Half 1.6 arms D-true + E-granite: the two verifier-triage arms that do not fit
# the CPU box (local/generator-B-verifier-triage-guide-v2.md).
#
#   arm D-true : google/t5_xxl_true_nli_mixture -- 11B params, 42.5GB in fp32.
#                Loaded in bf16 (~21GB), which fits the 3g.40gb MIG slice.
#   arm E      : Granite-as-judge, one fixed prompt, never tuned per arm.
#
# Arms base/large/minicheck already ran on CPU; this job re-runs them is NOT
# needed -- it scores the SAME shared pair set (same --seed, same --n,
# same --multihop), so the cells are head-to-head comparable with the local run.
#
# One-time on the login node before submitting (compute nodes are HF-offline):
#   export HF_HOME=/user/work/$USER/hf_cache
#   hf download google/t5_xxl_true_nli_mixture
#   hf download ibm-granite/granite-4.1-3b
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
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
REPORT="${1:?usage: sbatch run_verifier_triage.slurm REPORT DUMP}"
DUMP="${2:?usage: sbatch run_verifier_triage.slurm REPORT DUMP}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MODEL_CACHE_DIR=/user/work/$USER/hf_cache
export TRIAGE_DATA_DIR=/user/work/$USER/triage_data
export PYTHONUNBUFFERED=1
export GRANITE_MODEL_ID=ibm-granite/granite-4.1-3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

echo "=== Half 1.6 verifier triage: arms true + granite ==="
# --seed / --n / --multihop MUST match the CPU run or the arms stop being
# comparable: the pair set is constructed deterministically from them.
python scripts/verifier_triage.py \
  --arms true,granite \
  --n 150 \
  --multihop 60 \
  --seed 13 \
  --output "$REPORT" \
  --dump "$DUMP"

echo "=== report ==="
cat "$REPORT"
