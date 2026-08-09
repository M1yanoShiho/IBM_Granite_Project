#!/bin/bash
# QAMPARI — the Generator's module-level held-out. Five arms, one job.
# Pre-registered in docs/generator/heldout-preregistration.md BEFORE the data was
# loaded (commit c559d5c, 2026-08-09 15:57 +0100).
#
# Loaded once. Results stand whatever they show. No system change follows.
# Chain this on the dry run:
#   sbatch --dependency=afterok:<dryrun> scripts/run_qampari_heldout.slurm
#SBATCH --job-name=qampari-run
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MODEL_CACHE_DIR=/user/work/$USER/hf_cache/hub
export TRIAGE_DATA_DIR=/user/work/$USER/triage_data
export NLTK_DATA=/user/work/$USER/nltk_data
export PYTHONUNBUFFERED=1
export GRANITE_MODEL_ID=ibm-granite/granite-4.1-3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src

python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python scripts/g5_preflight.py --skip-load

echo "=== QAMPARI held-out: five arms, 400 queries, seed 13 ==="
python scripts/g5_verify_annotate.py \
  --dataset qampari \
  --limit 400 --top-k 5 --seed 13 \
  --max-error-rate 0.10 \
  --output-dir results/qampari

echo "=== routing stats ==="
cat results/qampari/routing-stats.json
