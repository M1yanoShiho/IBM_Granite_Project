#!/bin/bash
# G5: verify-and-annotate vs the delete filter vs baseline.
# (local/guides/verify-annotate-redesign-guide.md; pre-registered in the G5 ledger
# entry, commit 730e14f, BEFORE any arm ran.)
#
# Granite + TRUE co-reside for generation (proven to fit the 40GB slice in G2).
# TRUE is the production verifier and never judges; MiniCheck scores afterwards.
#
#   mkdir -p logs results && sbatch scripts/run_g5_verify_annotate.slurm
#SBATCH --job-name=g5-annotate
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

echo "=== G5: baseline / verify-only / verify-annotate ==="
python scripts/g5_verify_annotate.py \
  --limit 400 --top-k 5 --seed 13 \
  --output-dir results/g5

echo "=== routing stats ==="
cat results/g5/routing-stats.json
