#!/bin/bash
# Held-out plumbing dry run: do the three sets flow through the Generator?
# Schema and record counts only -- no metric is computed and no output inspected.
#
#   sbatch scripts/run_heldout_dryrun.slurm
#SBATCH --job-name=heldout-dry
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MODEL_CACHE_DIR=/user/work/$USER/hf_cache/hub
export HELDOUT_DATA_DIR=/user/work/$USER/heldout_data
export NLTK_DATA=/user/work/$USER/nltk_data
export PYTHONUNBUFFERED=1
export GRANITE_MODEL_ID=ibm-granite/granite-4.1-3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src

python scripts/g5_preflight.py --skip-load

echo "=== held-out plumbing dry run (no metrics) ==="
python scripts/heldout_dryrun.py --slice 3 --top-k 5
