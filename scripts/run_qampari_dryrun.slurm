#!/bin/bash
# QAMPARI plumbing dry run on GPU: schema, counts, lengths, claim splitting.
# No metric is computed and no generated output is inspected.
#
#   sbatch scripts/run_qampari_dryrun.slurm
#SBATCH --job-name=qampari-dry
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
export TRIAGE_DATA_DIR=/user/work/$USER/triage_data
export NLTK_DATA=/user/work/$USER/nltk_data
export PYTHONUNBUFFERED=1
export GRANITE_MODEL_ID=ibm-granite/granite-4.1-3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src

python scripts/g5_preflight.py --skip-load

echo "=== QAMPARI plumbing dry run (no metrics) ==="
python scripts/qampari_dryrun.py --slice 5 --top-k 5 --probe 20
