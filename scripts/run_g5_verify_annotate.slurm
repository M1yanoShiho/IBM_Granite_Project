#!/bin/bash
# G7: five arms -- baseline / verify-only / verify-annotate-{capped,open,nogate}.
# (Pre-registered in the G7 ledger entry, hpc-run-log, BEFORE any arm ran.)
#
# Two ablations are carried, and neither may be dropped: the capped arm attributes
# G6's effect to the contract lift itself rather than to the verification filter,
# and the open arm is G7's control -- identical routing with the entity gate still
# active, so gate removal is what nogate-vs-open measures.
#
# All five run in ONE job. Cross-run comparison has produced two separate confounds
# in this project already; a few extra GPU hours is cheap insurance against a third.
#
# Granite + TRUE co-reside for generation (proven to fit the 40GB slice in G2).
# TRUE is the production verifier and never judges; MiniCheck scores afterwards.
#
# Run the preflight on the LOGIN NODE first -- a G6 attempt spent 76 GPU-minutes
# discovering that TRUE's weights were not in the offline cache:
#   PYTHONPATH=src python scripts/g5_preflight.py
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

# Cheap cache check even though the login-node preflight should already have run:
# `set -e` makes a missing model stop the job in seconds rather than after 400
# cases of swallowed OSErrors.
echo "=== preflight ==="
python scripts/g5_preflight.py --skip-load

echo "=== G7: baseline / verify-only / annotate-capped / annotate-open / annotate-nogate ==="
# --max-error-rate: an arm that mostly failed must produce NO result files. The
# previous attempt wrote files from 35 empty records and exited 0.
python scripts/g5_verify_annotate.py \
  --limit 400 --top-k 5 --seed 13 \
  --max-error-rate 0.10 \
  --output-dir results/g5

echo "=== routing stats ==="
cat results/g5/routing-stats.json
