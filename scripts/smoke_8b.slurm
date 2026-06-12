#!/bin/bash
# Smoke test: load a Granite model on a GPU node and generate one sentence.
# Submit from the project root (on /user/work):  sbatch scripts/smoke_8b.slurm
# First:  mkdir -p logs
#SBATCH --job-name=granite-smoke
#SBATCH --account=coms039904
#SBATCH --partition=gpu_short          # 6h cap, fast queue — ideal for dev/smoke
#SBATCH --qos=normal
#SBATCH --gres=gpu:1                    # any GPU is fine for the smoke test
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# Caches on /user/work (home quota is small). Offline: the compute node has no
# internet, so weights must be pre-downloaded on the LOGIN node first
# (see docs/hpc-deployment.md, step 4).
export HF_HOME=/user/work/$USER/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Start small; switch to the 8B instruct repo once 3B works:
export GRANITE_MODEL_ID=ibm-granite/granite-4.1-3b

source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

python scripts/smoke_8b.py
