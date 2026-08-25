#!/bin/bash
# Experiment 04 Goal 2 only: four real baseline arms on one revealed synthetic case.
# No held-out bundle or formal output directory is accepted by this job.
#SBATCH --job-name=exp04-g2-base
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

WORKSPACE="/user/work/$USER/experiment04_goal2_workspace"
VENV="/user/work/$USER/experiment04_goal2_venv"
CACHE="/user/work/$USER/experiment04_goal2_cache"
OUTPUT="/user/work/$USER/experiment04_goal2_outputs/real-baselines"

module load languages/python/3.11.15
source "$VENV/bin/activate"
cd "$WORKSPACE"
export PYTHONPATH=src
export HF_HOME="$CACHE"
export MODEL_CACHE_DIR="$CACHE"
export NLTK_DATA="/user/home/$USER/nltk_data"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.__version__)"
python scripts/experiment04_goal2_server_baselines.py \
  --cache-dir "$CACHE" \
  --output-dir "$OUTPUT"
