#!/bin/bash
#SBATCH --job-name=exp05-g1-dense
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/user/work/fl25387/experiment05_goal1/logs/%x-%j.out

set -euo pipefail

repo=/user/work/fl25387/IBM_Granite_Project_latest
root=/user/work/fl25387/experiment05_goal1
python="$root/venv/bin/python"
model=$(find /user/work/fl25387/experiment04_goal2_cache/models--ibm-granite--granite-embedding-english-r2/snapshots -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)
test -n "$model"

export PYTHONPATH="$repo/src:/user/work/fl25387/experiment04_goal3_venv/lib/python3.11/site-packages"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

dataset=${DATASET:?DATASET must be kilt or dpr}
"$python" "$repo/scripts/experiment05_goal1_index.py" dense \
  --corpus-dir "$root/corpus/$dataset" \
  --index-dir "$root/indices/$dataset/dense" \
  --model-snapshot "$model" \
  --device cuda \
  --dimension 768 \
  --nlist 8192 \
  --m 96 \
  --nbits 8 \
  --nprobe 64 \
  --train-size 327680 \
  --batch-size 512 \
  --checkpoint-interval 1000000 \
  --seed 13
