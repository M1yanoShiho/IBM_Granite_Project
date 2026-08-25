#!/bin/bash
#SBATCH --job-name=exp05-g2-prepare
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/user/work/fl25387/experiment05_goal2/logs/%x-%j.out

set -euo pipefail

repo=/user/work/fl25387/IBM_Granite_Project_latest
goal1=/user/work/fl25387/experiment05_goal1
output_root=${OUTPUT_ROOT:-/user/work/fl25387/experiment05_goal2}
cache=/user/work/fl25387/experiment04_goal2_cache
python=/user/work/fl25387/experiment04_goal3_venv/bin/python

export PYTHONPATH="$repo/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MODEL_CACHE_DIR="$cache"

snapshot() {
  find "$cache/$1/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1
}

reranker=$(snapshot models--ibm-granite--granite-embedding-reranker-english-r2)
provence=$(snapshot models--naver--provence-reranker-debertav3-v1)
selector=$(snapshot models--cross-encoder--nli-deberta-v3-base)
selector_checkpoint=/user/work/fl25387/experiment04_goal3_custom_checkpoints/selector_seed13/model.safetensors
test -n "$reranker" -a -n "$provence" -a -n "$selector" -a -f "$selector_checkpoint"

dataset=${DATASET:?DATASET must be kilt-nq, kilt-tqa, or alce-asqa}
split=${SPLIT:-development}
mkdir -p "$output_root/logs" "$output_root/prepared/$split"
"$python" "$repo/scripts/experiment05_goal2_prepare.py" \
  --dataset "$dataset" \
  --runtime "$goal1/bundles/runtime/$split/$dataset.jsonl" \
  --retrieval "$output_root/retrieval/$split/$dataset.jsonl" \
  --output "$output_root/prepared/$split/$dataset.jsonl" \
  --reranker-snapshot "$reranker" \
  --provence-snapshot "$provence" \
  --selector-snapshot "$selector" \
  --selector-checkpoint "$selector_checkpoint" \
  --device cuda \
  --log-every 10
