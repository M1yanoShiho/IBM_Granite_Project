#!/bin/bash
#SBATCH --job-name=exp05-rapid-retr
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
module load languages/java-sdk/22.0.2

repo=/user/work/fl25387/IBM_Granite_Project_latest
goal1=/user/work/fl25387/experiment05_goal1
output_root=${OUTPUT_ROOT:-/user/work/fl25387/experiment05_goal2}
python="$goal1/venv/bin/python"
model=$(find /user/work/fl25387/experiment04_goal2_cache/models--ibm-granite--granite-embedding-english-r2/snapshots -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)
test -n "$model"

export PYTHONPATH="$repo/src:/user/work/fl25387/experiment04_goal3_venv/lib/python3.11/site-packages"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

dataset=${DATASET:?DATASET must be kilt-nq, kilt-tqa, or alce-asqa}
split=${SPLIT:-development}
case "$dataset" in
  kilt-nq|kilt-tqa) corpus=kilt ;;
  alce-asqa) corpus=dpr ;;
  *) echo "unsupported dataset: $dataset" >&2; exit 2 ;;
esac

mkdir -p "$output_root/logs" "$output_root/retrieval/$split"
"$python" "$repo/scripts/experiment05_rapid_retrieval.py" \
  --dataset "$dataset" \
  --runtime "$goal1/bundles/runtime/$split/$dataset.jsonl" \
  --bm25-index "$goal1/indices/$corpus/bm25" \
  --model-snapshot "$model" \
  --output "$output_root/retrieval/$split/$dataset.jsonl" \
  --manifest "$output_root/retrieval/$split/$dataset.manifest.json" \
  --candidate-pool-size 1000 \
  --final-top-k 10 \
  --reranker-pool-size 40 \
  --rrf-k 60 \
  --dense-batch-size "${DENSE_BATCH_SIZE:-512}" \
  --device cuda
