#!/bin/bash
#SBATCH --job-name=exp05-g2-smoke
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/user/work/fl25387/experiment05_goal2/logs/%x-%j.out

set -euo pipefail

repo=/user/work/fl25387/IBM_Granite_Project_latest
goal1=/user/work/fl25387/experiment05_goal1
goal2=/user/work/fl25387/experiment05_goal2
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

granite=$(snapshot models--ibm-granite--granite-4.1-3b)
true_model=$(snapshot models--google--t5_xxl_true_nli_mixture)
adapter13=/user/work/fl25387/experiment04_goal3_custom_checkpoints/grc_seed13
adapter42=/user/work/fl25387/experiment04_goal3_custom_checkpoints/grc_seed42
adapter73=/user/work/fl25387/experiment04_goal3_custom_checkpoints/grc_seed73
test -n "$granite" -a -n "$true_model"
test -f "$adapter13/adapter_model.safetensors" -a -f "$adapter42/adapter_model.safetensors" -a -f "$adapter73/adapter_model.safetensors"

dataset=${DATASET:?DATASET is required}
mode=${MODE:?MODE must be direct or grounded}
out="$goal2/smoke/$dataset"
mkdir -p "$goal2/logs" "$out"

common=(
  --mode "$mode"
  --dataset "$dataset"
  --runtime "$goal1/bundles/runtime/development/$dataset.jsonl"
  --prepared "$goal2/prepared/development/$dataset.jsonl"
  --output-dir "$out"
  --artifact-root "$goal2/sealed_evidence/smoke/$dataset/$mode"
  --model-snapshot "$granite"
  --limit 5
  --log-every 1
)

if [ "$mode" = direct ]; then
  arms=(bm25_rag hybrid_rag granite_rerank_rag provence_rag ablation_direct_generator)
elif [ "$mode" = grounded ]; then
  arms=(ours_seed13 ours_seed42 ours_seed73 ablation_bm25_retriever ablation_no_selector)
  common+=(
    --true-snapshot "$true_model"
    --adapter13 "$adapter13"
    --adapter42 "$adapter42"
    --adapter73 "$adapter73"
  )
else
  echo "unsupported MODE: $mode" >&2
  exit 2
fi

"$python" "$repo/scripts/experiment05_generate.py" "${common[@]}" --arms "${arms[@]}"
