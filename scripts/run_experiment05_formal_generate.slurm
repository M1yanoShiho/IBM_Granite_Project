#!/bin/bash
#SBATCH --job-name=exp05-formal-gen
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=/user/work/fl25387/experiment05_runs/logs/%x-%j.out

set -euo pipefail

repo=/user/work/fl25387/IBM_Granite_Project_latest
goal1=/user/work/fl25387/experiment05_goal1
runroot=/user/work/fl25387/experiment05_runs
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

dataset=${DATASET:?DATASET is required}
arm=${ARM:?ARM is required}
phase=${PHASE:?PHASE must be goal3 or goal4}
case "$phase" in goal3|goal4) ;; *) echo "unsupported PHASE: $phase" >&2; exit 2 ;; esac

case "$arm" in
  bm25_rag|hybrid_rag|granite_rerank_rag|provence_rag|ablation_direct_generator)
    mode=direct
    ;;
  ours_seed13|ours_seed42|ours_seed73|ablation_bm25_retriever|ablation_no_selector)
    mode=grounded
    ;;
  *) echo "unsupported ARM: $arm" >&2; exit 2 ;;
esac

out="$runroot/$phase/outputs/$dataset"
mkdir -p "$runroot/logs" "$out"
common=(
  --mode "$mode"
  --dataset "$dataset"
  --arms "$arm"
  --runtime "$goal1/bundles/runtime/formal/$dataset.jsonl"
  --prepared "$runroot/prepared/formal/$dataset.jsonl"
  --output-dir "$out"
  --artifact-root "$runroot/sealed_evidence/formal/$dataset/$arm"
  --model-snapshot "$granite"
  --log-every 10
)

if [ "$mode" = grounded ]; then
  common+=(
    --true-snapshot "$true_model"
    --adapter13 "$adapter13"
    --adapter42 "$adapter42"
    --adapter73 "$adapter73"
  )
fi

"$python" "$repo/scripts/experiment05_generate.py" "${common[@]}"
