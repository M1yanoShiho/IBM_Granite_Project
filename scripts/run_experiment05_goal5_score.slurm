#!/bin/bash
#SBATCH --job-name=exp05-g5-score
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
minicheck=$(snapshot models--lytang--MiniCheck-Flan-T5-Large)
dataset=${DATASET:?DATASET is required}
arm=${ARM:?ARM is required}
phase=${PHASE:?PHASE must be goal3 or goal4}

generation="$runroot/$phase/outputs/$dataset/generations/$arm.jsonl"
out="$runroot/goal5/scoring/$dataset/$arm"
mkdir -p "$out"
"$python" "$repo/scripts/experiment05_score_arm.py" \
  --dataset "$dataset" \
  --arm "$arm" \
  --runtime "$goal1/bundles/runtime/formal/$dataset.jsonl" \
  --sidecar "$goal1/bundles/scorer_only/formal/$dataset.jsonl" \
  --generation "$generation" \
  --scores "$out/query_scores.jsonl" \
  --claim-traces "$out/claim_traces.jsonl" \
  --aggregate "$out/aggregate.json" \
  --granite-snapshot "$granite" \
  --minicheck-snapshot "$minicheck" \
  --log-every 10
