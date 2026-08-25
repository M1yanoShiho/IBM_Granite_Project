#!/usr/bin/env bash
#SBATCH --job-name=exp04-goal4
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=experiment04_goal4-%x-%j.out

set -euo pipefail

: "${EXP04_DATASET:?EXP04_DATASET is required}"
: "${EXP04_RUNTIME:?EXP04_RUNTIME is required}"
: "${EXP04_SIDECAR:?EXP04_SIDECAR is required}"
: "${EXP04_GOAL3_DIR:?EXP04_GOAL3_DIR is required}"
: "${EXP04_ATTEMPT_DIR:?EXP04_ATTEMPT_DIR is required}"
: "${EXP04_ATTEMPT_ID:?EXP04_ATTEMPT_ID is required}"
: "${EXP04_SELECTOR_CHECKPOINT:?EXP04_SELECTOR_CHECKPOINT is required}"
: "${EXP04_ADAPTER13:?EXP04_ADAPTER13 is required}"
: "${EVIDENCE_RAG_ROOT:?EVIDENCE_RAG_ROOT is required}"
: "${EVIDENCE_RAG_VENV:?EVIDENCE_RAG_VENV is required}"
: "${MODEL_CACHE_DIR:?MODEL_CACHE_DIR is required}"

scorer_sidecar="$EXP04_SIDECAR"
unset EXP04_SIDECAR

workspace="$EVIDENCE_RAG_ROOT"
venv="$EVIDENCE_RAG_VENV"
cache="$MODEL_CACHE_DIR"
prepared="$EXP04_ATTEMPT_DIR/prepared_goal4.jsonl"

export MODEL_CACHE_DIR="$cache"
export PYTHONPATH="$workspace/src"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

source "$venv/bin/activate"
cd "$workspace"

python scripts/experiment04_goal4.py preflight \
  --dataset "$EXP04_DATASET" \
  --runtime "$EXP04_RUNTIME" \
  --goal3-dir "$EXP04_GOAL3_DIR" \
  --selector-checkpoint "$EXP04_SELECTOR_CHECKPOINT" \
  --adapter13 "$EXP04_ADAPTER13" \
  --output "$EXP04_ATTEMPT_DIR/preflight.json"

python scripts/experiment04_goal4.py prepare \
  --dataset "$EXP04_DATASET" \
  --runtime "$EXP04_RUNTIME" \
  --goal3-dir "$EXP04_GOAL3_DIR" \
  --prepared "$prepared" \
  --cache-dir "$cache" \
  --selector-checkpoint "$EXP04_SELECTOR_CHECKPOINT" \
  --device cuda

python scripts/experiment04_goal4.py generate-grounded \
  --dataset "$EXP04_DATASET" \
  --runtime "$EXP04_RUNTIME" \
  --goal3-dir "$EXP04_GOAL3_DIR" \
  --prepared "$prepared" \
  --output-dir "$EXP04_ATTEMPT_DIR" \
  --cache-dir "$cache" \
  --device cuda \
  --dtype bfloat16 \
  --adapter13 "$EXP04_ADAPTER13"

python scripts/experiment04_goal4.py generate-direct \
  --dataset "$EXP04_DATASET" \
  --runtime "$EXP04_RUNTIME" \
  --goal3-dir "$EXP04_GOAL3_DIR" \
  --prepared "$prepared" \
  --output-dir "$EXP04_ATTEMPT_DIR" \
  --cache-dir "$cache" \
  --device cuda \
  --dtype bfloat16

python scripts/experiment04_goal4.py freeze \
  --dataset "$EXP04_DATASET" \
  --runtime "$EXP04_RUNTIME" \
  --goal3-dir "$EXP04_GOAL3_DIR" \
  --prepared "$prepared" \
  --output-dir "$EXP04_ATTEMPT_DIR" \
  --attempt-id "$EXP04_ATTEMPT_ID"

python scripts/experiment04_goal4.py score \
  --dataset "$EXP04_DATASET" \
  --runtime "$EXP04_RUNTIME" \
  --goal3-dir "$EXP04_GOAL3_DIR" \
  --prepared "$prepared" \
  --output-dir "$EXP04_ATTEMPT_DIR" \
  --cache-dir "$cache" \
  --sidecar "$scorer_sidecar" \
  --device cuda \
  --dtype bfloat16
