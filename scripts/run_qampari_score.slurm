#!/bin/bash
# QAMPARI scoring: ALCE sentence-level citation metrics + the pre-registered
# correctness metric (answer recall by containment, uncapped and @5).
# Judge is MiniCheck; TRUE is the production verifier and never judges.
#
#   sbatch --dependency=afterok:<run> scripts/run_qampari_score.slurm
#SBATCH --job-name=qampari-score
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MODEL_CACHE_DIR=/user/work/$USER/hf_cache/hub
export TRIAGE_DATA_DIR=/user/work/$USER/triage_data
export NLTK_DATA=/user/work/$USER/nltk_data
export PYTHONUNBUFFERED=1
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src

python scripts/g5_score.py \
  --dataset qampari \
  --input-dir results/qampari \
  --output-dir results/qampari-scored \
  --report docs/generator/qampari-heldout.md

echo "=== paired significance ==="
for ARM in baseline verify-only verify-annotate-capped verify-annotate-open \
           verify-annotate-nogate; do
  test -f "results/qampari-scored/$ARM-report.json" \
    || { echo "missing report for arm '$ARM'" >&2; exit 1; }
done

# The pre-registered criterion is the FIRST pair, on the first two metrics.
for M in citation_precision coverage citation_recall answer_correctness answer_recall_at5; do
  for PAIR in "verify-annotate-nogate baseline" \
              "verify-annotate-nogate verify-annotate-open" \
              "verify-annotate-nogate verify-only" \
              "verify-annotate-open verify-annotate-capped" \
              "verify-only baseline"; do
    set -- $PAIR
    echo "--- $M : $1 vs $2 ---"
    python -m evidence_rag.evaluation.paired_metric_cli \
      --on-report results/qampari-scored/$1-report.json \
      --off-report results/qampari-scored/$2-report.json \
      --metric $M --seed 13
  done
done
