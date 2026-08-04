#!/bin/bash
# G5 scoring: ALCE sentence-level citation metrics + STR-EM over the three arms.
# Generation already ran (job 18258289); this is MiniCheck judging only.
#   sbatch scripts/run_g5_score.slurm
#SBATCH --job-name=g5-score
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
export NLTK_DATA=/user/work/$USER/nltk_data
export PYTHONUNBUFFERED=1
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src

python scripts/g5_score.py \
  --input-dir results/g5 \
  --output-dir results/g5-scored \
  --report docs/generator/g5-verify-annotate.md

echo "=== paired significance ==="
for M in citation_precision citation_recall coverage answer_correctness; do
  for PAIR in "verify-annotate verify-only" "verify-annotate baseline" "verify-only baseline"; do
    set -- $PAIR
    echo "--- $M : $1 vs $2 ---"
    python -m evidence_rag.evaluation.paired_metric_cli \
      --on-report results/g5-scored/$1-report.json \
      --off-report results/g5-scored/$2-report.json \
      --metric $M --seed 13 || true
  done
done
