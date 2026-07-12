#!/bin/bash
# V4 external validation: the FROZEN alpha=0.6 corroboration blend vs relevance-only on
# RAMDocs (COLM 2025, arXiv:2504.13079) — per-query conflicting-evidence doc sets WE did
# not construct. Answers "did you only solve a problem you manufactured?" with someone
# else's benchmark: correct-doc@1/@3, MRR of first correct doc, misinfo@1 (lower=better),
# + paired significance vs the relevance arm. No index is built (doc sets are given);
# relevance = Granite dense query-doc cosine. Pre-registration: docs/hpc-run-log.md.
#
# FIRST, on the LOGIN node (one-time; compute nodes are offline):
#   mkdir -p /user/work/$USER/ramdocs && wget \
#     https://raw.githubusercontent.com/HanNight/RAMDocs/main/RAMDocs_test.jsonl \
#     -O /user/work/$USER/ramdocs/RAMDocs_test.jsonl
# (Granite dense + 3B are already in the HF cache from previous runs.)
#
# Submit from the project root AFTER `git pull`:
#   mkdir -p logs results && sbatch scripts/run_ramdocs_corroboration.slurm
#   # args: [DATA] [TAG] [ALPHA] [LIMIT]   (LIMIT for a smoke run, e.g. 20)
#   sbatch scripts/run_ramdocs_corroboration.slurm /user/work/$USER/ramdocs/RAMDocs_test.jsonl smoke 0.6 20
#SBATCH --job-name=ramdocs-corrob
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
# 3B extraction, small doc sets; the sole rtx_3090 drains forever -> A100 40GB MIG slice.
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
DATA="${1:-/user/work/$USER/ramdocs/RAMDocs_test.jsonl}"
TAG="${2:-ramdocs}"
ALPHA="${3:-0.6}"
LIMIT="${4:-}"
EXTRA=()
[ -n "$LIMIT" ] && EXTRA=(--limit "$LIMIT")

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
# NOTE: no GRANITE_MODEL_ID override — default 3B is the certified extractor config.

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

python -m eval.ramdocs_eval \
    --data "$DATA" --alpha "$ALPHA" \
    --out-prefix "results/ramdocs_${TAG}" \
    "${EXTRA[@]}"

echo "===== significance vs relevance arm (correct@1 / misinfo@1 / MRR) ====="
python -m eval.significance --per-query-csv "results/ramdocs_${TAG}_correct1.csv" --reference relevance
python -m eval.significance --per-query-csv "results/ramdocs_${TAG}_misinfo1.csv" --reference relevance
python -m eval.significance --per-query-csv "results/ramdocs_${TAG}_mrr_correct.csv" --reference relevance
echo "===== DONE (tag=${TAG}) — record: git add -f results/ramdocs_${TAG}_*.csv ====="
