#!/bin/bash
# Phase 1: convex-combination hybrid (granite_dense + BM25) vs the RRF hybrid and
# pure dense, on the BEIR sets where RRF lost (finding #4). For each dataset:
#   1) tune alpha on its dev/train split + write the test alpha-curve (eval.tune_alpha)
#   2) run granite_dense, bm25, RRF hybrid, and the convex hybrid (at the tuned alpha)
#      with --per-query-out, so significance (convex vs dense, convex vs RRF) follows.
#
#   mkdir -p logs && sbatch scripts/run_convex_hybrid.slurm
#
# FIRST, on the LOGIN node (compute nodes are offline) pre-fetch the datasets' splits
# (models are already cached; BM25 needs none). dev/train share the same corpus as
# test, so this only adds the dev/train queries+qrels:
#   source scripts/env.sh
#   python -c "from eval.benchmarks.loader import load_benchmark as L; \
#     [L(d, split=s) for d, s in [('scifact','train'),('nfcorpus','dev'),('fiqa','dev')]]"
#SBATCH --job-name=convex-hybrid
#SBATCH --account=coms039904
#SBATCH --partition=gpu_short
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# Compute nodes have no python by default; load the module the venv was built with.
module load languages/python/3.12.3

export HF_HOME=/user/work/$USER/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets
export PYTHONUNBUFFERED=1

source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| torch', torch.__version__)"

CACHE=/user/work/$USER/index_cache

# dataset : tune-split : chunk-unit. SciFact/NFCorpus were audited in word mode,
# FiQA in token mode — match each so the alpha=1 point reproduces the published
# granite_dense nDCG@10 (a consistency check).
for SPEC in "scifact:train:word" "nfcorpus:dev:word" "fiqa:dev:token"; do
    D="${SPEC%%:*}"; REST="${SPEC#*:}"; TUNE="${REST%%:*}"; UNIT="${REST##*:}"
    echo "=== $D (tune on $TUNE, chunk-unit $UNIT) ==="

    python -m eval.tune_alpha \
        --dataset "$D" \
        --tune-split "$TUNE" \
        --test-split test \
        --chunk-unit "$UNIT" \
        --cache-dir "$CACHE" \
        --out "results/convex_alpha_curve_${D}.csv" \
        --best-alpha-out "results/convex_best_alpha_${D}.txt"

    ALPHA="$(cat "results/convex_best_alpha_${D}.txt")"
    echo "tuned alpha for $D = $ALPHA"

    python -m eval.run_benchmark \
        --dataset "$D" \
        --split test \
        --retrievers granite_dense bm25 hybrid_granite_bm25 convex_hybrid_granite_bm25 \
        --alpha "$ALPHA" \
        --chunk-unit "$UNIT" \
        --out "results/${D}_convex.csv" \
        --per-query-out "results/${D}_convex_per_query.csv" \
        --cache-dir "$CACHE"
done

echo "DONE. Per dataset: results/convex_alpha_curve_<d>.csv (curve),"
echo "results/<d>_convex.csv (headline), results/<d>_convex_per_query.csv (significance)."
