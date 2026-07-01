#!/bin/bash
# Phase 2: the SPLADE learned-sparse arm vs the weak BM25 arm, in the convex hybrid.
# Per dataset:
#   1) tune alpha for granite_dense + SPLADE on dev/train (eval.tune_alpha --lexical splade)
#   2) run granite_dense, bm25, splade, convex_hybrid_granite_bm25, convex_hybrid_granite_splade
#      at that tuned alpha, with --per-query-out, then significance.
#
# Both convex hybrids share the SPLADE-tuned alpha, so convex_splade vs convex_bm25 is a
# CONTROLLED comparison that isolates arm quality (BM25 vs SPLADE) at a fixed fusion weight.
# The deployment BM25 hybrid (at its own tuned alpha) is the Phase 1 result
# (results/<d>_convex.csv); this run's standalone `splade` vs `bm25` gives the arm-quality
# comparison directly.
#
#   mkdir -p logs && sbatch scripts/run_splade_hybrid.slurm
#
# FIRST, on the LOGIN node (compute nodes are offline), one-time setup:
#   source scripts/env.sh
#   bash scripts/convert_splade_safetensors.sh    # SPLADE .bin -> safetensors
#                                                  # (HPC torch 2.5.1 < 2.6 blocks torch.load)
#   python -c "from eval.benchmarks.loader import load_benchmark as L; \
#     [L(d, split=s) for d, s in [('scifact','train'),('nfcorpus','dev'),('fiqa','dev')]]"
#SBATCH --job-name=splade-hybrid
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
# SPLADE weights loaded from the locally-converted safetensors copy (see
# scripts/convert_splade_safetensors.sh) — the HPC's torch 2.5.1 cannot torch.load .bin.
export SPLADE_MODEL_ID="${SPLADE_MODEL_ID:-/user/work/$USER/splade_st}"
# SPLADE encodes the corpus in batches of SPLADE_BATCH_SIZE (default 32); lower it (e.g. 8)
# if the GPU OOMs. expandable_segments reduces CUDA fragmentation.
export SPLADE_BATCH_SIZE="${SPLADE_BATCH_SIZE:-32}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| torch', torch.__version__)"

CACHE=/user/work/$USER/index_cache

# dataset : tune-split : chunk-unit. chunk-unit is the DENSE arm's (SPLADE is doc-level);
# match each set's audited run so the alpha=1 point reproduces the published granite_dense.
for SPEC in "scifact:train:word" "nfcorpus:dev:word" "fiqa:dev:token"; do
    D="${SPEC%%:*}"; REST="${SPEC#*:}"; TUNE="${REST%%:*}"; UNIT="${REST##*:}"
    echo "=== $D (tune granite+splade on $TUNE, dense chunk-unit $UNIT) ==="

    python -m eval.tune_alpha \
        --dataset "$D" \
        --lexical splade \
        --tune-split "$TUNE" \
        --test-split test \
        --chunk-unit "$UNIT" \
        --cache-dir "$CACHE" \
        --out "results/splade_alpha_curve_${D}.csv" \
        --best-alpha-out "results/splade_best_alpha_${D}.txt"

    ALPHA="$(cat "results/splade_best_alpha_${D}.txt")"
    echo "tuned alpha (granite+splade) for $D = $ALPHA"

    python -m eval.run_benchmark \
        --dataset "$D" \
        --split test \
        --retrievers granite_dense bm25 splade \
                     convex_hybrid_granite_bm25 convex_hybrid_granite_splade \
        --alpha "$ALPHA" \
        --chunk-unit "$UNIT" \
        --out "results/${D}_splade.csv" \
        --per-query-out "results/${D}_splade_per_query.csv" \
        --cache-dir "$CACHE"

    echo "--- significance: vs granite_dense (convex_splade beats dense?) ---"
    python -m eval.significance --per-query-csv "results/${D}_splade_per_query.csv" --reference granite_dense
    echo "--- significance: vs convex_hybrid_granite_bm25 (strong vs weak arm, same alpha) ---"
    python -m eval.significance --per-query-csv "results/${D}_splade_per_query.csv" --reference convex_hybrid_granite_bm25
    echo "--- significance: vs bm25 (standalone splade vs bm25) ---"
    python -m eval.significance --per-query-csv "results/${D}_splade_per_query.csv" --reference bm25
done

echo "DONE. Per dataset: results/splade_alpha_curve_<d>.csv (curve),"
echo "results/<d>_splade.csv (headline), results/<d>_splade_per_query.csv (significance)."
echo "Sanity: the alpha=1.0 row of each curve must still reproduce published granite_dense"
echo "(0.767 / 0.375 / 0.459 for scifact / nfcorpus / fiqa)."
