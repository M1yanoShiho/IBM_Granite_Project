#!/bin/bash
# Phase 1 — NIAH scale degradation curve: run ONE retriever config over a grid of
# corpus sizes, rebuilding the SAME needle + distractors against a growing
# background (load_niah_task --max-docs), collecting needle-found@10 / MRR per size.
# The results/niah_scale_<tag>_<N>.csv points ARE the headline figure.
#
#   mkdir -p logs results && sbatch scripts/run_niah_scale.slurm results/niah_nq_task.json granite_dense nq
#   # extend to 21M (fp32 won't fit -> compressed index):
#   IDX=ivfpq SIZES="21000000" sbatch scripts/run_niah_scale.slurm results/niah_nq_task.json granite_dense nq21m
#
# Prefetch on the LOGIN node = same as run_niah.slurm (Granite embedding [+reranker
# if you sweep granite_rerank], SPLADE converted if you sweep splade/hybrid).
#SBATCH --job-name=niah-scale
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
TASK="${1:?usage: sbatch scripts/run_niah_scale.slurm <task.json> [retriever] [tag]}"
RETRIEVER="${2:-granite_dense}"
TAG="${3:-nq}"
IDX="${IDX:-hnsw}"
SIZES="${SIZES:-10000 100000 1000000 5000000}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export SPLADE_MODEL_ID="${SPLADE_MODEL_ID:-/user/work/$USER/splade_st}"
export SPLADE_BATCH_SIZE="${SPLADE_BATCH_SIZE:-32}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# One retriever, a FRESH index per size (each size is a different corpus, so no
# cache reuse across sizes). Smallest-first: if a big size OOMs, the rest are done.
for N in $SIZES; do
  echo "=================== max_docs=$N ($RETRIEVER, $IDX) ==================="
  python -m eval.run_niah \
      --task "$TASK" --retrievers "$RETRIEVER" \
      --max-docs "$N" --k 10 --index-type "$IDX" \
      --out "results/niah_scale_${TAG}_${N}.csv" \
    || echo "size $N failed (likely OOM) — continuing"
done

echo "=================== curve: needle-found@10 vs corpus size ==================="
for N in $SIZES; do
  [ -f "results/niah_scale_${TAG}_${N}.csv" ] && { echo -n "size=$N : "; tail -n +2 "results/niah_scale_${TAG}_${N}.csv"; }
done
