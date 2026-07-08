#!/bin/bash
# Finding 12 / Table 4a — ranking-vs-recall decomposition of the NIAH diagnostic.
# Runs one retriever (default granite_dense) over the frozen task at k=10 AND k=100
# (reusing the cached HNSW index), then derives the split:
#   buried 11-100 = found@100 - found@10   (RANKING failure)
#   unreachable   = 1 - found@100          (RECALL  failure)
# "The bottleneck is ranking, not recall" = buried >> unreachable.
#
#   mkdir -p logs results && sbatch scripts/run_niah_diagnostic.slurm results/niah_nq300_frozen.json
#   # sanity: the k=10 number should reproduce the certified dense 0.493 (nq300cert2).
#   # other retriever/tag: sbatch scripts/run_niah_diagnostic.slurm <task.json> q2d_granite nq300q2d
#
# Prefetch on the LOGIN node (one-time): the Granite dense embedder (same as run_niah.slurm).
#SBATCH --job-name=niah-diag
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
TASK="${1:?usage: sbatch scripts/run_niah_diagnostic.slurm <task.json> [retriever] [tag]}"
RETRIEVER="${2:-granite_dense}"
TAG="${3:-nq300}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Same task/retriever/index, two cut-offs. --cache-dir reuses the HNSW index built for the
# cert runs (no doc re-embedding); only the queries are embedded.
for K in 10 100; do
  echo "=================== $RETRIEVER  k=$K ==================="
  python -m eval.run_niah \
      --task "$TASK" --retrievers "$RETRIEVER" --k "$K" \
      --index-type hnsw --cache-dir /user/work/$USER/index_cache \
      --out "results/niah_diag_${TAG}_k${K}.csv" \
      --per-query-out "results/niah_diag_${TAG}_k${K}_perq.csv"
done

echo "=================== Table 4a — ranking vs recall ==================="
python -m eval.niah_diagnostic \
    --k10-csv  "results/niah_diag_${TAG}_k10.csv" \
    --k100-csv "results/niah_diag_${TAG}_k100.csv" \
    --retriever "$RETRIEVER" \
    --out "results/niah_diag_${TAG}_decomposition.csv"

echo "=================== DONE — commit: git add -f results/niah_diag_${TAG}_*.csv ==================="
