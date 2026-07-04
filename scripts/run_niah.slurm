#!/bin/bash
# Baseline NIAH retriever comparison: score granite_dense / splade / convex hybrid
# / +rerank on a built task by needle-found@k + MRR (of the DESIGNATED needle),
# then significance vs granite_dense. This is the "which retrieval design finds the
# needle best" result — the closed loop for raising the system ceiling.
#
#   mkdir -p logs results && sbatch scripts/run_niah.slurm results/niah_nq_task.json nq
#
# FIRST, on the LOGIN node (one-time; compute nodes are offline):
#   source scripts/env.sh
#   bash scripts/convert_splade_safetensors.sh                      # SPLADE .bin -> safetensors
#   hf download ibm-granite/granite-embedding-english-r2            # dense retriever
#   hf download ibm-granite/granite-embedding-reranker-english-r2   # granite_rerank cross-encoder
#SBATCH --job-name=niah-eval
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
TASK="${1:?usage: sbatch scripts/run_niah.slurm <task.json> [tag]}"
TAG="${2:-nq}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export SPLADE_MODEL_ID="${SPLADE_MODEL_ID:-/user/work/$USER/splade_st}"
export SPLADE_BATCH_SIZE="${SPLADE_BATCH_SIZE:-32}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

# Indexes are cached (--cache-dir): the dense arm is built once and reused by
# granite_dense / the convex hybrid / granite_rerank; SPLADE is built once too.
python -m eval.run_niah \
    --task "$TASK" \
    --retrievers granite_dense splade convex_hybrid_granite_splade granite_rerank \
    --k 10 --index-type hnsw \
    --out "results/niah_eval_${TAG}.csv" \
    --per-query-out "results/niah_eval_${TAG}_per_query.csv" \
    --cache-dir /user/work/$USER/index_cache

echo "=================== needle-found@10 + MRR per retriever ==================="
cat "results/niah_eval_${TAG}.csv"
echo "=================== significance vs granite_dense ==================="
python -m eval.significance --per-query-csv "results/niah_eval_${TAG}_per_query.csv" --reference granite_dense
echo "=================== DONE ==================="
