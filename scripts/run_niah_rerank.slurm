#!/bin/bash
# Finding 13 — reranking is null on the NIAH task (reproduce Table 4b's reranking row + COMMIT it).
# One run_niah over the frozen task compares the query-transforms (q2d / hyde) against every
# reranker on the SAME first stage: pointwise cross-encoder (granite_rerank, and stacked on q2d /
# hyde) and listwise-LLM (granite_listrank). Then significance vs q2d and vs dense shows the
# rerankers give NO significant gain while the query-transforms do. One shared LLMClient across
# q2d / hyde / listrank (OOM-safe); the cached HNSW index is reused (no doc re-embed).
#
#   mkdir -p logs results && sbatch scripts/run_niah_rerank.slurm results/niah_nq300_frozen.json
#   # drop the SLOW listwise arm if walltime is tight (granite_listrank = many LLM calls/query):
#   RETRIEVERS="granite_dense q2d_granite granite_rerank q2d_granite_rerank hyde_granite_rerank" \
#       sbatch scripts/run_niah_rerank.slurm results/niah_nq300_frozen.json nq300rr_xlistwise
#
# Prefetch on the LOGIN node: Granite dense embedder + the cross-encoder reranker
# (ibm-granite/granite-embedding-reranker-english-r2) + the generator LLM (granite-4.1-3b).
#SBATCH --job-name=niah-rerank
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
TASK="${1:?usage: sbatch scripts/run_niah_rerank.slurm <task.json> [tag]}"
TAG="${2:-nq300rr}"
RETRIEVERS="${RETRIEVERS:-granite_dense q2d_granite granite_rerank q2d_granite_rerank hyde_granite_rerank granite_listrank}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

echo "=================== reranking comparison (needle-found@10) ==================="
echo "retrievers: $RETRIEVERS"
# run_niah auto-loads ONE shared LLMClient (retrievers_need_llm) for q2d/hyde/listrank.
# granite_listrank (listwise) is the slow arm — many LLM calls/query (why the original was n=100).
python -m eval.run_niah \
    --task "$TASK" --retrievers $RETRIEVERS --k 10 \
    --index-type hnsw --cache-dir /user/work/$USER/index_cache \
    --out "results/niah_rerank_${TAG}.csv" \
    --per-query-out "results/niah_rerank_${TAG}_perq.csv"

echo "=================== significance vs q2d_granite (rerankers should be ns / negative) ==================="
python -m eval.significance --per-query-csv "results/niah_rerank_${TAG}_perq.csv" --reference q2d_granite

echo "=================== significance vs granite_dense (q2d/hyde should be sig) ==================="
python -m eval.significance --per-query-csv "results/niah_rerank_${TAG}_perq.csv" --reference granite_dense

echo "=================== DONE — commit: git add -f results/niah_rerank_${TAG}*.csv ==================="
echo "(reproduces finding 13 / Table 4b reranking-null, upgraded to n=300: q2d/hyde beat dense,"
echo " no reranker beats q2d — pointwise cross-encoder, q2d/hyde-stacked, or listwise)"
