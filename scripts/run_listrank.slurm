#!/bin/bash
# Listwise LLM reranker (A4) — retest the "reranking fails" finding with a
# RankGPT-style LISTWISE reranker (the project's own Granite LLM) instead of the
# pointwise cross-encoder. Per dataset, compare on nDCG@10:
#   granite_dense (first stage) -> granite_rerank (cross-encoder) -> granite_listrank (listwise)
# so we see whether a listwise reranker helps where the cross-encoder did not
# (finding #5 was "reranking neutral/negative" — but only a pointwise cross-encoder
# on near-saturated sets was tried, so that conclusion is dataset-/method-limited).
#
# Cost: the listwise reranker makes ~ceil(pool/step) Granite generations per query.
# --rerank-pool 20 with the default window (20) => ONE generation per query (cheap);
# raise RERANK_POOL (e.g. 100) for deeper sliding-window reranking at more cost.
#
# ONE-TIME on the LOGIN node (compute nodes are offline). Prefetch the generator +
# both retrieval models + the dataset:
#   source scripts/env.sh
#   for m in ibm-granite/granite-4.1-8b \
#            ibm-granite/granite-embedding-english-r2 \
#            ibm-granite/granite-embedding-reranker-english-r2; do hf download "$m"; done
#   python -c "from eval.benchmarks.loader import load_benchmark as L; L('fiqa', split='test')"
#
# Then submit from the project root:
#   mkdir -p logs && sbatch scripts/run_listrank.slurm
#   # deeper pool:  RERANK_POOL=100 sbatch scripts/run_listrank.slurm
#SBATCH --job-name=granite-listrank
#SBATCH --account=coms039904
#SBATCH --partition=gpu                 # 14-day cap; loads the 8B as the reranker LLM
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_3090:1           # 24GB, fits 8B bf16 + the dense/cross-encoder models
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# Compute nodes have no python by default — load the module the venv was built with.
module load languages/python/3.12.3

export HF_HOME=/user/work/$USER/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets
export GRANITE_MODEL_ID=ibm-granite/granite-4.1-8b   # instruct variant: follows the ranking prompt
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| torch', torch.__version__)"

CACHE=/user/work/$USER/index_cache

# Candidate pool the LLM reranker sees. 20 == the default window => 1 generation/query.
POOL="${RERANK_POOL:-20}"

# Hard-ish sets where first-stage recall still has room (unlike the saturated small
# sets where the cross-encoder showed no gain). Add more as needed.
for D in fiqa; do
    echo "===== listwise rerank: dataset=$D (rerank-pool=$POOL) ====="
    python -m eval.run_benchmark \
        --dataset "$D" \
        --split test \
        --retrievers granite_dense granite_rerank granite_listrank \
        --rerank-pool "$POOL" \
        --out "results/${D}_listrank.csv" \
        --per-query-out "results/${D}_listrank_per_query.csv" \
        --cache-dir "$CACHE"

    echo "--- significance: listwise vs dense first stage (does reranking help at all?) ---"
    python -m eval.significance --per-query-csv "results/${D}_listrank_per_query.csv" --reference granite_dense
    echo "--- significance: listwise vs cross-encoder rerank (listwise > pointwise?) ---"
    python -m eval.significance --per-query-csv "results/${D}_listrank_per_query.csv" --reference granite_rerank
done

echo "DONE. results/<d>_listrank.csv (nDCG table), results/<d>_listrank_per_query.csv (significance)."
# If the GPU OOMs with all three models co-resident, drop granite_rerank from
# --retrievers (compare granite_dense vs granite_listrank only); the 8B is the heavy one.
