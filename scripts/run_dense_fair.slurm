#!/bin/bash
# Generic FAIR retrieval benchmark on ANY BEIR dataset — pass the dataset name as
# the first argument. Purpose: test whether the enhancements (hybrid, reranker)
# that gave NO gain on the small, near-saturated SciFact/NFCorpus sets DO help on a
# LARGER / HARDER corpus (e.g. fiqa, trec-covid), where reranking gains are
# documented to materialise.
#
#   mkdir -p logs && sbatch scripts/run_dense_fair.slurm fiqa
#   mkdir -p logs && sbatch scripts/run_dense_fair.slurm trec-covid
#
# FIRST, on the LOGIN node (compute nodes are offline) pre-fetch the models + the
# dataset. The models are already cached if you ran SciFact/NFCorpus — only the
# dataset is new (replace fiqa with your dataset on the last line):
#   module load languages/python/3.12.3 && source /user/work/$USER/venv/bin/activate
#   export HF_HOME=/user/work/$USER/hf_cache IR_DATASETS_HOME=/user/work/$USER/ir_datasets
#   for m in ibm-granite/granite-embedding-english-r2 ibm-granite/granite-embedding-small-english-r2 \
#            ibm-granite/granite-embedding-reranker-english-r2 sentence-transformers/all-MiniLM-L6-v2 \
#            thenlper/gte-base intfloat/e5-base-v2 BAAI/bge-base-en-v1.5; do hf download "$m"; done
#   cd /user/work/$USER/IBM_Granite_Project
#   python -c "from eval.benchmarks.loader import load_benchmark; load_benchmark('fiqa')"
#
# For a very large corpus (e.g. trec-covid, 171K docs) the 6h gpu_short cap may be
# tight — switch to '--partition=gpu' and a longer '--time'.
#SBATCH --job-name=granite-fair
#SBATCH --account=coms039904
#SBATCH --partition=gpu_short
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
DATASET="${1:?usage: sbatch scripts/run_dense_fair.slurm <beir-dataset>  (e.g. fiqa, trec-covid)}"

# Compute nodes have no python by default; load the module the venv was built with.
module load languages/python/3.12.3

# Caches on /user/work; offline (models + dataset pre-fetched on the login node).
export HF_HOME=/user/work/$USER/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets
export PYTHONUNBUFFERED=1

source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| torch', torch.__version__)"

# Larger corpora have longer documents than SciFact's abstracts -> token-mode
# chunking caps each chunk at the model's max_seq_length (no silent truncation).
# The index cache is keyed by (dataset, retriever, chunk params), so this is built
# fresh per dataset without touching the SciFact/NFCorpus caches.
python -m eval.run_benchmark \
    --dataset "$DATASET" \
    --retrievers granite_dense granite_small_dense bm25 st_dense gte_dense e5_dense bge_dense \
                 hybrid_granite_bm25 hybrid_granite_small_bm25 \
                 granite_rerank granite_small_rerank hybrid_granite_bm25_rerank \
    --chunk-unit token \
    --out "results/${DATASET}_fair.csv" \
    --per-query-out "results/${DATASET}_per_query.csv" \
    --cache-dir /user/work/$USER/index_cache
