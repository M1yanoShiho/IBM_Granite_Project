#!/bin/bash
# FAIR retrieval benchmark on a GPU node: Granite vs BM25 vs the modern, same-class
# dense baselines (gte / e5 / bge), plus a small-Granite efficiency check
# (granite_small_dense, ~47M) — so the comparison no longer rests on the old, small
# MiniLM (st_dense) alone. granite_dense + bm25 + st_dense are kept too, so their
# rows reproduce the original table (a harness sanity check) in the same CSV.
#
# Writes a SEPARATE file (results/scifact_fair.csv) so the historical 3-retriever
# results/benchmark_results.csv is preserved for comparison.
#
# Submit from the project root on /user/work:
#   mkdir -p logs && sbatch scripts/run_dense_scifact_fair.slurm
#
# FIRST, on the LOGIN node (it has internet; compute nodes do NOT) pre-fetch the
# embedding models AND the SciFact corpus into the /user/work caches:
#   module load languages/python/3.12.3 && source /user/work/$USER/venv/bin/activate
#   export HF_HOME=/user/work/$USER/hf_cache
#   export IR_DATASETS_HOME=/user/work/$USER/ir_datasets
#   hf download ibm-granite/granite-embedding-english-r2         # cached from earlier runs
#   hf download ibm-granite/granite-embedding-small-english-r2   # NEW (~47M small Granite = efficiency test)
#   hf download sentence-transformers/all-MiniLM-L6-v2           # cached from earlier runs
#   hf download thenlper/gte-base                                # cached from the first fair run
#   hf download intfloat/e5-base-v2                              # cached from the first fair run
#   hf download BAAI/bge-base-en-v1.5                            # cached from the first fair run
#   cd /user/work/$USER/IBM_Granite_Project
#   python -c "from eval.benchmarks.loader import load_benchmark; load_benchmark('scifact')"
# (the last line caches SciFact so the offline job can read it).
#SBATCH --job-name=granite-dense-fair
#SBATCH --account=coms039904
#SBATCH --partition=gpu_short           # 6h cap, fast queue — embedding 5 small models is minutes
#SBATCH --qos=normal
#SBATCH --gres=gpu:1                     # any GPU; all embedding models are small (<1GB each)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# Compute nodes have no python by default — load the same module the venv was
# built with before activating it.
module load languages/python/3.12.3

# Caches on /user/work (home quota is small). Offline: the compute node has no
# internet, so the embedding models AND the SciFact corpus must be pre-fetched on
# the LOGIN node first (see the header above). The job FAILS fast if a model is
# missing (HF_HUB_OFFLINE=1), so don't skip the pre-fetch.
export HF_HOME=/user/work/$USER/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets
export PYTHONUNBUFFERED=1                # stream stdout to the log live

source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

# Sanity: embedding silently falls back to CPU (and may time out) on a +cpu torch
# build — print whether the GPU is actually visible here.
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| torch', torch.__version__)"

# Seven retrievers, one CSV. granite_dense/bm25/st_dense reproduce the old numbers;
# gte/e5/bge are the fair modern peers; granite_small_dense (~47M) is the efficiency
# test. Reuses the existing index_cache, so only the new small-Granite index builds.
python -m eval.run_benchmark \
    --dataset scifact \
    --retrievers granite_dense granite_small_dense bm25 st_dense gte_dense e5_dense bge_dense \
    --out results/scifact_fair.csv \
    --per-query-out results/scifact_per_query.csv \
    --cache-dir /user/work/$USER/index_cache
