#!/bin/bash
# FAIR retrieval benchmark on NFCorpus — a SECOND dataset, deliberately chosen
# because it is medical IR where BM25 is historically strong (often beats dense).
# Two purposes: (1) break the single-dataset (SciFact-only) limitation — one result
# on one dense-friendly set is an anecdote, not a finding; (2) stress-test the dense
# retrievers and feed the failure-mode analysis (where/why does dense lose to BM25?).
# Same 7 retrievers as the SciFact fair run, incl. the small-Granite efficiency
# check. Writes results/nfcorpus_fair.csv.
#
# Submit from the project root on /user/work:
#   mkdir -p logs && sbatch scripts/run_dense_nfcorpus.slurm
#
# FIRST, on the LOGIN node (internet; compute nodes have none) pre-fetch the
# embedding models + the NFCorpus corpus into the /user/work caches. The models are
# the SAME as the SciFact fair run, so if you ran that they are already cached and
# only the dataset is new:
#   module load languages/python/3.12.3 && source /user/work/$USER/venv/bin/activate
#   export HF_HOME=/user/work/$USER/hf_cache
#   export IR_DATASETS_HOME=/user/work/$USER/ir_datasets
#   hf download ibm-granite/granite-embedding-english-r2
#   hf download ibm-granite/granite-embedding-small-english-r2
#   hf download sentence-transformers/all-MiniLM-L6-v2
#   hf download thenlper/gte-base
#   hf download intfloat/e5-base-v2
#   hf download BAAI/bge-base-en-v1.5
#   cd /user/work/$USER/IBM_Granite_Project
#   python -c "from eval.benchmarks.loader import load_benchmark; load_benchmark('nfcorpus')"
# (the last line caches NFCorpus so the offline job can read it).
#SBATCH --job-name=granite-dense-nfcorpus
#SBATCH --account=coms039904
#SBATCH --partition=gpu_short           # 6h cap, fast queue — NFCorpus is tiny (~3.6K docs)
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
# internet, so the embedding models AND the NFCorpus corpus must be pre-fetched on
# the LOGIN node first (see the header above). The job FAILS fast if a model or the
# dataset is missing (HF_HUB_OFFLINE=1), so don't skip the pre-fetch.
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

# Same 7 retrievers as the SciFact fair run, on NFCorpus. The index cache is keyed
# by (dataset, retriever), so sharing /user/work/$USER/index_cache with the SciFact
# run is safe — NFCorpus indexes get their own "nfcorpus__*" keys, built fresh here.
python -m eval.run_benchmark \
    --dataset nfcorpus \
    --retrievers granite_dense granite_small_dense bm25 st_dense gte_dense e5_dense bge_dense \
    --out results/nfcorpus_fair.csv \
    --per-query-out results/nfcorpus_per_query.csv \
    --cache-dir /user/work/$USER/index_cache
