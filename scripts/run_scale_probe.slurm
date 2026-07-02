#!/bin/bash
# SCALE-FEASIBILITY probe (NIAH spec §7): how the Granite dense HNSW index's BUILD
# time + PEAK memory grow with corpus size, to set the honest "scale ceiling" for
# the headline. Sweeps --max-docs and records /usr/bin/time -v (Elapsed + Maximum
# resident set size). Queries are capped (--max-queries) so the measured time is
# ~ the index-build cost, and the cache is deliberately OFF (a cache hit would skip
# the build we are timing).
#
#   mkdir -p logs results && sbatch scripts/run_scale_probe.slurm nq dev
#
# FIRST, on the LOGIN node (compute nodes are offline) pre-fetch model + dataset:
#   module load languages/python/3.12.3 && source /user/work/$USER/venv/bin/activate
#   export HF_HOME=/user/work/$USER/hf_cache IR_DATASETS_HOME=/user/work/$USER/ir_datasets
#   hf download ibm-granite/granite-embedding-english-r2
#   cd /user/work/$USER/IBM_Granite_Project
#   python -c "from eval.benchmarks.loader import load_benchmark; load_benchmark('nq','dev',max_queries=1,max_docs=1000)"
#SBATCH --job-name=niah-scale-probe
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G                   # ~21M x 768-d fp32 ≈ 64GB vectors + HNSW graph
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
DATASET="${1:?usage: sbatch scripts/run_scale_probe.slurm <dataset> [split]}"
SPLIT="${2:-dev}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

# Smallest-first: if a larger size OOMs or times out, the smaller results are
# already logged (|| true keeps the sweep going). The largest size that completes
# within this allocation is the honest scale ceiling for the headline.
for N in 1000000 5000000 21000000; do
  echo "=================== max_docs=$N ==================="
  /usr/bin/time -v python -m eval.run_benchmark \
      --dataset "$DATASET" --split "$SPLIT" \
      --retrievers granite_dense \
      --index-type hnsw --chunk-unit token \
      --max-docs "$N" --max-queries 100 \
      --out "results/scale_probe_${DATASET}_${N}.csv" \
    2>&1 | grep -E "Corpus:|Elapsed \(wall|Maximum resident|max_docs=" || true
done
echo "=================== DONE ==================="
echo "Per max_docs, read: Elapsed (wall clock) = build+eval time; Maximum resident set size = peak RAM."
echo "Largest max_docs that completed within the allocation = the scale ceiling."
