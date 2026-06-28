#!/bin/bash
# SCALE demonstration: Granite dense retrieval on a LARGE corpus, FLAT (exact)
# vs HNSW (ANN). Shows the ANN index keeps retrieval quality (nDCG@10 ~ unchanged)
# while slashing query latency (ms_per_query) — i.e. it finds the needle in a big
# haystack just as well, but fast enough to be practical at millions of documents.
# This is the EFFICIENCY-axis optimization (not nDCG), so it's a near-sure win.
#
# Only granite_dense is run (the full multi-retriever sweep is infeasible at
# millions of docs — keep that for moderate sets via run_dense_fair.slurm).
#
#   mkdir -p logs && sbatch scripts/run_scale_demo.slurm <dataset> [split]
#   e.g.  sbatch scripts/run_scale_demo.slurm dbpedia-entity        # 4.6M docs, test split
#         sbatch scripts/run_scale_demo.slurm msmarco dev           # 8.8M docs, dev split
#
# FIRST, on the LOGIN node (compute nodes are offline) pre-fetch the model + the
# dataset (replace dbpedia-entity / its split as needed):
#   module load languages/python/3.12.3 && source /user/work/$USER/venv/bin/activate
#   export HF_HOME=/user/work/$USER/hf_cache IR_DATASETS_HOME=/user/work/$USER/ir_datasets
#   hf download ibm-granite/granite-embedding-english-r2
#   cd /user/work/$USER/IBM_Granite_Project
#   python -c "from eval.benchmarks.loader import load_benchmark; load_benchmark('dbpedia-entity','test')"
#SBATCH --job-name=granite-scale
#SBATCH --account=coms039904
#SBATCH --partition=gpu               # large corpus: use the long queue, NOT gpu_short
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                    # the flat index of millions of 768-d vectors is tens of GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
DATASET="${1:?usage: sbatch scripts/run_scale_demo.slurm <dataset> [split]}"
SPLIT="${2:-test}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets
export PYTHONUNBUFFERED=1
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| torch', torch.__version__)"

# Same Granite embeddings, two index structures. (Each run encodes the corpus —
# the index cache is keyed per index-type — so it is embedded twice; acceptable
# for a one-off demo. token-mode chunking avoids silent truncation of long docs.)
for IDX in flat hnsw; do
  echo "=================== index_type=$IDX ==================="
  python -m eval.run_benchmark \
      --dataset "$DATASET" --split "$SPLIT" \
      --retrievers granite_dense \
      --index-type "$IDX" --chunk-unit token \
      --out "results/${DATASET}_${IDX}.csv" \
      --cache-dir /user/work/$USER/index_cache
done

echo "=================== recall vs latency (the result) ==================="
echo "-- flat --";  cut -d, -f1,13,15 "results/${DATASET}_flat.csv" | column -s, -t  # retriever, ndcg@10, ms_per_query
echo "-- hnsw --";  cut -d, -f1,13,15 "results/${DATASET}_hnsw.csv" | column -s, -t
