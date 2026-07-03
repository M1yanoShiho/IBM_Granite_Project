#!/bin/bash
# Build the first hard NIAH task (Sources A + C) on a GPU node: Granite dense +
# SPLADE retrieval signals -> build_task -> conservative hardness gate ->
# results/niah_<dataset>_task.json. WATCH the printed "Hardness gate" line: it must
# say saturated=False / passes_gate=True, else raise MINE_K and rebuild.
#
#   mkdir -p logs results && sbatch scripts/run_build_niah_task.slurm nq dev
#
# FIRST, on the LOGIN node (compute nodes are offline), one-time setup:
#   source scripts/env.sh
#   bash scripts/convert_splade_safetensors.sh           # SPLADE .bin -> safetensors
#   hf download ibm-granite/granite-embedding-english-r2  # dense retriever
#   hf download ibm-granite/granite-4.1-3b                # LLM generator + answerability judge
#   python -c "from eval.benchmarks.loader import load_benchmark; load_benchmark('nq','dev',max_queries=1,max_docs=1000)"
#SBATCH --job-name=niah-build-task
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
DATASET="${1:?usage: sbatch scripts/run_build_niah_task.slurm <dataset> [split]}"
SPLIT="${2:-dev}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
# SPLADE weights from the locally-converted safetensors copy (torch 2.5.1 < 2.6
# cannot torch.load .bin); lower SPLADE_BATCH_SIZE if the GPU OOMs.
export SPLADE_MODEL_ID="${SPLADE_MODEL_ID:-/user/work/$USER/splade_st}"
export SPLADE_BATCH_SIZE="${SPLADE_BATCH_SIZE:-32}"
# The LLM used BOTH to generate counterfactuals and as the answerability judge.
export GRANITE_MODEL_ID="${GRANITE_MODEL_ID:-ibm-granite/granite-4.1-3b}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

# START SMALL: validate the pipeline + read the gate verdict fast, then scale up by
# raising MAX_QUERIES / MAX_DOCS / MINE_K (env overrides) on a re-run. The LLM
# answerability judge runs ~(1 + MINE_K*2) times per query, so MINE_K drives cost.
MAX_QUERIES="${MAX_QUERIES:-100}"
MAX_DOCS="${MAX_DOCS:-200000}"
MINE_K="${MINE_K:-5}"

python -m eval.build_niah_task \
    --dataset "$DATASET" --split "$SPLIT" \
    --max-queries "$MAX_QUERIES" --max-docs "$MAX_DOCS" \
    --sparse-arm splade --mine-k "$MINE_K" \
    --index-type hnsw \
    --out "results/niah_${DATASET}_task.json"

echo "=================== DONE ==================="
echo "Inspect results/niah_${DATASET}_task.json and the 'Hardness gate' line above."
echo "If saturated=True -> raise the distractor count: MINE_K=20 sbatch scripts/run_build_niah_task.slurm $DATASET $SPLIT"
