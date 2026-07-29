#!/bin/bash
# Retriever comparison: runs `prepare -> retriever` for each config passed as an
# argument, sequentially (one Python process at a time, so Granite is loaded once per
# arm — never two at once — avoiding the double-load OOM on the 40GB MIG slice), then
# prints a MRR / Recall@{5,10,20} comparison table into the .out.
#
# One-time on the login node before submitting (compute nodes have no internet):
#   module load languages/python/3.12.3
#   python -m venv /user/work/$USER/venv && source /user/work/$USER/venv/bin/activate
#   pip install -e ".[granite,benchmark,dev]"
#   export HF_HOME=/user/work/$USER/hf_cache IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONPATH=src
#   hf download ibm-granite/granite-4.1-3b
#   hf download ibm-granite/granite-embedding-english-r2
#   evidence-rag-materialize-benchmark scifact --split test --output data/benchmarks/scifact/test
#   python -m evidence_rag.materializer.base_cli --split dev --output runs/niah-base \
#     --corpus-size 100000 --query-limit 2000 --seed 42
#   mkdir -p logs runs
#
# Submit from the project root, e.g. the SciFact pilot:
#   sbatch scripts/run_retriever_eval.slurm \
#     configs/experiments/retr_scifact_bm25.toml \
#     configs/experiments/retr_scifact_strong-bm25.toml \
#     configs/experiments/retr_scifact_hybrid-rrf.toml \
#     configs/experiments/retr_scifact_hybrid-convex.toml \
#     configs/experiments/retr_scifact_granite-dense.toml \
#     configs/experiments/retr_scifact_query2doc.toml \
#     configs/experiments/retr_scifact_hyde.toml \
#     configs/experiments/retr_scifact_decompose.toml
#SBATCH --job-name=retriever-eval
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dh25275@bristol.ac.uk

set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo "usage: sbatch scripts/run_retriever_eval.slurm CONFIG.toml [CONFIG.toml ...]" >&2
  exit 2
fi

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MODEL_CACHE_DIR=/user/work/$USER/hf_cache/hub   # hub-level dir (transformers cache_dir=)
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export GRANITE_MODEL_ID=ibm-granite/granite-4.1-3b
export GRANITE_EMBEDDING_MODEL_ID=ibm-granite/granite-embedding-english-r2
export LLM_DEVICE=cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

for CONFIG in "$@"; do
  echo "=== $CONFIG : prepare -> retriever ==="
  python -m evidence_rag.cli.experiment --config "$CONFIG" prepare
  python -m evidence_rag.cli.experiment --config "$CONFIG" retriever
done

echo "=== summary: MRR / Recall@{5,10,20} ==="
python scripts/summarize_retriever_eval.py "$@"
