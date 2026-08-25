#!/bin/bash
# Full-pipeline comparison: runs `prepare -> pipeline` for each config passed as an
# argument, sequentially (one Python process at a time, so Granite is loaded once per
# arm — never two at once — mirroring run_retriever_eval.slurm's reason).
#
# Unlike run_retriever_eval.slurm, which stops after the retriever stage, this runs the
# three modules end to end through EvidenceRAGPipeline.run_with_trace() and writes the
# system.core.* metrics. That is what R7 needs: the same corpus and queries scored at
# both ends of the chain, so an upstream retrieval gain can be followed downstream.
#
# Prereqs are identical to run_retriever_eval.slurm (venv, HF cache, materialised
# dataset). GPU is requested because the hybrid arm encodes with Granite and the
# decompose arm calls the LLM; the bm25 arms ignore it.
#
# Submit from the project root, e.g. the R7 four-arm set:
#   mkdir -p logs runs && sbatch scripts/run_pipeline_eval.slurm \
#     configs/experiments/pipe_2wiki_bm25.toml \
#     configs/experiments/pipe_2wiki_strong-bm25.toml \
#     configs/experiments/pipe_2wiki_hybrid-rrf.toml \
#     configs/experiments/pipe_2wiki_decompose.toml
#SBATCH --job-name=pipeline-eval
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dh25275@bristol.ac.uk

set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo "usage: sbatch scripts/run_pipeline_eval.slurm CONFIG.toml [CONFIG.toml ...]" >&2
  exit 2
fi

# 3.12.3, matching run_retriever_eval.slurm — the venv holding sentence-transformers was
# built under it. This script first copied 3.11.15 from run_retriever_scaling.slurm, which
# is CPU-only and never imports Granite, so the mismatch stayed invisible until the
# hybrid arm reached GraniteEmbedder and job 18325274 died there.
module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MODEL_CACHE_DIR=/user/work/$USER/hf_cache/hub
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
  echo "=== $CONFIG : prepare -> pipeline ==="
  python -m evidence_rag.cli.experiment --config "$CONFIG" prepare
  python -m evidence_rag.cli.experiment --config "$CONFIG" pipeline
done

echo "=== summary: retriever vs system metrics ==="
python scripts/summarize_pipeline_eval.py "$@"
