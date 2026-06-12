#!/bin/bash
# Full RAG benchmark run on a larger GPU. Submit from the project root:
#   mkdir -p logs && sbatch scripts/run_rag.slurm
#
# NOTE: eval/run_rag.py is still a skeleton. Before this produces a table you
# need: a gold-answer dataset loader (NQ/DocFinQA), eval/rag_metrics.py
# implemented, and a working DenseRetriever. See docs/dev-log.md.
#SBATCH --job-name=granite-rag
#SBATCH --account=coms039904
#SBATCH --partition=gpu                 # 14-day cap; use for the real benchmark
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_3090:1           # 24GB, fits 8B bf16. For the 40GB A100
                                        # MIG slice use: gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# Compute nodes have no python by default — load the same module the venv was
# built with before activating it.
module load languages/python/3.12.3

export HF_HOME=/user/work/$USER/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GRANITE_MODEL_ID=ibm-granite/<8B-INSTRUCT>   # TODO: confirm exact repo on HF

source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project

python -m eval.run_rag --dataset nq
