#!/bin/bash
# Corpus-size scaling probe for the sparse retriever: how index-build time and
# per-query latency grow as the corpus grows. CPU only -- no GPU, no LLM, no model
# download -- so it queues on the cheap partition and needs none of the granite setup.
#
# Prereqs (login node, once): the venv from run_retriever_eval.slurm's header, plus a
# materialised dataset. SciFact is the smallest useful one:
#   evidence-rag-materialize-benchmark scifact --split test --output data/benchmarks/scifact/test
#
# Submit from the project root:
#   mkdir -p logs results && sbatch scripts/run_retriever_scaling.slurm
#
# To probe a decade further, point --manifest at a larger corpus (e.g. runs/niah-base
# built with --corpus-size 100000) and extend --sizes; nothing else changes.
#SBATCH --job-name=retriever-scaling
#SBATCH --account=coms039904
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dh25275@bristol.ac.uk

set -euo pipefail

module load languages/python/3.11.15
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src PYTHONUNBUFFERED=1

MANIFEST="${1:-data/benchmarks/scifact/test/manifest.json}"
OUTPUT="${2:-results/retriever-scaling-scifact.json}"

python scripts/retriever_scaling.py \
  --manifest "$MANIFEST" \
  --sizes 500 1000 2000 3000 4000 5183 \
  --queries 50 \
  --output "$OUTPUT"
