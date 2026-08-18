#!/bin/bash
# R13 run-to-run variance probe: the same latency curve, three times, in ONE allocation.
#
# R13's conclusion (superlinear scaling, fitted exponent 1.15-1.3, R9's asymptotic claim
# withdrawn) rests on a single 6m46s job. The reported quantities are mean / p50 / p95
# milliseconds -- p95 in particular is the highest-variance statistic in the run, and node
# contention feeds straight into it.
#
# The three repeats MUST share one allocation. Three separate sbatch submissions would land
# on three different nodes and measure the nodes, not the run-to-run spread -- that is R6b's
# lesson, and R13's own design already invokes it for the latency/memory pairing.
#
# Memory is measured once, not three times: R13 established that B/chunk is insensitive to
# node contention (login-node 5552/5101 matched the job's readings).
#
# Submit from the project root:
#   sbatch scripts/run_retriever_scaling_repeat.slurm \
#     runs/niah-base-200k/manifest.json \
#     results/r13-repeat/retriever-scaling-nq-inverted \
#     "12500 25000 50000 100000 200000"
#SBATCH --job-name=retriever-scaling-repeat
#SBATCH --account=coms039904
#SBATCH --partition=compute
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dh25275@bristol.ac.uk

set -euo pipefail

module load languages/python/3.11.15
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src PYTHONUNBUFFERED=1

MANIFEST="${1:-runs/niah-base-200k/manifest.json}"
OUTPUT_STEM="${2:-results/r13-repeat/retriever-scaling-nq-inverted}"
SIZES="${3:-12500 25000 50000 100000 200000}"

mkdir -p "$(dirname "$OUTPUT_STEM")"

echo "== node: $(hostname)  job: ${SLURM_JOB_ID:-none}"
echo "== sizes: $SIZES"

for rep in 1 2 3; do
  echo "== latency repeat $rep/3"
  python scripts/retriever_scaling.py \
    --manifest "$MANIFEST" \
    --sizes $SIZES \
    --queries 50 \
    --output "${OUTPUT_STEM}-rep${rep}.json"
done

echo "== memory (once; insensitive to contention per R13)"
python scripts/retriever_memory.py \
  --manifest "$MANIFEST" \
  --sizes $SIZES \
  --output "${OUTPUT_STEM}-memory.json"
