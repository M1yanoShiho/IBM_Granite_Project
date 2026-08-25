#!/bin/bash
#SBATCH --job-name=exp05-g1-bm25
#SBATCH --account=coms039904
#SBATCH --partition=compute
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/user/work/fl25387/experiment05_goal1/logs/%x-%j.out

set -euo pipefail
module load languages/java-sdk/22.0.2

repo=/user/work/fl25387/IBM_Granite_Project_latest
root=/user/work/fl25387/experiment05_goal1
python="$root/venv/bin/python"
export PYTHONPATH="$repo/src:/user/work/fl25387/experiment04_goal3_venv/lib/python3.11/site-packages"

dataset=${DATASET:?DATASET must be kilt or dpr}
"$python" "$repo/scripts/experiment05_goal1_index.py" bm25 \
  --corpus-dir "$root/corpus/$dataset" \
  --index-dir "$root/indices/$dataset/bm25" \
  --threads 16 \
  --k1 0.9 \
  --b 0.4
