#!/bin/bash
#SBATCH --job-name=exp05-g1-normalize
#SBATCH --account=coms039904
#SBATCH --partition=compute
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/user/work/fl25387/experiment05_goal1/logs/%x-%j.out

set -euo pipefail

repo=/user/work/fl25387/IBM_Granite_Project_latest
root=/user/work/fl25387/experiment05_goal1
python="$root/venv/bin/python"
export PYTHONPATH="$repo/src"

case "${DATASET:?DATASET must be kilt or dpr}" in
  kilt)
    "$python" "$repo/scripts/experiment05_goal1_index.py" normalize \
      --dataset kilt-wikipedia-20190801 \
      --source-kind kilt \
      --source "$root/source_cache/kilt_knowledgesource.json" \
      --output-dir "$root/corpus/kilt" \
      --max-words 100 \
      --shard-size 250000
    ;;
  dpr)
    "$python" "$repo/scripts/experiment05_goal1_index.py" normalize \
      --dataset dpr-wikipedia-psgs-w100 \
      --source-kind dpr \
      --source "$root/source_cache/psgs_w100.tsv.gz" \
      --output-dir "$root/corpus/dpr" \
      --shard-size 250000
    ;;
  *)
    echo "unsupported DATASET=$DATASET" >&2
    exit 2
    ;;
esac
