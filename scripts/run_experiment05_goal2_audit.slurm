#!/bin/bash
#SBATCH --job-name=exp05-g2-audit
#SBATCH --account=coms039904
#SBATCH --partition=compute
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/user/work/fl25387/experiment05_goal2/logs/%x-%j.out

set -euo pipefail

repo=/user/work/fl25387/IBM_Granite_Project_latest
goal1=/user/work/fl25387/experiment05_goal1
goal2=/user/work/fl25387/experiment05_goal2
cache=/user/work/fl25387/experiment04_goal2_cache
python=/user/work/fl25387/experiment04_goal3_venv/bin/python

export PYTHONPATH="$repo/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

tokenizer=$(find "$cache/models--ibm-granite--granite-4.1-3b/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)
mkdir -p "$goal2/audits"
"$python" "$repo/scripts/experiment05_goal2_audit.py" \
  --goal1-root "$goal1" \
  --goal2-root "$goal2" \
  --tokenizer-snapshot "$tokenizer" \
  --formal-output-root /user/work/fl25387/experiment05_runs/goal3/outputs \
  --output "$goal2/audits/goal2_audit.json"
