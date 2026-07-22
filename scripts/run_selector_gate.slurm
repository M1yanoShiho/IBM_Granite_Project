#!/bin/bash
# E2: gate-on/off paired selector comparison (spec §12 main comparison,
# docs/superpowers/specs/2026-07-20-gated-corroboration-selector-design.md).
# Two experiment configs must differ ONLY in [selector]:
#   gate-on : name = "gated-corroboration"  (alpha/margin/support_cap/top_n per spec §7)
#   gate-off: name = "corroboration"        (same alpha/top_n — isolates the gate causally)
#
# STATUS: BLOCKED until the spec §14 dataset decision produces the two TOMLs.
# Do not submit before docs/hpc-run-log.md E2 has its BEFORE section completed.
#
# One-time on the login node (compute nodes are offline):
#   hf download ibm-granite/granite-4.1-3b
# Submit from the project root AFTER `git pull`:
#   mkdir -p logs runs && sbatch scripts/run_selector_gate.slurm \
#     configs/experiments/<gate_on>.toml configs/experiments/<gate_off>.toml
#
# Model selection: GraniteLLMClient reads GRANITE_MODEL_ID
# (src/evidence_rag/generator/granite.py); default ibm-granite/granite-4.1-3b.
# 3B is pinned here deliberately — the 2026-07-06 nested-CV re-eval showed the 8B
# extractor does NOT beat 3B for corroboration extraction (+.013 ns vs +.037).
#SBATCH --job-name=selector-gate
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
CONFIG_ON="${1:?usage: sbatch run_selector_gate.slurm CONFIG_ON CONFIG_OFF}"
CONFIG_OFF="${2:?usage: sbatch run_selector_gate.slurm CONFIG_ON CONFIG_OFF}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export GRANITE_MODEL_ID=ibm-granite/granite-4.1-3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

for CONFIG in "$CONFIG_ON" "$CONFIG_OFF"; do
  echo "=== running $CONFIG ==="
  python -m evidence_rag.cli.experiment --config "$CONFIG" all
done

echo "=== selector reports (first-read numbers) ==="
for CONFIG in "$CONFIG_ON" "$CONFIG_OFF"; do
  python - "$CONFIG" <<'PY'
import pathlib
import sys
import tomllib

config_path = pathlib.Path(sys.argv[1]).resolve()
raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
output = (config_path.parent / raw["output"]["directory"]).resolve()
report = output / "selector_report.json"
print(config_path.name, "->", report)
print(report.read_text(encoding="utf-8") if report.is_file() else "MISSING")
PY
done
