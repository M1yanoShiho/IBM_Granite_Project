#!/bin/bash
# E2: gate-on/off harmful-in-context (spec docs/superpowers/specs/2026-07-21-harmful-in-context-metric-design.md).
# Runs each arm through prepare -> retriever -> selector ONLY (no generator), so Granite
# is loaded once per arm (sequential processes, never two at once) — avoids the double-load
# OOM on the 40GB MIG slice. Then runs the offline harm report so the .out carries the numbers.
#
# One-time on the login node before submitting:
#   export PYTHONPATH=src HF_HOME=/user/work/$USER/hf_cache IR_DATASETS_HOME=/user/work/$USER/ir_datasets
#   hf download ibm-granite/granite-4.1-3b
#   python -m evidence_rag.materializer.base_cli --split dev --output runs/niah-base --corpus-size 100000 --query-limit 2000 --seed 42
#   python -m evidence_rag.materializer.cli --base-manifest runs/niah-base/manifest.json --output runs/niah-injected --seed 42
#
# Then, before every submission (seconds, login node, no GPU) -- this is the same check the job
# runs below, and running it here turns a rejected arm into a message instead of a queue slot:
#   PYTHONPATH=src python scripts/preflight_arms.py configs/experiments/A.toml configs/experiments/B.toml
#
# Submit from the project root:
#   mkdir -p logs && sbatch scripts/run_selector_gate.slurm \
#     configs/experiments/niah_e2_gate_on.toml configs/experiments/niah_e2_gate_off.toml runs/niah-injected/provenance.jsonl
#SBATCH --job-name=selector-gate
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
CONFIG_ON="${1:?usage: sbatch run_selector_gate.slurm CONFIG_ON CONFIG_OFF PROVENANCE}"
CONFIG_OFF="${2:?usage: sbatch run_selector_gate.slurm CONFIG_ON CONFIG_OFF PROVENANCE}"
PROVENANCE="${3:?usage: sbatch run_selector_gate.slurm CONFIG_ON CONFIG_OFF PROVENANCE [PARENT_INDEX]}"
# Optional 4th arg: the SAME_SOURCE sidecar, required by any config using
# support_unit = "parent". Building the selector fails loudly without it rather than
# silently falling back to the old per-chunk vote counting.
PARENT_INDEX="${4:-}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export GRANITE_MODEL_ID=ibm-granite/granite-4.1-3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src
if [ -n "$PARENT_INDEX" ]; then export SOURCE_PARENT_INDEX="$PARENT_INDEX"; fi

# The tree that ran, recorded rather than assumed -- a queued job executes the checkout as it
# stands when it STARTS, and "do not pull while a job is queued" has already failed as a rule.
echo "[tree] $(git rev-parse HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo ' +uncommitted')"

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

# EVERY arm's stored manifest is checked before ANY arm computes. `prepare` makes this check
# itself, but only when that arm's turn comes -- so in a sequential job the last arm's verdict
# arrives after every earlier arm has finished. Job 18235971 died in arm 2's prepare over a
# mismatch that was decidable at second zero, five hours and nineteen minutes in. The check
# needs no GPU and no corpus; there is no reason for it to run late. `set -e` above makes a
# non-zero exit here end the job before the first prepare.
echo "=== preflight: stored manifests vs index on disk, all arms ==="
python scripts/preflight_arms.py "$CONFIG_ON" "$CONFIG_OFF"

for CONFIG in "$CONFIG_ON" "$CONFIG_OFF"; do
  echo "=== $CONFIG : prepare -> retriever -> selector (no generator) ==="
  python -m evidence_rag.cli.experiment --config "$CONFIG" prepare
  python -m evidence_rag.cli.experiment --config "$CONFIG" retriever
  python -m evidence_rag.cli.experiment --config "$CONFIG" selector
done

resolve_output() {
  python - "$1" <<'PY'
import pathlib
import sys
import tomllib

config = pathlib.Path(sys.argv[1]).resolve()
raw = tomllib.loads(config.read_text(encoding="utf-8"))
print((config.parent / raw["output"]["directory"]).resolve())
PY
}
ON_DIR="$(resolve_output "$CONFIG_ON")"
OFF_DIR="$(resolve_output "$CONFIG_OFF")"

echo "=== E2 harmful-in-context: gate-on vs gate-off ==="
python -m evidence_rag.evaluation.harm_cli \
  --provenance "$PROVENANCE" \
  --selected-on "$ON_DIR/selected_evidence_sets.jsonl" \
  --selected-off "$OFF_DIR/selected_evidence_sets.jsonl" \
  --candidates "$ON_DIR/candidate_sets.jsonl" \
  --dataset-signature niah-e2

echo "=== guardrail: Required Recall (conditional_document_recall) each arm ==="
for DIR in "$ON_DIR" "$OFF_DIR"; do
  python - "$DIR/selector_report.json" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
recall = report.get("aggregate", {}).get("selector.core.conditional_document_recall")
print(sys.argv[1], "conditional_document_recall =", recall)
PY
done
