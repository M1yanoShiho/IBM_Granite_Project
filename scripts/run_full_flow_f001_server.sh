#!/usr/bin/env bash
set -euo pipefail

REPO=/home/fl25387/projects/IBM_Granite_Project_latest
RUNTIME=/scratch/fl25387/IBM_Granite_Project_latest
PYTHON=$RUNTIME/envs/selector_mis_py311/bin/python
GRANITE=$RUNTIME/model-cache/models--ibm-granite--granite-4.1-3b/snapshots/c0650403e44e78ec0262dab1c90914c65b196c4e
TRUE_NLI=$RUNTIME/model-cache/models--google--t5_xxl_true_nli_mixture/snapshots/aa6cfe1dd4257853bfdd772992045f41bfc14988
OUT=$RUNTIME/runs/full-flow/F001

export HF_HOME=$RUNTIME/model-cache
export MODEL_CACHE_DIR=$RUNTIME/model-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=$REPO/src

mkdir -p "$RUNTIME/runs/full-flow/F000"

"$PYTHON" "$REPO/scripts/full_flow_joint.py" prepare \
  --queries "$REPO/runs/selector-beam-v1/pools/niah-dev/queries.jsonl" \
  --candidate-pool "$REPO/runs/selector-beam-v1/pools/niah-dev/candidate_sets.jsonl" \
  --roles "$RUNTIME/runs/selector-adaptive-risk-v1/R002/niah-dev/role_assignments.jsonl" \
  --decision-trace "$RUNTIME/runs/selector-lean-v3/L003/final/decision_trace_seed13.jsonl" \
  --gold "$REPO/runs/selector-beam-v1/pools/niah-dev/gold_cases.jsonl" \
  --output "$RUNTIME/runs/full-flow/F000/input_manifest.json"

"$PYTHON" "$REPO/scripts/full_flow_joint.py" run \
  --queries "$REPO/runs/selector-beam-v1/pools/niah-dev/queries.jsonl" \
  --candidate-pool "$REPO/runs/selector-beam-v1/pools/niah-dev/candidate_sets.jsonl" \
  --roles "$RUNTIME/runs/selector-adaptive-risk-v1/R002/niah-dev/role_assignments.jsonl" \
  --decision-trace "$RUNTIME/runs/selector-lean-v3/L003/final/decision_trace_seed13.jsonl" \
  --granite-snapshot "$GRANITE" \
  --nli-backend true \
  --nli-model-id "$TRUE_NLI" \
  --output-dir "$OUT"

"$PYTHON" "$REPO/scripts/full_flow_joint.py" score \
  --generations "$OUT/generations.jsonl" \
  --gold "$REPO/runs/selector-beam-v1/pools/niah-dev/gold_cases.jsonl" \
  --output-json "$OUT/report.json" \
  --output-report "$OUT/REPORT.md"
