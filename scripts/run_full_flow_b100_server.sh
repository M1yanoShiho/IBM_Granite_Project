#!/usr/bin/env bash
set -euo pipefail

REPO=/home/fl25387/projects/IBM_Granite_Project_latest
RUNTIME=/scratch/fl25387/IBM_Granite_Project_latest
PYTHON=$RUNTIME/envs/selector_mis_py311/bin/python
GRANITE=$RUNTIME/model-cache/models--ibm-granite--granite-4.1-3b/snapshots/c0650403e44e78ec0262dab1c90914c65b196c4e
TRUE_NLI=$RUNTIME/model-cache/models--google--t5_xxl_true_nli_mixture/snapshots/aa6cfe1dd4257853bfdd772992045f41bfc14988
OUT=${1:-$RUNTIME/runs/full-flow/B100-v1}
CHANGED_LIMIT=${2:-}

export HF_HOME=$RUNTIME/model-cache
export MODEL_CACHE_DIR=$RUNTIME/model-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=$REPO/src:$REPO/scripts

PREPARE_ARGS=(
  prepare
  --queries "$REPO/runs/selector-beam-v1/pools/niah-dev/queries.jsonl"
  --candidate-pool "$REPO/runs/selector-beam-v1/pools/niah-dev/candidate_sets.jsonl"
  --roles "$RUNTIME/runs/selector-adaptive-risk-v1/R002/niah-dev/role_assignments.jsonl"
  --decision-trace "$RUNTIME/runs/selector-lean-v3/L003/final/decision_trace_seed13.jsonl"
  --gold "$REPO/runs/selector-beam-v1/pools/niah-dev/gold_cases.jsonl"
  --output-dir "$OUT"
)
if [[ -n "$CHANGED_LIMIT" ]]; then
  PREPARE_ARGS+=(--changed-limit "$CHANGED_LIMIT")
fi

"$PYTHON" "$REPO/scripts/full_flow_b100.py" "${PREPARE_ARGS[@]}"
"$PYTHON" "$REPO/scripts/full_flow_b100.py" run \
  --contexts "$OUT/runtime_contexts.jsonl" \
  --granite-snapshot "$GRANITE" \
  --true-snapshot "$TRUE_NLI" \
  --output-dir "$OUT"
"$PYTHON" "$REPO/scripts/full_flow_b100.py" score \
  --contexts "$OUT/runtime_contexts.jsonl" \
  --audit "$OUT/context_audit.jsonl" \
  --generations "$OUT/generations.jsonl" \
  --gold "$REPO/runs/selector-beam-v1/pools/niah-dev/gold_cases.jsonl" \
  --output-cases "$OUT/B100_cases.jsonl" \
  --output-json "$OUT/report.json" \
  --output-report "$OUT/B100_DIAGNOSTIC_REPORT.md"

test -s "$OUT/prepare_manifest.json"
test -s "$OUT/run_manifest.json"
test -s "$OUT/generations.jsonl"
test -s "$OUT/B100_cases.jsonl"
test -s "$OUT/report.json"
test -s "$OUT/B100_DIAGNOSTIC_REPORT.md"
