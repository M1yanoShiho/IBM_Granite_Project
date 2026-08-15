#!/usr/bin/env bash
set -euo pipefail

REPO=/home/fl25387/projects/IBM_Granite_Project_latest
RUNTIME=/scratch/fl25387/IBM_Granite_Project_latest
PYTHON=$RUNTIME/envs/selector_mis_py311/bin/python
GRANITE=$RUNTIME/model-cache/models--ibm-granite--granite-4.1-3b/snapshots/c0650403e44e78ec0262dab1c90914c65b196c4e
TRUE_NLI=$RUNTIME/model-cache/models--google--t5_xxl_true_nli_mixture/snapshots/aa6cfe1dd4257853bfdd772992045f41bfc14988
F001=$RUNTIME/runs/full-flow/F001/generations.jsonl
TRACE=$RUNTIME/runs/selector-lean-v3/L003/final/decision_trace_seed13.jsonl
GOLD=$REPO/runs/selector-beam-v1/pools/niah-dev/gold_cases.jsonl
OUT=$RUNTIME/runs/full-flow/F004

export PYTHONPATH=$REPO/src:$REPO/scripts
export HF_HOME=$RUNTIME/model-cache
export MODEL_CACHE_DIR=$RUNTIME/model-cache

"$PYTHON" "$REPO/scripts/full_flow_f004.py" run \
  --f001-generations "$F001" \
  --selector-trace "$TRACE" \
  --granite-snapshot "$GRANITE" \
  --true-model-id "$TRUE_NLI" \
  --output-dir "$OUT"

"$PYTHON" "$REPO/scripts/full_flow_f004.py" score \
  --generations "$OUT/generations.jsonl" \
  --gold "$GOLD" \
  --output-json "$OUT/report.json" \
  --output-report "$OUT/REPORT.md"
