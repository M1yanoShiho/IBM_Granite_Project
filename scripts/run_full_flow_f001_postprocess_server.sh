#!/usr/bin/env bash
set -euo pipefail

REPO=/home/fl25387/projects/IBM_Granite_Project_latest
RUNTIME=/scratch/fl25387/IBM_Granite_Project_latest
PYTHON=$RUNTIME/envs/selector_mis_py311/bin/python
OUT=$RUNTIME/runs/full-flow/F001
GOLD=$REPO/runs/selector-beam-v1/pools/niah-dev/gold_cases.jsonl
MINICHECK=$RUNTIME/model-cache/models--lytang--MiniCheck-Flan-T5-Large/snapshots/96eafd01cee2d16cf81aaa2fb226b14f422a37b3
MAIN_PID=${1:?pass the F001 runner PID}

export PYTHONPATH=$REPO/src:$REPO/scripts
export HF_HOME=$RUNTIME/model-cache
export MODEL_CACHE_DIR=$RUNTIME/model-cache

while [ ! -f "$OUT/report.json" ]; do
  if ! kill -0 "$MAIN_PID" 2>/dev/null; then
    echo "F001 runner exited before report.json was written" >&2
    exit 1
  fi
  sleep 30
done

"$PYTHON" "$REPO/scripts/full_flow_citation_score.py" \
  --generations "$OUT/generations.jsonl" \
  --model-id "$MINICHECK" \
  --device cuda:0 \
  --output-json "$OUT/citation_report.json" \
  --output-rows "$OUT/citation_per_case.jsonl" \
  --output-report "$OUT/CITATION_REPORT.md"

"$PYTHON" "$REPO/scripts/full_flow_diagnose.py" \
  --generations "$OUT/generations.jsonl" \
  --gold "$GOLD" \
  --output-rows "$OUT/F002_diagnosis_rows.jsonl" \
  --output-json "$OUT/F002_summary.json" \
  --output-report "$OUT/F002_REPORT.md"
