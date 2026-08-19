#!/usr/bin/env bash
set -euo pipefail

REPO=/home/fl25387/projects/IBM_Granite_Project_latest
RUNTIME=/scratch/fl25387/IBM_Granite_Project_latest
PYTHON=$RUNTIME/envs/selector_mis_py311/bin/python
GRANITE=$RUNTIME/model-cache/models--ibm-granite--granite-4.1-3b/snapshots/c0650403e44e78ec0262dab1c90914c65b196c4e
TRUE_NLI=$RUNTIME/model-cache/models--google--t5_xxl_true_nli_mixture/snapshots/aa6cfe1dd4257853bfdd772992045f41bfc14988
MINICHECK=$RUNTIME/model-cache/models--lytang--MiniCheck-Flan-T5-Large/snapshots/96eafd01cee2d16cf81aaa2fb226b14f422a37b3
ROUTE=$REPO/docs/full-flow/experiments/02_generator_selector_alignment_2026-08-15
OUT=$RUNTIME/runs/full-flow/G230-v1

export HF_HOME=$RUNTIME/hf-cache
export MODEL_CACHE_DIR=$RUNTIME/model-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=$REPO/src:$REPO/scripts

COMMON=(
  --queries "$REPO/runs/selector-beam-v1/pools/niah-dev/queries.jsonl"
  --candidate-pool "$REPO/runs/selector-beam-v1/pools/niah-dev/candidate_sets.jsonl"
  --roles "$RUNTIME/runs/selector-adaptive-risk-v1/R002/niah-dev/role_assignments.jsonl"
  --decision-trace "$RUNTIME/runs/selector-lean-v3/L003/final/decision_trace_seed13.jsonl"
  --stress-contexts "$ROUTE/artifacts/B100/runtime_contexts.jsonl"
  --a002-manifest "$ROUTE/artifacts/A002/run_manifest.json"
  --b100-manifest "$ROUTE/artifacts/B100/run_manifest.json"
  --model-snapshot "$GRANITE"
  --true-snapshot "$TRUE_NLI"
)

command=${1:-}
case "$command" in
  run-gn)
    shift
    "$PYTHON" "$REPO/scripts/full_flow_g230.py" run \
      --mode gn \
      "${COMMON[@]}" \
      --gn-run-dir "$RUNTIME/runs/full-flow/F006/mixed-v1" \
      --gn-eval-manifest "$REPO/docs/full-flow/experiments/01_selector_generator_bridge_2026-08-12/artifacts/F005/eval/run_manifest.json" \
      --output-dir "$OUT/gn" \
      "$@"
    ;;
  run-seed)
    seed=${2:?usage: run-seed SEED [extra run arguments]}
    shift 2
    "$PYTHON" "$REPO/scripts/full_flow_g230.py" run \
      --mode draft-pair \
      --seed "$seed" \
      "${COMMON[@]}" \
      --gc-run-dir "$RUNTIME/runs/full-flow/G220-v1/formal/gc-seed$seed" \
      --gm-run-dir "$RUNTIME/runs/full-flow/G220-v1/formal/gm-seed$seed" \
      --output-dir "$OUT/seed$seed" \
      "$@"
    ;;
  score)
    mkdir -p "$OUT/score"
    "$PYTHON" "$REPO/scripts/full_flow_g230.py" score \
      --a002-generations "$ROUTE/artifacts/A002/generations.jsonl" \
      --b100-generations "$ROUTE/artifacts/B100/generations.jsonl" \
      --gn-generations "$OUT/gn/generations.jsonl" \
      --seed13-generations "$OUT/seed13/generations.jsonl" \
      --seed42-generations "$OUT/seed42/generations.jsonl" \
      --seed73-generations "$OUT/seed73/generations.jsonl" \
      --gold "$REPO/runs/selector-beam-v1/pools/niah-dev/gold_cases.jsonl" \
      --output-json "$OUT/score/answer_report.json" \
      --output-cases "$OUT/score/scored_cases.jsonl" \
      --output-report "$OUT/score/ANSWER_REPORT.md"
    ;;
  citation)
    mkdir -p "$OUT/citation"
    "$PYTHON" "$REPO/scripts/full_flow_g230_citation_score.py" \
      --a002-generations "$ROUTE/artifacts/A002/generations.jsonl" \
      --b100-generations "$ROUTE/artifacts/B100/generations.jsonl" \
      --gn-generations "$OUT/gn/generations.jsonl" \
      --seed13-generations "$OUT/seed13/generations.jsonl" \
      --seed42-generations "$OUT/seed42/generations.jsonl" \
      --seed73-generations "$OUT/seed73/generations.jsonl" \
      --answer-report "$OUT/score/answer_report.json" \
      --model-id "$MINICHECK" \
      --device cuda \
      --output-json "$OUT/citation/citation_report.json" \
      --output-rows "$OUT/citation/citation_cases.jsonl" \
      --output-report "$OUT/citation/CITATION_REPORT.md"
    ;;
  *)
    echo "usage: $0 {run-gn|run-seed SEED|score|citation} [extra arguments]" >&2
    exit 2
    ;;
esac
