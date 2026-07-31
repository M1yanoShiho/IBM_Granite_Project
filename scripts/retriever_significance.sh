#!/bin/bash
# Paired significance tests between retriever variants (CPU, offline, seconds).
#
# Wraps evidence_rag.evaluation.paired_metric_cli: for a fixed set of variant pairs it
# reads each arm's runs/retr-<DS>-<variant>/retriever_report.json, pairs per-case metrics
# by query_id, and prints the mean delta + randomization p-value + bootstrap CI. Run on
# the login node AFTER the retriever job for the dataset has finished.
#
# Usage:
#   scripts/retriever_significance.sh <dataset-tag>        # e.g. scifact | nq | 2wiki
# The pairs below are "ON OFF" (delta = ON - OFF); ON is the richer/candidate variant.
set -euo pipefail

DS="${1:?usage: scripts/retriever_significance.sh <dataset-tag>   e.g. scifact | nq | 2wiki}"
export PYTHONPATH="${PYTHONPATH:-src}"

# Comparisons worth reporting. Query-transform ablations answer "does rewriting pay off?".
PAIRS=(
  "query2doc strong-bm25"       # Query2Doc rewrite vs its own sparse base
  "hyde granite-dense"          # HyDE rewrite vs its own dense base
  "decompose strong-bm25"       # Decomposition vs its own sparse base
  "hybrid-rrf strong-bm25"      # Rank fusion vs best sparse
  "hybrid-convex granite-dense" # Score fusion vs dense
  "strong-bm25 bm25"            # Tuned sparse vs baseline sparse
)
METRICS=(retriever.core.document_mrr retriever.core.document_recall_at_10)

report() { echo "runs/retr-${DS}-$1/retriever_report.json"; }

printf '%-26s %-34s %8s %8s %8s %10s %7s\n' "comparison (ON vs OFF)" "metric" "mean_on" "mean_off" "delta" "p_value" "n"
printf '%s\n' "-------------------------------------------------------------------------------------------------------------"
for pair in "${PAIRS[@]}"; do
  read -r ON OFF <<<"$pair"
  ON_REPORT="$(report "$ON")"; OFF_REPORT="$(report "$OFF")"
  if [[ ! -f "$ON_REPORT" || ! -f "$OFF_REPORT" ]]; then
    printf '%-26s %s\n' "${ON} vs ${OFF}" "(missing report; skipped)"
    continue
  fi
  for METRIC in "${METRICS[@]}"; do
    python -m evidence_rag.evaluation.paired_metric_cli \
      --on-report "$ON_REPORT" --off-report "$OFF_REPORT" --metric "$METRIC" \
    | python - "$ON vs $OFF" "$METRIC" <<'PY'
import json, sys
label, metric = sys.argv[1], sys.argv[2]
d = json.loads(sys.stdin.read())
print(f"{label:<26} {metric:<34} {d['mean_on']:8.4f} {d['mean_off']:8.4f} "
      f"{d['delta']:+8.4f} {d['p_value']:10.4f} {d['n_paired']:7d}")
PY
  done
done
