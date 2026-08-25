# Selector Lean v3

The final research Selector is the two-head NLI risk model frozen by
`configs/selector/lean_v3.toml`. It predicts independent protection and harm risks, calibrates
one conservative deletion policy on development data, and opens decision-dev only after that
policy has been frozen.

The implementation is `src/evidence_rag/cli/run_selector_lean.py`. Install the project, then
inspect the three public stages:

```bash
evidence-rag-selector-train --help
evidence-rag-selector-train fit --help
evidence-rag-selector-train calibrate --help
evidence-rag-selector-train final-evaluate --help
```

## Frozen training sequence

Run `fit` for both seeds 13 and 42 using the same NLI base snapshot and R004 materialized
train-fit bundle. The two variants were `NLI-base` and the preregistered `NLI-pair` fallback.
The final selected variant was `NLI-base`.

```bash
evidence-rag-selector-train fit \
  --config configs/selector/lean_v3.toml \
  --r004-root "$SELECTOR_TRAIN_BUNDLE" \
  --model-snapshot "$NLI_MODEL_SNAPSHOT" \
  --variant NLI-base --seed 13 \
  --output-dir "$SELECTOR_RUNS/fit-seed13"
```

Repeat with `--seed 42`. Next run `calibrate` with the two fit directories, the frozen NIAH
and 2Wiki calibration inputs, the Granite snapshot, and a new output policy path. Finally run
`final-evaluate` once using that policy and its pinned code/input/projection hashes. Argparse
marks every required path explicitly; no personal HPC location is assumed.

## Result and decision boundary

- `results/selector/misleading_evidence_summary.json`: evidence gate `PASS`; harmful evidence
  reduction was 12.95%/13.10% for seeds 13/42 with zero observed required-recall or chain loss.
- `results/selector/blind_answer_gate.json`: answer gate `FAIL`; macro answer delta was
  -0.1353 percentage points and the 95% interval crossed zero.
- The frozen experimental decision was `KEEP_TOPK10`. The trained Selector remains available
  as a module and in the demonstration runtime, but the evidence does not support claiming a
  final-answer improvement or replacing TopK as the confirmatory production decision.

Both result files point to the same immutable archive artifact and SHA-256, so the positive
evidence result cannot be separated from the negative answer result.
