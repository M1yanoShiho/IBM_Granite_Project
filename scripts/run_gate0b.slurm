#!/bin/bash
# Gate 0B sweep: zero-training relation checkpoints over 0B-1 (VitaminC official test) and
# 0B-2 (mutation-log task probe). Protocol: docs/selector/M0_PROTOCOL_FREEZE.md §3.1-§3.3.
#
# All candidate models run in ONE sweep on purpose. A single arm cannot separate "this
# checkpoint is insufficient" from "no zero-shot model is sufficient", and that distinction is
# what decides whether the training path (M0 §3.8) starts.
#
# One-time on the LOGIN node before submitting. Compute nodes run HF-offline, so the
# checkpoints must already be cached or the job dies on the first from_pretrained:
#   export HF_HOME=/user/work/$USER/hf_cache
#   hf download tals/albert-xlarge-vitaminc-mnli
#   hf download MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
#
# Also on the login node — build the two pair files (both are pure CPU):
#   export PYTHONPATH=src
#   python -m evidence_rag.cli.export_vitaminc \
#     --out-test data/gate0b/vitaminc_test.jsonl \
#     --removed-log data/gate0b/vitaminc_decontamination.json
#   python -m evidence_rag.cli.export_task_probe \
#     --manifest runs/niah-train-injected/manifest.json \
#     --provenance runs/niah-train-injected/provenance.jsonl \
#     --output data/gate0b/task_pairs.jsonl
#
# NOTE the probe manifest points at the TRAIN split. Building it from dev or the sealed test
# would put the acceptance set and the evaluation set on the same queries.
#
# The directory is niah-train-INJECTED, not niah-train. The probe needs the counterfactual
# documents and the mutation log, and neither exists before injection — the pre-injection
# directory cannot produce a single pair. This header said `runs/niah-train/` until 2026-08-03;
# that command never worked, which is how R011b came to be run off the record. See R012 AFTER.
#
# R012b (hypothesis-form ablation, design §2.4) needs two more pair files. Rungs 1 and 2 are
# pure CPU; rung 3 must be pre-generated HERE because compute nodes are HF-offline and a
# seq2seq converter is not deterministic — the cache makes the arm reproducible and auditable.
#   python -m evidence_rag.cli.export_task_probe \
#     --manifest runs/niah-train-injected/manifest.json \
#     --provenance runs/niah-train-injected/provenance.jsonl \
#     --output data/gate0b/task_pairs_qa.jsonl \
#     --hypothesis-form question_answer
#   hf download MarkS/bart-base-qa2d                   # ~558MB, AFL-3.0, verified 2026-08-03
#   python -m evidence_rag.cli.export_qa2d \
#     --manifest runs/niah-train-injected/manifest.json \
#     --provenance runs/niah-train-injected/provenance.jsonl \
#     --output data/gate0b/qa2d_cache.jsonl \
#     --model MarkS/bart-base-qa2d
#   python -m evidence_rag.cli.export_task_probe \
#     --manifest runs/niah-train-injected/manifest.json \
#     --provenance runs/niah-train-injected/provenance.jsonl \
#     --output data/gate0b/task_pairs_qa2d.jsonl \
#     --hypothesis-form qa2d --qa2d-cache data/gate0b/qa2d_cache.jsonl
#
# The checkpoint was chosen on 2026-08-03 over domenicrosati/question_converter-3b, which has no
# declared licence, no published evaluation, and does not load under this project's pinned
# torch < 2.6. Evidence and the rejection reasons are in the run log under R012b.
#
# The input format is no longer a human obligation: export_qa2d::INPUT_TEMPLATE registers the
# verified format per checkpoint and refuses an unregistered id BEFORE it touches the Hub, the
# same discipline gate0b::LABEL_ORDER applies to label orders. A wrong format would not crash —
# it yields fluent sentences that are not the QA2D transform — so it is enforced, not documented.
#
# READ THE REPORT LINE BEFORE READING THE RUNG-3 DELTA. export_qa2d prints
# `non_initial_uppercase_rate`: questions and answers both arrive lowercased, so any capital past
# position 0 came from the converter. If that rate is high, rung 3 varies casing as well as
# sentence form and its increment cannot be attributed to form alone (R012b 判读纪律 5).
#
# Submit from the project root. Positional args, all optional, in order:
#   TASK_PAIRS  EXTERNAL_PAIRS  OUTPUT  MODELS  DUMP
# EXTERNAL_PAIRS takes "none" to run 0B-2 only; MODELS is one quoted, space-separated list of
# checkpoint ids; DUMP is empty or "none" for off.
#
# Both tiers, both checkpoints:
#   mkdir -p logs results/gate0b && sbatch scripts/run_gate0b.slurm
#
# 0B-2 only (fast; this tier carries the go/no-go thresholds):
#   sbatch scripts/run_gate0b.slurm data/gate0b/task_pairs.jsonl none \
#     results/gate0b/sweep-task-only.json
#
# 0B-2 only, one checkpoint, with the per-pair dump the kind breakdown needs:
#   sbatch scripts/run_gate0b.slurm data/gate0b/task_pairs.jsonl none \
#     results/gate0b/sweep-task-only.json "tals/albert-xlarge-vitaminc-mnli" \
#     results/gate0b/dump-task-only.jsonl
#SBATCH --job-name=gate0b
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
# Pass "none" as EXTERNAL_PAIRS to run 0B-2 only. That tier is 5888 pairs versus 55197 and
# carries the go/no-go thresholds, so it is worth having on its own turnaround.
TASK_PAIRS="${1:-data/gate0b/task_pairs.jsonl}"
EXTERNAL_PAIRS="${2:-data/gate0b/vitaminc_test.jsonl}"
OUTPUT="${3:-results/gate0b/sweep.json}"
MODELS="${4:-tals/albert-xlarge-vitaminc-mnli MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli}"
# Per-pair predictions for the kind breakdown. Off unless a path is given; the CLI creates the
# parent directory and the aggregate in $OUTPUT is byte-identical either way.
DUMP="${5:-}"
if [ "$EXTERNAL_PAIRS" = "none" ]; then EXTERNAL_ARG=""; else EXTERNAL_ARG="--external-pairs $EXTERNAL_PAIRS"; fi
if [ -z "$DUMP" ] || [ "$DUMP" = "none" ]; then DUMP_ARG=""; else DUMP_ARG="--dump $DUMP"; fi

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

# gpu:1 is enough: the largest arm is DeBERTa-v3-large at ~435M params, inference only. Do NOT
# request a100 — that is for the 8B generative model, not for encoders.
echo "=== Gate 0B sweep (argmax head, no threshold) ==="
# $EXTERNAL_ARG, $DUMP_ARG and $MODELS are deliberately unquoted: the two *_ARG variables must
# disappear entirely when off, and $MODELS must word-split into one argument per checkpoint id.
# No path here contains spaces. --models is last because it takes nargs="+".
python -m evidence_rag.cli.gate0b \
  --task-pairs "$TASK_PAIRS" \
  $EXTERNAL_ARG \
  --output "$OUTPUT" \
  $DUMP_ARG \
  --models $MODELS

echo "=== Gate 0B report ==="
cat "$OUTPUT"
