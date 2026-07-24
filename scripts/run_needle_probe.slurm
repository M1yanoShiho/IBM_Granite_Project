#!/bin/bash
# E1-diag: needle-extraction failure-mode probe (systematic-debugging Phase 3).
# For each needle-in-window injected query, extract from the needle chunk ONLY (~1 call/query,
# ~15-20 min on 3B), classify recovered/wrong/none, split by visibility. Answers whether the
# ~50% needle-gold-recovery is the extractor abstaining (NONE -> prompt fix) or mis-extracting
# (WRONG -> a bigger extractor might help) when the gold answer is visible.
#
# NOTE: gres is gpu:1 here (not the runbook default gpu:3g.40gb:1) because the MIG slice was
# unavailable for the E1 cluster-eval run; a full GPU schedules and the 3B extractor fits easily.
#
# Submit from the project root:
#   mkdir -p logs results && sbatch scripts/run_needle_probe.slurm \
#     runs/e2-gate-on/candidate_sets.jsonl runs/niah-injected/manifest.json \
#     runs/niah-injected/provenance.jsonl results/e1-needle-probe.json results/e1-needle-probe.jsonl
#SBATCH --job-name=needle-probe
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
CANDIDATES="${1:?usage: sbatch run_needle_probe.slurm CANDIDATES MANIFEST PROVENANCE OUTPUT DUMP}"
MANIFEST="${2:?usage: sbatch run_needle_probe.slurm CANDIDATES MANIFEST PROVENANCE OUTPUT DUMP}"
PROVENANCE="${3:?usage: sbatch run_needle_probe.slurm CANDIDATES MANIFEST PROVENANCE OUTPUT DUMP}"
OUTPUT="${4:?usage: sbatch run_needle_probe.slurm CANDIDATES MANIFEST PROVENANCE OUTPUT DUMP}"
DUMP="${5:?usage: sbatch run_needle_probe.slurm CANDIDATES MANIFEST PROVENANCE OUTPUT DUMP}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export GRANITE_MODEL_ID=ibm-granite/granite-4.1-3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

echo "=== E1-diag needle-extraction probe ==="
python -m evidence_rag.evaluation.needle_probe_cli \
  --candidates "$CANDIDATES" \
  --manifest "$MANIFEST" \
  --provenance "$PROVENANCE" \
  --output "$OUTPUT" \
  --dump "$DUMP"

echo "=== probe summary ==="
cat "$OUTPUT"
echo "=== visible failures: NONE examples (first 10) ==="
grep '"visible": true' "$DUMP" | grep '"outcome": "none"' | head -10 || true
echo "=== visible failures: WRONG examples (first 10) ==="
grep '"visible": true' "$DUMP" | grep '"outcome": "wrong"' | head -10 || true
