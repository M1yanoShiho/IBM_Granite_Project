#!/bin/bash
# OCR smoke test: prove the new PDF embedded-figure caption + in-figure OCR path
# runs end to end on real models (Docling OCR + Granite Vision), not test fakes.
#
# Path exercised (loaders/dispatch.py): caption_pdf_pictures=true builds the
# Docling converter with generate_picture_images=True -> extract_pictures pulls
# the embedded figure -> phase 2 Granite Vision captions it -> phase 3
# extract_ocr_text_from_image OCRs the SAME picture and appends "Text in image:".
# PASS = an image-source document whose text carries the figure's sentinel digits.
#
# Prereqs (run ONCE on the login node, which has internet -- compute nodes do not):
#   source /user/work/$USER/venv/bin/activate
#   export HF_HOME=/user/work/$USER/hf_cache
#   pip install -e '.[ingestion]' && pip install fpdf2
#   hf download ibm-granite/granite-vision-3.3-2b
#   docling-tools models download -o /user/work/$USER/docling_artifacts layout tableformer easyocr
#   PYTHONPATH=src python scripts/make_ocr_smoke_pdf.py
#
# Submit from the project root:
#   mkdir -p logs runs/ocr-smoke && sbatch scripts/run_ingest_ocr_smoke.slurm
#SBATCH --job-name=ingest-ocr-smoke
#SBATCH --account=coms039904
#SBATCH --partition=gpu_short
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dh25275@bristol.ac.uk

set -euo pipefail

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# cache_dir= expects the hub-level dir (.../models--org--repo), not the HF_HOME
# root -- see the fix in scripts/run_verifier_triage.slurm.
export MODEL_CACHE_DIR=/user/work/$USER/hf_cache/hub
# Offline Docling: layout/tableformer/EasyOCR weights prefetched on the login node.
export DOCLING_ARTIFACTS_PATH=/user/work/$USER/docling_artifacts
export GRANITE_VISION_MODEL_ID=ibm-granite/granite-vision-3.3-2b
export LLM_DEVICE=cuda
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

echo "=== ingest runs/ocr-smoke/input with caption + in-figure OCR ==="
# `python -m` (not the evidence-rag-ingest console script) so a fresh `pip install`
# is not required for the entry point to resolve.
python -m evidence_rag.cli.ingest \
  --input  runs/ocr-smoke/input \
  --output runs/ocr-smoke/out \
  --config configs/ingestion/ocr_smoke.toml

echo "=== verify an image-source document carries OCR'd in-figure text ==="
python - <<'PY'
import json
import sys

docs = [json.loads(line) for line in open("runs/ocr-smoke/out/documents.jsonl")]
images = [d for d in docs if (d.get("metadata") or {}).get("source_type") == "image"]

print(f"total docs: {len(docs)} | image-source docs: {len(images)}")
for d in images:
    print("-" * 60)
    print(d["text"][:400])

def ocr_part(text: str) -> str:
    marker = "Text in image:"
    return text.split(marker, 1)[1] if marker in text else ""

# The sentinel "REVENUE 2024 42 PERCENT" lives only in the figure, never in the
# PDF body -- so its digits appearing in an image doc's OCR section prove OCR ran.
passed = any(
    "42" in ocr_part(d["text"]) and "Text in image:" in d["text"]
    for d in images
)
print("=" * 60)
print("OCR SMOKE:", "PASS" if passed else "FAIL")
sys.exit(0 if passed else 1)
PY
