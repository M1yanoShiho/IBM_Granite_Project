#!/usr/bin/env python3
"""Smoke-test the Granite Vision image loader from the terminal.

Captions one or more images and prints the description plus provenance
metadata. Requires torch + transformers; the first run downloads the vision
model (~5 GB — set MODEL_CACHE_DIR to shared scratch on the HPC). On a GPU it
also prints allocated VRAM after the loader returns, which should be ~0 MiB:
that verifies the load-caption-release contract of ``caption_images``.

    python scripts/smoke_image_loader.py chart.png photo.jpg
    python scripts/smoke_image_loader.py img.png --device cpu --max-new-tokens 128
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.loaders.image_loader import (
    DEFAULT_CAPTION_PROMPT,
    DEFAULT_VISION_MODEL_ID,
    caption_images,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="+", help="Image file(s) to caption")
    ap.add_argument(
        "--model-id",
        default=None,
        help=f"Vision model repo id (default: env GRANITE_VISION_MODEL_ID "
        f"or {DEFAULT_VISION_MODEL_ID})",
    )
    ap.add_argument(
        "--device",
        default=None,
        choices=("auto", "cuda", "cpu"),
        help="Placement (default: env LLM_DEVICE or 'auto')",
    )
    ap.add_argument("--prompt", default=DEFAULT_CAPTION_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    import torch  # after arg parsing so --help works without the ML stack

    print(f"Captioning {len(args.images)} image(s); the model is loaded once "
          "for the batch and released afterwards...")
    t0 = time.perf_counter()
    records = caption_images(
        args.images,
        prompt=args.prompt,
        model_id=args.model_id,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    elapsed = time.perf_counter() - t0

    for rec in records:
        print(f"\n--- {rec.doc_id} | {rec.metadata} ---")
        print(rec.text)

    print(f"\nDone: {len(records)} caption(s) in {elapsed:.1f}s")
    if torch.cuda.is_available():
        mib = torch.cuda.memory_allocated() / 2**20
        print(f"CUDA memory still allocated after release: {mib:.1f} MiB "
              f"({'OK — VRAM freed' if mib < 64 else 'WARNING — model may not have been released'})")
    else:
        print("CUDA not available — ran on CPU, no VRAM check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
