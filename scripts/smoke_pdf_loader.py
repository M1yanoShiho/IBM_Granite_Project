#!/usr/bin/env python3
"""Smoke-test the Docling PDF loader from the terminal.

Parses one or more PDFs and prints each page's Markdown plus its provenance
metadata, so you can eyeball the conversion before it goes anywhere near the
index. Requires ``docling`` (see requirements.txt); the first run downloads
Docling's layout models (a few hundred MB) into the Hugging Face cache.

    python scripts/smoke_pdf_loader.py path/to/file.pdf [more.pdf ...]
    python scripts/smoke_pdf_loader.py file.pdf --max-chars 0   # full text
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.loaders.pdf_loader import build_converter, load_pdf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pdfs", nargs="+", help="PDF file(s) to convert")
    ap.add_argument(
        "--max-chars",
        type=int,
        default=800,
        help="Markdown preview length per page (0 = print everything)",
    )
    args = ap.parse_args()

    print("Loading Docling converter (first run downloads layout models)...")
    converter = build_converter()

    failures = 0
    for pdf in args.pdfs:
        t0 = time.perf_counter()
        try:
            records = load_pdf(pdf, converter=converter)
        except Exception as exc:  # keep going: report per-file failures at the end
            print(f"\n=== {pdf}: FAILED ({type(exc).__name__}: {exc}) ===")
            failures += 1
            continue
        elapsed = time.perf_counter() - t0

        print(f"\n=== {pdf}: {len(records)} record(s) in {elapsed:.1f}s ===")
        if not records:
            print("!! No text extracted — scanned PDF without OCR, or empty file?")
            failures += 1
            continue
        for rec in records:
            print(f"\n--- {rec.doc_id} | {rec.metadata} ---")
            if 0 < args.max_chars < len(rec.text):
                print(rec.text[: args.max_chars] + f"\n[... {len(rec.text)} chars total]")
            else:
                print(rec.text)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
