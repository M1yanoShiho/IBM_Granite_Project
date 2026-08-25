"""CLI: ingest a directory of txt/PDF/image files into a ``documents.jsonl``.

Ingestion switches (PDF chunking mode, embedded-picture captioning, in-figure
OCR, vision model) are read from an ``[ingestion]`` TOML section via
``load_ingestion_config``; omit ``--config`` to use the defaults. This is the
config-file front door for ``loaders.load_directory``.
"""

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.infrastructure.config import IngestionConfig, load_ingestion_config
from evidence_rag.loaders.dispatch import load_directory_from_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest a directory of txt/PDF/image files into documents.jsonl"
    )
    parser.add_argument("--input", required=True, type=Path, help="source directory to ingest")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="output directory (documents.jsonl is written here)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="TOML file with an [ingestion] section (defaults when omitted)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = (
        load_ingestion_config(arguments.config)
        if arguments.config is not None
        else IngestionConfig()
    )

    documents = load_directory_from_config(arguments.input, config)

    output_dir = arguments.output
    output_dir.mkdir(parents=True, exist_ok=True)
    documents_path = output_dir / "documents.jsonl"
    documents_path.write_text(
        "".join(f"{document.model_dump_json()}\n" for document in documents),
        encoding="utf-8",
    )

    by_source = Counter(
        document.metadata.source_type if document.metadata is not None else "txt"
        for document in documents
    )
    print(
        json.dumps(
            {
                "documents": str(documents_path),
                "document_count": len(documents),
                "by_source_type": dict(sorted(by_source.items())),
                "caption_pdf_pictures": config.caption_pdf_pictures,
                "image_ocr": config.image_ocr,
                "pdf_mode": config.pdf_mode,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
