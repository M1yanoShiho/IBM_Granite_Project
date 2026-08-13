"""Count what a fixed word window actually breaks, on real documents.

`SectionChunker` (R7) is justified by two claims: a word window cuts tables in half, and it
severs headings from the sections they title. Both were demonstrated on a synthetic fixture
that was written to demonstrate them, which shows the failures are *possible*, not that they
are *common*. Whether `section` is worth recommending depends on whether the rate on real
documents is 3% or 40%, and nothing measured that.

This measures it, and deliberately measures only that. It reports structural damage, not
retrieval quality — quality needs queries and gold labels, which a pile of documents does
not provide, and conflating the two is how a mechanism gets sold as an improvement.

Method, per document and per chunker:

- **Tables.** A table is *intact* if some single chunk contains its whole token sequence
  contiguously, and *split* otherwise. Comparison is on token sequences rather than raw
  text because the word chunker collapses newlines, so a raw-substring test would report
  every table as split regardless of where the cut fell.
- **Headings.** A heading is *severed* if no single chunk holds both the heading and the
  opening of the section it introduces. A chunk carrying a heading whose body starts
  elsewhere cannot be matched on its own subject, which is the failure being counted.

Structure is parsed with the same `_parse_blocks` the section chunker uses, so "what counts
as a table" is identical for both arms and the comparison cannot be rigged by disagreeing
about the ground truth.

Usage (CPU only, no Docling needed for .md/.txt):

    PYTHONPATH=src python scripts/chunker_structure_audit.py docs --output results/chunk-audit.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evidence_rag.contracts.models import Document
from evidence_rag.infrastructure.corpus import _parse_blocks, build_chunker

BODY_PROBE = 8  # tokens of the following block that must travel with a heading


def _tokens(text: str) -> list[str]:
    return text.split()


def _contains(haystack: list[str], needle: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    first = needle[0]
    span = len(needle)
    for index in range(len(haystack) - span + 1):
        if haystack[index] == first and haystack[index : index + span] == needle:
            return True
    return False


def _audit_document(document: Document, chunker_name: str, *, chunk_size: int, overlap: int) -> dict[str, int]:
    chunker = build_chunker(chunker_name, chunk_size=chunk_size, overlap=overlap)
    chunk_tokens = [_tokens(chunk.text) for chunk in chunker.chunk(document)]
    blocks = _parse_blocks(document.text)

    tables = splits = headings = severed = 0
    for position, block in enumerate(blocks):
        if block.kind == "table":
            tables += 1
            if not any(_contains(tokens, _tokens(block.text)) for tokens in chunk_tokens):
                splits += 1
        elif block.kind == "heading":
            following = blocks[position + 1] if position + 1 < len(blocks) else None
            if following is None or following.kind == "heading":
                # A heading immediately followed by another heading titles nothing yet;
                # counting it would inflate the rate with cases that cannot be severed.
                continue
            headings += 1
            probe = _tokens(following.text)[:BODY_PROBE]
            together = any(
                _contains(tokens, _tokens(block.text)) and _contains(tokens, probe)
                for tokens in chunk_tokens
            )
            if not together:
                severed += 1

    return {
        "chunks": len(chunk_tokens),
        "tables": tables,
        "tables_split": splits,
        "headings": headings,
        "headings_severed": severed,
    }


def _load(directory: Path) -> list[Document]:
    documents = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue
        documents.append(
            Document(
                document_id=str(path.relative_to(directory)).replace("\\", "/"),
                text=text,
                source_uri=str(path),
            )
        )
    return documents


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--chunk-size", type=int, default=120)
    parser.add_argument("--overlap", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    documents = _load(arguments.directory)
    if not documents:
        print(f"no .md/.txt documents under {arguments.directory}", file=sys.stderr)
        return 1

    totals: dict[str, dict[str, int]] = {}
    for name in ("word", "section"):
        aggregate = {"chunks": 0, "tables": 0, "tables_split": 0, "headings": 0, "headings_severed": 0}
        for document in documents:
            counts = _audit_document(
                document, name, chunk_size=arguments.chunk_size, overlap=arguments.overlap
            )
            for key, value in counts.items():
                aggregate[key] += value
        totals[name] = aggregate

    print(f"{len(documents)} documents, chunk_size={arguments.chunk_size} overlap={arguments.overlap}\n")
    print(f"{'chunker':>8} {'chunks':>7} {'tables':>7} {'split':>7} {'split %':>8} {'headings':>9} {'severed':>8} {'severed %':>10}")
    for name, row in totals.items():
        split_rate = row["tables_split"] / row["tables"] * 100 if row["tables"] else 0.0
        severed_rate = row["headings_severed"] / row["headings"] * 100 if row["headings"] else 0.0
        print(
            f"{name:>8} {row['chunks']:7d} {row['tables']:7d} {row['tables_split']:7d} "
            f"{split_rate:7.1f}% {row['headings']:9d} {row['headings_severed']:8d} {severed_rate:9.1f}%"
        )

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(
                {
                    "directory": str(arguments.directory),
                    "documents": len(documents),
                    "chunk_size": arguments.chunk_size,
                    "overlap": arguments.overlap,
                    "totals": totals,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
