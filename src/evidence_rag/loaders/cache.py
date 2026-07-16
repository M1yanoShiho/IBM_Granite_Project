"""File-level ingestion cache: skip re-parsing / re-captioning unchanged sources.

Parsing a PDF with Docling or captioning an image with Granite Vision is
orders of magnitude more expensive than hashing the source file, so the cache
key is ``sha256(file bytes + parameter fingerprint)``. The fingerprint must
encode every parameter that affects loader output (mode, prompts, model ids,
loader versions); any change naturally invalidates old entries.

Hashing reads the whole source file on every lookup — a deliberate
correctness-over-speed trade-off. If re-ingesting very large PDFs frequently
ever becomes a bottleneck, a ``size + mtime`` fast path could be added at the
cost of missing same-size in-place edits.
"""

import hashlib
import json
import logging
from pathlib import Path

from pydantic import ValidationError

from evidence_rag.contracts.models import Document

logger = logging.getLogger("evidence_rag.loaders")


class DocumentCache:
    """Stores one JSON file per (source file, fingerprint) with its ``Document`` list."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(path: Path, fingerprint: str) -> str:
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        digest.update(b"\x00")
        digest.update(fingerprint.encode("utf-8"))
        return digest.hexdigest()

    def _entry(self, path: Path, fingerprint: str) -> Path:
        return self.cache_dir / f"{self._key(path, fingerprint)}.json"

    def get(self, path: Path, fingerprint: str) -> list[Document] | None:
        """Return the cached documents for ``path``, or None on miss/corruption."""
        entry = self._entry(path, fingerprint)
        if not entry.is_file():
            return None
        try:
            payload = json.loads(entry.read_text(encoding="utf-8"))
            documents = [Document.model_validate(item) for item in payload]
        except (json.JSONDecodeError, ValidationError, TypeError):
            logger.warning("Ignoring corrupt ingestion cache entry %s", entry, exc_info=True)
            return None
        logger.debug("Ingestion cache hit for %s", path)
        return documents

    def put(self, path: Path, fingerprint: str, documents: list[Document]) -> None:
        payload = [document.model_dump(mode="json") for document in documents]
        self._entry(path, fingerprint).write_text(json.dumps(payload), encoding="utf-8")
