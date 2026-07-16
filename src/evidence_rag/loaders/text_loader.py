"""Plain-text loader: read ``.txt``/``.md`` files into ``Document`` records."""

from pathlib import Path

from evidence_rag.contracts.models import Document, SourceMetadata

TEXT_EXTENSIONS = frozenset({".txt", ".md"})


def load_text_file(path: str | Path) -> Document:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Text file not found: {resolved}")
    return Document(
        document_id=resolved.name,
        text=resolved.read_text(encoding="utf-8"),
        source_uri=str(resolved),
        metadata=SourceMetadata(source_type="txt", file_name=resolved.name),
    )
