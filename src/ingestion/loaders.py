"""Document loaders.

Loads source content into a uniform ``(doc_id, text)`` form. Two sources are
supported:

1. **Benchmark corpora** (BEIR / MS MARCO / Natural Questions) — used for the
   primary precision/recall evaluation; see ``eval.benchmarks.loader``.
2. **Real documents** (PDF / Word / HTML / plain text) — used by the demo app
   and any enterprise-style scenario.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple


def load_text_file(path: str | Path) -> str:
    """Load a single plain-text document."""
    raise NotImplementedError("TODO: read and return the document text.")


def load_documents(directory: str | Path) -> Iterable[Tuple[str, str]]:
    """Yield ``(doc_id, text)`` for every supported document in ``directory``.

    Supported formats: ``.txt``, ``.md``, and (later) ``.pdf`` / ``.docx``.
    """
    raise NotImplementedError("TODO: walk directory, parse files, yield (id, text).")
