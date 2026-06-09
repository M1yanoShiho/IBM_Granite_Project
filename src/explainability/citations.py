"""Answer provenance / citations.

Maps spans of a generated answer to the retrieved chunks that support them, so
the system can show *where* each claim came from. This is the basis for the
"trust in retrieved outputs" requirement and underpins faithfulness checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.retrieval.retriever import RetrievedChunk


@dataclass
class Citation:
    """A link between a piece of the answer and its supporting source.

    Attributes
    ----------
    answer_span:
        The portion of the generated answer being attributed.
    source_chunk_id:
        Identifier of the supporting chunk.
    score:
        Confidence/support score for the attribution.
    """

    answer_span: str
    source_chunk_id: str
    score: float


def attribute_answer(
    answer: str,
    retrieved: List[RetrievedChunk],
) -> List[Citation]:
    """Attribute parts of ``answer`` to the ``retrieved`` chunks that support them."""
    raise NotImplementedError(
        "TODO: align answer spans to supporting chunks (e.g. via embedding overlap)."
    )
