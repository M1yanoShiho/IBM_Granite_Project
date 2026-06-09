"""Data processing utilities for the Needle in a Haystack (NIAH) experiment.

This module is responsible for turning raw "haystack" documents and synthetic
"needle" facts into evaluation-ready prompts. The core idea of NIAH is:

1. Take a long document (the *haystack*).
2. Trim/pad it to a target **context length** (e.g. 1k, 4k, 16k tokens).
3. Insert a known fact (the *needle*) at a controlled **depth** (e.g. 0%, 25%,
   50%, 75%, 100% of the way through the document).
4. Ask the model a question whose answer is the needle, and check whether it
   was retrieved.

Scope: this module belongs to the **secondary** long-context stress test only.
General document chunking for the primary RAG/retrieval pipeline lives in
``src.ingestion.chunker`` — do not duplicate it here.

Functions here are intentionally left as documented skeletons — fill in the
implementation as the project develops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Needle:
    """A single synthetic fact to inject into a haystack.

    Attributes
    ----------
    text:
        The sentence inserted into the document (e.g. "The secret code is 42.").
    question:
        The question used to probe retrieval (e.g. "What is the secret code?").
    answer:
        The ground-truth answer used for scoring (e.g. "42").
    """

    text: str
    question: str
    answer: str


@dataclass
class HaystackSample:
    """A fully assembled NIAH test case ready to send to a model.

    Attributes
    ----------
    context:
        The haystack document with the needle injected.
    needle:
        The needle that was injected into ``context``.
    context_length:
        Target context length (in tokens) for this sample.
    depth_percent:
        Depth at which the needle was injected, as a percentage (0–100).
    """

    context: str
    needle: Needle
    context_length: int
    depth_percent: float
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_haystack(path: str | Path) -> str:
    """Load a raw haystack document from disk.

    Parameters
    ----------
    path:
        Path to a text file under ``data/raw/``.

    Returns
    -------
    str
        The full document contents as a single string.
    """
    raise NotImplementedError("TODO: read and return the document text.")


def load_needles(path: str | Path) -> List[Needle]:
    """Load synthetic needle facts from disk.

    Parameters
    ----------
    path:
        Path to a file under ``data/needles/`` (e.g. JSON or CSV) describing
        the needle text, probe question, and expected answer.

    Returns
    -------
    list of Needle
        The parsed needles.
    """
    raise NotImplementedError("TODO: parse and return needle definitions.")


# --------------------------------------------------------------------------- #
# Needle injection
# --------------------------------------------------------------------------- #
def inject_needle(
    haystack: str,
    needle: Needle,
    depth_percent: float,
) -> str:
    """Insert ``needle`` into ``haystack`` at a given relative depth.

    The needle is placed near the sentence/paragraph boundary closest to
    ``depth_percent`` so the surrounding text stays coherent.

    Parameters
    ----------
    haystack:
        The source document.
    needle:
        The fact to inject.
    depth_percent:
        Where to inject, as a percentage of document length (0 = start,
        100 = end).

    Returns
    -------
    str
        The haystack with the needle injected.
    """
    raise NotImplementedError("TODO: insert needle at the target depth.")


def build_sample(
    haystack: str,
    needle: Needle,
    context_length: int,
    depth_percent: float,
) -> HaystackSample:
    """Assemble a single NIAH test case.

    Trims/pads ``haystack`` to ``context_length``, injects ``needle`` at
    ``depth_percent``, and wraps everything in a :class:`HaystackSample`.

    Parameters
    ----------
    haystack:
        The source document.
    needle:
        The fact to inject.
    context_length:
        Target context length in tokens.
    depth_percent:
        Injection depth as a percentage (0–100).

    Returns
    -------
    HaystackSample
        The assembled, ready-to-evaluate sample.
    """
    raise NotImplementedError("TODO: trim to length, inject needle, wrap.")
