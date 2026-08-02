"""QA2D hypothesis form, served from a pre-generated cache (design §2.4, ablation rung 3).

A QA2D-style converter (Chen, Choi & Durrett, "Can NLI Models Verify QA Systems' Predictions?",
Findings of EMNLP 2021) fuses a question and an answer into a natural declarative sentence:

    ("who won the 1960 election", "Kennedy") -> "Kennedy won the 1960 election."

It is a seq2seq model, so unlike the two deterministic rungs in `claims.py` it is neither
reproducible by inspection nor available where the probe runs: this cluster's compute nodes are
fully offline. Generation therefore happens ONCE on the login node
(`evidence_rag.cli.export_qa2d`, greedy decoding) and this module only ever READS the resulting
cache. Nothing here imports torch or transformers.

A cache miss is a hard error, never a fallback to the frozen template. Falling back would mix
rung-0 rows into the QA2D arm, and the resulting file would still look well formed while the
comparison it exists to support had quietly stopped being a comparison.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

QUESTION_FIELD = "question"
ANSWER_FIELD = "answer"
DECLARATIVE_FIELD = "declarative"

CacheKey = tuple[str, str]


class Qa2dCacheMiss(LookupError):
    """The probe asked for a pair the pre-generated cache does not carry."""


def cache_key(question: str, answer: str) -> CacheKey:
    """The one normalisation both the generator and the lookup must use.

    If the writer and the reader ever disagree here, every lookup misses and the arm cannot run
    at all — so this function is the single definition, imported by both sides rather than
    reimplemented.
    """
    return (question.strip(), answer.strip())


@dataclass(frozen=True)
class Qa2dLookup:
    """A `HypothesisForm` backed by the cache instead of by a model."""

    declarative_by_pair: Mapping[CacheKey, str]

    def __call__(self, question: str, answer: str) -> str:
        key = cache_key(question, answer)
        declarative = self.declarative_by_pair.get(key)
        if declarative is None:
            raise Qa2dCacheMiss(
                f"no QA2D sentence cached for question={key[0]!r} answer={key[1]!r}; "
                f"regenerate the cache on the login node with evidence_rag.cli.export_qa2d "
                f"(cache holds {len(self.declarative_by_pair)} pairs)"
            )
        return declarative


def load_qa2d_cache(path: Path) -> Qa2dLookup:
    """Read the JSONL cache written by `evidence_rag.cli.export_qa2d`."""
    declarative_by_pair: dict[CacheKey, str] = {}
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        key = cache_key(row[QUESTION_FIELD], row[ANSWER_FIELD])
        declarative = str(row[DECLARATIVE_FIELD])
        previous = declarative_by_pair.get(key)
        if previous is not None and previous != declarative:
            raise ValueError(
                f"conflicting QA2D sentences for {key!r} at {path}:{line_number}: "
                f"{previous!r} then {declarative!r}"
            )
        declarative_by_pair[key] = declarative
    return Qa2dLookup(declarative_by_pair)
