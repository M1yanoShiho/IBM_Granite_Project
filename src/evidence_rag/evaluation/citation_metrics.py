"""Sentence-level ALCE citation metrics with an injectable independent judge."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CitationExample:
    example_id: str
    sentences: tuple[str, ...]
    citations: tuple[tuple[str, ...], ...]
    documents: Mapping[str, str]

    def __post_init__(self) -> None:
        if len(self.sentences) != len(self.citations):
            raise ValueError("each sentence needs exactly one citation list")


class CachedEntailment:
    def __init__(self, entails: Callable[[str, str], bool]) -> None:
        self._entails = entails
        self._cache: dict[tuple[str, str], bool] = {}
        self.calls = 0

    def __call__(self, premise: str, hypothesis: str) -> bool:
        key = (premise, hypothesis)
        if key not in self._cache:
            self.calls += 1
            self._cache[key] = bool(self._entails(premise, hypothesis))
        return self._cache[key]


def score_citation_examples(
    examples: Sequence[CitationExample],
    entails: Callable[[str, str], bool],
) -> tuple[dict[str, dict[str, float | int]], int]:
    """Return ALCE precision/recall per query and the unique judge-call count."""

    judge = CachedEntailment(entails)
    per_example: dict[str, dict[str, float | int]] = {}
    for example in examples:
        if not example.sentences:
            continue
        entailed_sentences = 0
        precise_citations = 0
        total_citations = 0
        for sentence, references in zip(
            example.sentences,
            example.citations,
            strict=True,
        ):
            valid = all(reference in example.documents for reference in references)
            if not references or not valid:
                joint_entailment = 0
            else:
                total_citations += len(references)
                joint = "\n".join(example.documents[reference] for reference in references)
                joint_entailment = int(judge(joint, sentence))
            entailed_sentences += joint_entailment

            if joint_entailment and len(references) > 1:
                for reference in references:
                    if judge(example.documents[reference], sentence):
                        precise_citations += 1
                        continue
                    rest = tuple(other for other in references if other != reference)
                    if not judge(
                        "\n".join(example.documents[other] for other in rest),
                        sentence,
                    ):
                        precise_citations += 1
            else:
                precise_citations += joint_entailment

        per_example[example.example_id] = {
            "precision": precise_citations / total_citations if total_citations else 0.0,
            "recall": entailed_sentences / len(example.sentences),
            "sentences": len(example.sentences),
            "citations": total_citations,
        }
    return per_example, judge.calls


def citation_example_to_json(example: CitationExample) -> dict[str, Any]:
    """Development/debug helper; formal runs do not emit document text to logs."""

    return {
        "example_id": example.example_id,
        "sentences": list(example.sentences),
        "citations": [list(items) for items in example.citations],
        "documents": dict(example.documents),
    }
