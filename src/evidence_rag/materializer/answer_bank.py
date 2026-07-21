"""Frozen, seeded replacement bank (spec §5).

Replacement values are real answer strings from the corpus, grouped by mechanical class,
so a counterfactual is a plausible same-class value. Selection is deterministic
(seed + class + gold value) and excludes anything canonically equal to the gold.
"""

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256

from evidence_rag.selector.answer_norm import canonicalize_answer

_INTEGER = re.compile(r"\d+")
_DECIMAL = re.compile(r"\d+\.\d+")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def string_class(surface: str) -> str:
    canonical = canonicalize_answer(surface)
    if _ISO_DATE.fullmatch(canonical):
        return "date"
    if _INTEGER.fullmatch(canonical):
        value = int(canonical)
        return "year" if 1000 <= value <= 2999 else "integer"
    if _DECIMAL.fullmatch(canonical):
        return "decimal"
    tokens = surface.split()
    kind = "name" if surface[:1].isupper() else "noun"
    return f"{kind}-{len(tokens)}"


class AnswerBank:
    def __init__(self, by_class: Mapping[str, Iterable[str]], *, seed: int = 42) -> None:
        self._by_class = {key: tuple(sorted(set(values))) for key, values in by_class.items()}
        self._seed = seed
        payload = json.dumps(
            {"seed": seed, "by_class": {k: list(v) for k, v in self._by_class.items()}},
            sort_keys=True,
            ensure_ascii=True,
        )
        self._hash = sha256(payload.encode("utf-8")).hexdigest()

    @property
    def content_hash(self) -> str:
        return self._hash

    def select(
        self,
        gold_value: str,
        gold_aliases: Sequence[str],
        string_class: str,
    ) -> str | None:
        blocked = {canonicalize_answer(gold_value)} | {
            canonicalize_answer(alias) for alias in gold_aliases
        }
        candidates = [
            value
            for value in self._by_class.get(string_class, ())
            if canonicalize_answer(value) not in blocked
        ]
        if not candidates:
            return None
        digest = sha256(
            f"{self._seed}:{string_class}:{gold_value}".encode("utf-8")
        ).hexdigest()
        return candidates[int(digest, 16) % len(candidates)]


def build_answer_bank(values: Iterable[str], *, seed: int = 42) -> AnswerBank:
    by_class: dict[str, list[str]] = {}
    for value in values:
        by_class.setdefault(string_class(value), []).append(value)
    return AnswerBank(by_class, seed=seed)
