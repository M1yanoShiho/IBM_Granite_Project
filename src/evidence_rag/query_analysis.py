import re
from typing import Protocol

from evidence_rag.contracts.models import Query, QueryChecklist

WORD = re.compile(r"[A-Za-z][A-Za-z0-9&.-]*|\d+(?:\.\d+)?%?")
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}
METRICS = {
    "revenue",
    "growth",
    "profit",
    "margin",
    "income",
    "cash",
    "assets",
    "liabilities",
    "termination",
    "notice",
    "clause",
    "rate",
    "percentage",
}


class QueryAnalyzer(Protocol):
    def analyze(self, query: Query) -> QueryChecklist: ...


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0) for match in WORD.finditer(text))


def _content_terms(text: str) -> tuple[str, ...]:
    seen: set[str] = set()
    terms: list[str] = []
    for token in _tokens(text):
        lowered = token.lower()
        if lowered in STOPWORDS or len(lowered) < 2:
            continue
        if lowered not in seen:
            seen.add(lowered)
            terms.append(token)
    return tuple(terms)


class RuleBasedQueryAnalyzer:
    """Small default analyzer that builds a per-query task checklist."""

    def analyze(self, query: Query) -> QueryChecklist:
        terms = _content_terms(query.text)
        focus = " ".join(terms[:6]) if terms else query.text.strip()
        required = tuple(term for term in terms if term.lower() in METRICS)
        if not required:
            required = (focus,)

        constraints: list[str] = []
        for year in YEAR.findall(query.text):
            constraints.append(f"year:{year}")
        for token in terms:
            if token[:1].isupper():
                constraints.append(f"entity:{token}")

        return QueryChecklist(
            query_id=query.query_id,
            focus=focus,
            required_facts=required,
            constraints=tuple(dict.fromkeys(constraints)),
        )
