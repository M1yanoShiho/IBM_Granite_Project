import json
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


class TextGenerator(Protocol):
    """Structural stand-in for ``generator.granite.TextGenerator``.

    Declared here rather than imported so this module keeps no dependency on the
    Generator package: ``pipeline`` is allowed to import ``query_analysis``, and
    pulling a concrete model client in through it would drag transformers into
    every pipeline construction. Protocols are structural, so ``GraniteLLMClient``
    satisfies this without either module knowing about the other.
    """

    def generate(self, prompt: str) -> str: ...


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


def _constraints(text: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    constraints: list[str] = []
    for year in YEAR.findall(text):
        constraints.append(f"year:{year}")
    for token in terms:
        if token[:1].isupper():
            constraints.append(f"entity:{token}")
    return tuple(dict.fromkeys(constraints))


class RuleBasedQueryAnalyzer:
    """Small default analyzer that builds a per-query task checklist.

    Built for enterprise documents: ``required_facts`` come from the ``METRICS``
    vocabulary. On open-domain questions that vocabulary does not fire, so the
    checklist is legitimately empty -- a domain limit, not a defect. Kept
    alongside ``GraniteQueryAnalyzer`` so that limit can be reported as such.
    """

    def analyze(self, query: Query) -> QueryChecklist:
        terms = _content_terms(query.text)
        focus = " ".join(terms[:6]) if terms else query.text.strip()
        # When the metrics vocabulary does not fire there is no genuine
        # requirement to state, so emit NOTHING. The previous fallback used
        # ``focus`` -- a bag of question tokens -- as a required fact, which asked
        # the completeness checker "has the answer stated 'can adipose tissue
        # found body'?" on every open-domain query: a malformed question, and a
        # category error since focus is a topic, not a fact.
        required = tuple(term for term in terms if term.lower() in METRICS)

        return QueryChecklist(
            query_id=query.query_id,
            focus=focus,
            required_facts=required,
            constraints=_constraints(query.text, terms),
        )


CHECKLIST_PROMPT = (
    "A question can have more than one reasonable interpretation. List what a "
    "complete answer is obliged to cover.\n\n"
    "Rules:\n"
    "- Write each item as an obligation on the answer, never as a fact. Say what "
    "the answer must specify; do not say what the answer is. You do not know the "
    "answer and must not guess it.\n"
    "- Only list an item when the question genuinely has that separate reading.\n"
    "- If the question has a single reasonable interpretation, return an empty list.\n\n"
    'Return JSON only, in this shape: {{"requirements": ["The answer must specify ..."]}}\n\n'
    "Question: {question}"
)


def _extract_json_object(raw: str) -> dict[str, object]:
    """First complete JSON object in ``raw``, tolerating fences and trailing prose.

    A deliberate small duplicate of ``generator.json_parsing.parse_json_object``:
    importing it would make ``query_analysis`` -- and therefore ``pipeline`` --
    depend on the Generator package.
    """
    text = raw.strip()
    if text.startswith("```"):
        newline = text.find("\n")
        text = text[newline + 1 :] if newline != -1 else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    start = text.find("{")
    if start == -1:
        return {}
    try:
        data, _end = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


class GraniteQueryAnalyzer:
    """Build the checklist with an LLM, from the question text and nothing else.

    A rule-based analyzer cannot know that a question has several readings --
    anticipating ambiguity needs world knowledge -- so the completeness mechanism
    had never had a fair test. This asks the model what a complete answer is
    *obliged to cover*.

    **Input is the question text only.** No gold answers, no ``qa_pairs``, no
    annotator fields, no retrieved evidence: ``QueryChecklist`` is an input to the
    Generator, so anything gold-derived would be oracle leakage.

    **Requirements, not assertions.** "The answer must specify the men's
    international record holder" is a requirement; "Ali Daei holds the record" is
    an assertion the analyzer cannot know, and a hallucinated one would
    manufacture spurious gaps -- the failure mode this construction exists to
    avoid. Whether the model honours that distinction is measured, not assumed.

    **Zero requirements is a valid answer.** Most questions have one reading and
    carry no completeness obligation beyond being answered. Forcing at least one
    item per query would recreate the always-fires problem.
    """

    def __init__(
        self,
        llm: TextGenerator,
        prompt_template: str = CHECKLIST_PROMPT,
        max_requirements: int = 6,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_requirements = max_requirements

    def analyze(self, query: Query) -> QueryChecklist:
        terms = _content_terms(query.text)
        focus = " ".join(terms[:6]) if terms else query.text.strip()
        data = _extract_json_object(
            self.llm.generate(self.prompt_template.format(question=query.text))
        )
        raw_requirements = data.get("requirements")
        requirements: list[str] = []
        if isinstance(raw_requirements, list):
            for entry in raw_requirements:
                if not isinstance(entry, str):
                    continue
                cleaned = " ".join(entry.split())
                if cleaned and cleaned not in requirements:
                    requirements.append(cleaned)
        # A malformed or missing list yields an empty checklist rather than an
        # exception: an unusable analyzer response is "no known obligation", which
        # is a legal state, and crashing the query would bias the sample.
        return QueryChecklist(
            query_id=query.query_id,
            focus=focus,
            required_facts=tuple(requirements[: self.max_requirements]),
            constraints=_constraints(query.text, terms),
        )
