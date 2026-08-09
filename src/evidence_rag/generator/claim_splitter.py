import json
import re

from evidence_rag.contracts.models import sentence_spans
from evidence_rag.generator.granite import GraniteLLMClient, TextGenerator
from evidence_rag.generator.json_parsing import parse_json_object
from evidence_rag.generator.models import Claim, ClaimSpan

SPLIT_PROMPT = (
    "Split the answer into atomic, self-contained factual claims that could each "
    "be checked against source documents. Resolve pronouns without adding "
    "information. Skip any sentence that only describes the answer itself or where "
    "its information came from (e.g. 'this is based on...', 'the information is "
    "sourced from...') rather than asserting a fact about the world. Return JSON "
    "only in this shape: "
    '{{"claims":[{{"source_text":"exact contiguous text from the answer",'
    '"text":"self-contained claim"}}]}}.\n\nAnswer:\n{answer}'
)

# Real Granite frequently returns a `source_text` that PARAPHRASES the answer
# rather than quoting it verbatim (the dominant G2 chain error). We keep the exact
# substring as the fast path, then fall back to anchoring the claim on the answer
# sentence it overlaps most -- so a paraphrased source_text still yields a real
# answer span for repair, instead of aborting the whole verify->repair chain.
_STOPWORDS = frozenset(
    "a an the of in on at to for and or is are was were be been by with as that this it its "
    "from has have had also his her their he she they there which who whom but not into than "
    "then when while about over under after before".split()
)
_MIN_SENTENCE_OVERLAP = 0.3
"""Minimum share of the claim's content tokens that must appear in an answer
sentence before we anchor to it; below this the claim is treated as unlocatable
and skipped (a genuinely off-answer claim, not a paraphrase)."""

# --- over-split suppression -------------------------------------------------
# The audit found three kinds of claim that are not answers to the question:
# meta-narrative about the answer itself, fragments whose subject was never
# resolved, and restatements of another claim. They inflate the unsupported and
# abstention rates, multiply entity_check comparisons, and lengthen answers --
# which feeds the citation-dilution problem directly.

_META_NARRATIVE = re.compile(
    r"\b(?:"
    r"the (?:answer|information|evidence|passage|document|source|text|context)s?\b"
    r"|this (?:answer|information|statement)\b"
    r"|(?:is|was|are|were) (?:sourced|derived|taken) from"
    r"|comes from the"
    r"|according to the (?:evidence|passage|document|source|text|context)"
    r"|mentioned in the (?:answer|evidence|passage|document|text)"
    r")",
    re.IGNORECASE,
)

_UNRESOLVED_SUBJECT = re.compile(
    r"^(?:it|its|this|that|these|those|they|them|their|he|him|his|she|her)\b",
    re.IGNORECASE,
)

def _is_meta_narrative(text: str) -> bool:
    """Talks about the answer or where it came from, rather than asserting a fact."""
    return bool(_META_NARRATIVE.search(text))


def _has_unresolved_subject(text: str) -> bool:
    """Opens with a pronoun or demonstrative the splitter failed to resolve.

    A2's own contract is 'atomic, self-contained ... resolve pronouns', so
    "This film aired on NBC in 1973" is a splitter failure: nothing downstream can
    tell which film, and the verifier cannot check it against evidence.
    """
    return bool(_UNRESOLVED_SUBJECT.match(text.strip()))


def _drop_over_split(
    pending: list[tuple[str, str, str, ClaimSpan]],
) -> list[tuple[str, str, str, ClaimSpan]]:
    """Remove non-answers and true duplicates without guessing entailment.

    A longer claim is not automatically a better claim: it may combine the
    shorter fact with a distinct date, number, or entity. Lexical subsumption
    therefore cannot safely choose between them. Only claims with the same
    normalized token sequence are duplicates here; uncertain cases remain
    visible for faithfulness and evidence verification.
    """
    kept = [
        item
        for item in pending
        if not _is_meta_narrative(item[2]) and not _has_unresolved_subject(item[2])
    ]
    result: list[tuple[str, str, str, ClaimSpan]] = []
    seen: set[tuple[str, ...]] = set()
    for item in kept:
        key = tuple(re.findall(r"[a-z0-9]+", item[2].casefold()))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


_sentence_spans = sentence_spans
"""The system's one sentence rule, from ``contracts``.

This module used to carry its own, knowing 10 abbreviations against the
contract's ~50 -- so ``"Acme Inc. in Ohio"`` was two sentences to the splitter and
one to the validator. G7 lost an answer to exactly that class of disagreement
between two other copies of the rule; this was the third copy."""


def _locate_span(answer_text: str, source_text: str, claim_text: str, cursor: int) -> ClaimSpan | None:
    """Find the answer region this claim came from.

    1. exact substring (verbatim ``source_text``) at or after ``cursor``;
    2. otherwise the best token-overlapping answer sentence anywhere in the
       answer, if the overlap clears ``_MIN_SENTENCE_OVERLAP``; else ``None``.

    The fallback deliberately permits several atomic claims to share one source
    sentence. A paraphrased first claim may anchor the whole sentence and advance
    ``cursor`` past it; treating that cursor as a hard fallback boundary would
    silently discard every later claim extracted from the same sentence.
    """
    exact = answer_text.find(source_text, cursor)
    if exact >= 0:
        return ClaimSpan(start=exact, end=exact + len(source_text))
    needle = _content_tokens(source_text) | _content_tokens(claim_text)
    if not needle:
        return None
    best_overlap = 0.0
    best_span: tuple[int, int] | None = None
    for start, end in _sentence_spans(answer_text):
        sentence_tokens = _content_tokens(answer_text[start:end])
        if not sentence_tokens:
            continue
        overlap = len(needle & sentence_tokens) / len(needle)
        if overlap > best_overlap:
            best_overlap = overlap
            best_span = (start, end)
    if best_span is not None and best_overlap >= _MIN_SENTENCE_OVERLAP:
        return ClaimSpan(start=best_span[0], end=best_span[1])
    return None

FAITHFULNESS_PROMPT = (
    "Check whether each rewritten claim has exactly the same meaning as its source text. "
    "Return JSON only in this shape: "
    '{{"results":[{{"claim_id":"claim-1","faithful":true}}]}}.\n\nItems:\n{items}'
)


class ClaimSplitter:
    """Split a draft answer into atomic claims and independently check rewrites."""

    def __init__(self, llm: TextGenerator | None = None) -> None:
        self.llm = llm or GraniteLLMClient()

    def split(self, answer_text: str) -> tuple[Claim, ...]:
        if not answer_text.strip():
            return ()
        split_data = self._load_json(
            self.llm.generate(SPLIT_PROMPT.format(answer=answer_text))
        )
        raw_claims = split_data.get("claims")
        if not isinstance(raw_claims, list):
            raise ValueError("splitter output must contain a claims array")
        pending: list[tuple[str, str, str, ClaimSpan]] = []
        cursor = 0
        for index, item in enumerate(raw_claims, start=1):
            claim_id = f"claim-{index}"
            if not isinstance(item, dict):
                raise ValueError("each claim must contain nonempty source_text and text")
            source_text = item.get("source_text")
            claim_text = item.get("text")
            if not isinstance(source_text, str) or not source_text.strip():
                raise ValueError("each claim must contain nonempty source_text and text")
            if not isinstance(claim_text, str) or not claim_text.strip():
                raise ValueError("each claim must contain nonempty source_text and text")
            span = _locate_span(answer_text, source_text, claim_text, cursor)
            if span is None:
                # a claim we cannot anchor to any answer sentence is treated as
                # unlocatable and skipped, rather than aborting the whole chain;
                # the faithfulness self-check below and the verifier are the layers
                # that judge whether a located claim is actually right.
                continue
            # Faithfulness is defined against what A1 actually wrote, not against
            # the splitter's own (often paraphrased) ``source_text`` field.
            located_source = answer_text[span.start : span.end]
            pending.append((claim_id, located_source, claim_text, span))
            cursor = span.end

        # suppress over-split claims BEFORE the faithfulness call: they are not
        # answers, so spending a check on them is waste, and letting them through
        # lengthens the answer and multiplies entity_check comparisons.
        pending = _drop_over_split(pending)
        if not pending:
            return ()

        items = json.dumps(
            [
                {"claim_id": claim_id, "source_text": source, "claim_text": text}
                for claim_id, source, text, _span in pending
            ],
            ensure_ascii=False,
        )
        check_data = self._load_json(
            self.llm.generate(FAITHFULNESS_PROMPT.format(items=items))
        )
        raw_results = check_data.get("results")
        if not isinstance(raw_results, list):
            raise ValueError("faithfulness output must contain a results array")
        faithful_by_id = {
            item["claim_id"]: item["faithful"] for item in raw_results
        }
        expected_ids = {claim_id for claim_id, _source, _text, _span in pending}
        if set(faithful_by_id) != expected_ids or len(faithful_by_id) != len(
            raw_results
        ):
            raise ValueError("faithfulness results must cover exactly the split claims")
        return tuple(
            Claim(
                claim_id=claim_id,
                text=text,
                span=span,
                faithful_to_answer=faithful_by_id[claim_id],
            )
            for claim_id, _source, text, span in pending
        )

    @staticmethod
    def _load_json(raw: str) -> dict[str, object]:
        # tolerate the trailing prose a real Granite model appends after the JSON
        return parse_json_object(raw)
