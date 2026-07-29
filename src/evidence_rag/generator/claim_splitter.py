import json
import re

from evidence_rag.generator.granite import GraniteLLMClient, TextGenerator
from evidence_rag.generator.json_parsing import parse_json_object
from evidence_rag.generator.models import Claim, ClaimSpan

SPLIT_PROMPT = (
    "Split the answer into atomic, self-contained factual claims. Resolve pronouns "
    "without adding information. Return JSON only in this shape: "
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
_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]+|\S[^.!?]*$")
_MIN_SENTENCE_OVERLAP = 0.3
"""Minimum share of the claim's content tokens that must appear in an answer
sentence before we anchor to it; below this the claim is treated as unlocatable
and skipped (a genuinely off-answer claim, not a paraphrase)."""


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    return [
        (match.start(), match.end())
        for match in _SENTENCE_RE.finditer(text)
        if match.group().strip()
    ]


def _locate_span(answer_text: str, source_text: str, claim_text: str, cursor: int) -> ClaimSpan | None:
    """Find the answer region this claim came from, at or after ``cursor``.

    1. exact substring (verbatim ``source_text``) -- unchanged fast path;
    2. otherwise the best token-overlapping answer sentence, if the overlap
       clears ``_MIN_SENTENCE_OVERLAP``; else ``None`` (claim is skipped).
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
        if end <= cursor:
            continue
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
            pending.append((claim_id, source_text, claim_text, span))
            cursor = span.end

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
