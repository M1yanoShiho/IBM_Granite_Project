"""Experiment 05 six-metric scorer contract."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from evidence_rag.contracts.models import sentence_spans
from evidence_rag.generator.json_parsing import parse_json_object

CANONICAL_ABSTENTION = "I cannot answer because the provided evidence is insufficient."


class EntailmentJudge(Protocol):
    def entails(self, premise: str, hypothesis: str) -> bool: ...


class NLIClassifier(Protocol):
    def classify(self, premise: str, hypothesis: str) -> str: ...


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


class NLIEntailmentJudge:
    """Small scorer-only adapter around the frozen MiniCheck classifier."""

    def __init__(self, model: NLIClassifier) -> None:
        self.model = model

    def entails(self, premise: str, hypothesis: str) -> bool:
        return self.model.classify(premise, hypothesis) == "entailment"


@dataclass(frozen=True)
class ExtractedClaim:
    claim_id: str
    text: str
    sentence_index: int


@dataclass(frozen=True)
class ClaimExtractionRecord:
    claim: ExtractedClaim
    source_text: str
    source_start: int
    source_end: int


SCORER_CLAIM_PROMPT = (
    "Extract the atomic factual claims asserted by the raw answer. Use the question "
    "only to make a short answer self-contained. Do not extract citation markers, "
    "statements about the answer/evidence, or statements that evidence is insufficient. "
    "Rewrite minimally and do not add implied details or background knowledge. "
    "For example, if the answer says only 'The Moon is Earth's natural satellite', "
    "do not add that it orbits Earth. "
    "Each source_text must be an exact contiguous substring of the raw answer. Return "
    "JSON only as: "
    '{{"claims":[{{"source_text":"exact answer text","text":"self-contained claim"}}]}}.\n\n'
    "Question:\n{question}\n\nRaw answer:\n{answer}"
)

_SCORER_META = re.compile(
    r"(?:provided )?evidence (?:is|was|seems|appears)|"
    r"(?:cannot|can't|unable to) answer|"
    r"(?:this|the) answer (?:uses|is based|comes|was derived)|"
    r"according to (?:the )?(?:evidence|passage|context)",
    re.IGNORECASE,
)


def _escape_unescaped_json_quotes(raw: str) -> str:
    """Escape only quotes that cannot terminate the current JSON string.

    Granite occasionally copies a quoted nickname from ``source_text`` without
    JSON-escaping it.  A quote can terminate a JSON string only when the next
    non-whitespace character is a structural delimiter.  Repairing the other
    quotes preserves the generated text verbatim and leaves all other malformed
    responses to the normal parser error path.
    """

    repaired: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(raw):
        if not in_string:
            repaired.append(char)
            if char == '"':
                in_string = True
            continue
        if escaped:
            repaired.append(char)
            escaped = False
            continue
        if char == "\\":
            repaired.append(char)
            escaped = True
            continue
        if char != '"':
            repaired.append(char)
            continue
        following = raw[index + 1 :].lstrip()
        terminates = not following or following[0] in ":}]"
        if following.startswith(","):
            # A value-closing quote is followed by the next quoted object key.
            # A quote copied from answer text may instead precede an ordinary
            # prose comma, so the comma alone is not a structural delimiter.
            terminates = following[1:].lstrip().startswith('"')
        if terminates:
            repaired.append(char)
            in_string = False
        else:
            repaired.append('\\"')
    return "".join(repaired)


def _quote_bare_claim_keys(raw: str) -> str:
    """Quote a fixed claim key only when it occurs outside a JSON string."""

    repaired: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(raw):
        char = raw[index]
        if in_string:
            repaired.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            repaired.append(char)
            in_string = True
            index += 1
            continue
        repaired.append(char)
        index += 1
        if char not in "{,":
            continue
        match = re.match(r'(\s*)(source_text|text)(?:")?(\s*:)', raw[index:])
        if match is None:
            continue
        repaired.append(f'{match.group(1)}"{match.group(2)}"{match.group(3)}')
        index += match.end()
    return "".join(repaired)


def _balance_json_delimiters(raw: str) -> str:
    """Repair only missing or mismatched object/array delimiters outside strings."""

    closing = {"{": "}", "[": "]"}
    opening = {"}": "{", "]": "["}
    stack: list[str] = []
    repaired: list[str] = []
    in_string = False
    escaped = False
    for char in raw:
        if in_string:
            repaired.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            repaired.append(char)
            in_string = True
        elif char in closing:
            repaired.append(char)
            stack.append(char)
        elif char in opening:
            wanted = opening[char]
            if wanted in stack:
                while stack and stack[-1] != wanted:
                    repaired.append(closing[stack.pop()])
                repaired.append(char)
                stack.pop()
            elif stack:
                repaired.append(closing[stack.pop()])
            # An unmatched extra closer is omitted.
        else:
            repaired.append(char)
    while stack:
        repaired.append(closing[stack.pop()])
    return "".join(repaired)


def _close_string_before_claim_key(raw: str, *, preserve_comma: bool) -> str:
    """Insert a missing string terminator before the next fixed claim key."""

    repaired: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(raw):
        if not in_string:
            repaired.append(char)
            if char == '"':
                in_string = True
            continue
        if escaped:
            repaired.append(char)
            escaped = False
            continue
        if char == "\\":
            repaired.append(char)
            escaped = True
            continue
        if char == '"':
            repaired.append(char)
            in_string = False
            continue
        if char == "," and re.match(
            r'\s*"(?:source_text|text)"\s*:', raw[index + 1 :]
        ):
            repaired.append(',",' if preserve_comma else '",')
            in_string = False
            continue
        repaired.append(char)
    return "".join(repaired)


def _insert_missing_comma_before_claim_key(raw: str) -> str:
    """Insert a missing field delimiter before a fixed claim key."""

    return re.sub(
        r'"\s+(?="(?:source_text|text)"\s*:)',
        '",',
        raw,
    )


def _claim_sources_are_locatable(data: Mapping[str, Any], answer: str) -> bool:
    items = data.get("claims")
    return isinstance(items, list) and all(
        isinstance(item, Mapping)
        and isinstance(item.get("source_text"), str)
        and bool(item["source_text"].strip())
        and item["source_text"] in answer
        for item in items
    )


class ScorerClaimExtractor:
    """Granite scorer-only claim extraction from the same question/answer surface."""

    def __init__(self, generator: TextGenerator) -> None:
        self.generator = generator
        self.last_records: tuple[ClaimExtractionRecord, ...] = ()

    def extract(self, question: str, answer: str) -> tuple[ExtractedClaim, ...]:
        self.last_records = ()
        if not question.strip() or not answer.strip():
            return ()
        raw = self.generator.generate(
            SCORER_CLAIM_PROMPT.format(question=question, answer=answer)
        )
        try:
            data = parse_json_object(raw)
        except ValueError as original_error:
            # JSON does not define ``\'`` as an escape. Granite sometimes emits
            # it when changing double-quoted titles to apostrophes; removing the
            # unnecessary slash preserves the intended text exactly.
            base = _quote_bare_claim_keys(raw.replace("\\'", "'"))
            candidates = [
                (base, False),
                (_insert_missing_comma_before_claim_key(base), False),
            ]
            candidates.extend(
                (
                    _close_string_before_claim_key(base, preserve_comma=preserve),
                    True,
                )
                for preserve in (True, False)
            )
            data = None
            for candidate, require_all_sources in candidates:
                candidate = _escape_unescaped_json_quotes(candidate)
                candidate = _balance_json_delimiters(candidate)
                try:
                    parsed = parse_json_object(candidate)
                except ValueError:
                    continue
                if not require_all_sources or _claim_sources_are_locatable(
                    parsed, answer
                ):
                    data = parsed
                    break
            if data is None:
                raise original_error
        assert data is not None
        items = data.get("claims")
        if not isinstance(items, list):
            raise ValueError("claim extractor output must contain a claims array")
        spans = sentence_spans(answer)
        records: list[ClaimExtractionRecord] = []
        cursor = 0
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError("each extracted claim must be an object")
            source = item.get("source_text")
            text = item.get("text")
            if not isinstance(source, str) or not source.strip():
                raise ValueError("each extracted claim must have nonempty source_text")
            if not isinstance(text, str) or not text.strip():
                # Granite can omit the normalized field while still returning a
                # faithful source span.  Reuse only that exact answer substring;
                # never synthesize or rewrite claim text during recovery.
                if source not in answer:
                    raise ValueError("each extracted claim must have nonempty text")
                text = source
            if _SCORER_META.search(source) or _SCORER_META.search(text):
                continue
            start = answer.find(source, cursor)
            if start < 0:
                start = answer.find(source)
            if start < 0:
                # Non-locatable rewrites are not faithful to the raw-answer surface.
                continue
            sentence_index = next(
                (index for index, (left, right) in enumerate(spans) if left <= start < right),
                -1,
            )
            if sentence_index < 0:
                continue
            claim = ExtractedClaim(
                claim_id=f"claim-{len(records) + 1}",
                text=_normalize_text(text),
                sentence_index=sentence_index,
            )
            records.append(
                ClaimExtractionRecord(
                    claim=claim,
                    source_text=source,
                    source_start=start,
                    source_end=start + len(source),
                )
            )
            cursor = start + len(source)
        self.last_records = tuple(records)
        return tuple(record.claim for record in records)


def canonical_fact(fact_question: str, alias: str) -> str:
    question = _normalize_text(fact_question)
    answer = _normalize_text(alias)
    if not question or not answer:
        raise ValueError("canonical fact question and alias must be non-blank")
    return f'For the question "{question}", the answer is "{answer}".'


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _sentences(answer: str) -> tuple[str, ...]:
    normalized = answer.strip()
    if not normalized:
        return ()
    return tuple(
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", normalized)
        if sentence.strip()
    )


def _deduplicate_claims(
    claims: Sequence[ExtractedClaim], judge: EntailmentJudge
) -> tuple[tuple[ExtractedClaim, ...], dict[str, str]]:
    unique: list[ExtractedClaim] = []
    canonical_by_original: dict[str, str] = {}
    for claim in claims:
        if not claim.claim_id or not claim.text.strip() or claim.sentence_index < 0:
            raise ValueError("extracted claim is malformed")
        duplicate_of: str | None = None
        for existing in unique:
            if judge.entails(existing.text, claim.text) and judge.entails(
                claim.text, existing.text
            ):
                duplicate_of = existing.claim_id
                break
        if duplicate_of is None:
            unique.append(claim)
            canonical_by_original[claim.claim_id] = claim.claim_id
        else:
            canonical_by_original[claim.claim_id] = duplicate_of
    return tuple(unique), canonical_by_original


def _zero_score(sidecar: Mapping[str, Any], output: Mapping[str, Any]) -> dict[str, Any]:
    reference_count = len(sidecar.get("reference_fact_groups", []))
    return {
        "query_id": sidecar.get("query_id"),
        "metrics": {
            "rfc": 0.0,
            "vrfc": 0.0,
            "ucr": None,
            "cp": 0.0,
            "cr": 0.0,
            "rr": 0.0,
        },
        "counts": {
            "reference_facts": reference_count,
            "matched_reference_facts": 0,
            "matched_and_validly_cited_reference_facts": 0,
            "claims": 0,
            "unsupported_claims": 0,
            "citation_links": 0,
            "supporting_citation_links": 0,
            "duplicate_citation_links": 0,
            "invalid_citation_links": 0,
        },
        "runtime_error": bool(output.get("runtime_error")),
        "scorer_error": False,
    }


def score_query(
    sidecar: Mapping[str, Any],
    output: Mapping[str, Any],
    claims: Sequence[ExtractedClaim],
    judge: EntailmentJudge,
) -> dict[str, Any]:
    if output.get("query_id") != sidecar.get("query_id"):
        raise ValueError("output and sidecar query IDs differ")
    fact_groups = sidecar.get("reference_fact_groups")
    if not isinstance(fact_groups, list) or not fact_groups:
        raise ValueError("sidecar has no reference fact groups")
    answer = output.get("answer_text")
    if not isinstance(answer, str):
        raise ValueError("answer_text must be a string")
    if (
        output.get("runtime_error")
        or output.get("abstained") is True
        or not answer.strip()
        or _normalize_text(answer) == CANONICAL_ABSTENTION
        or not claims
    ):
        return _zero_score(sidecar, output)

    unique_claims, canonical_by_original = _deduplicate_claims(claims, judge)
    if not unique_claims:
        return _zero_score(sidecar, output)
    claim_by_id = {claim.claim_id: claim for claim in unique_claims}
    sentence_claims: dict[int, list[str]] = {}
    for claim in claims:
        canonical_id = canonical_by_original[claim.claim_id]
        bucket = sentence_claims.setdefault(claim.sentence_index, [])
        if canonical_id not in bucket:
            bucket.append(canonical_id)

    raw_presented = output.get("presented_evidence_records")
    if not isinstance(raw_presented, list):
        raise ValueError("presented_evidence_records must be a list")
    presented: dict[int, Mapping[str, Any]] = {}
    for record in raw_presented:
        if not isinstance(record, Mapping):
            raise ValueError("presented evidence record must be an object")
        ordinal = record.get("prompt_ordinal")
        evidence_id = record.get("evidence_id")
        text = record.get("text")
        if (
            not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or ordinal <= 0
            or ordinal in presented
            or not isinstance(evidence_id, str)
            or not evidence_id
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError("presented evidence record is malformed")
        presented[ordinal] = record

    support_cache: dict[tuple[str, str], bool] = {}

    def evidence_supports(record: Mapping[str, Any], claim: ExtractedClaim) -> bool:
        key = (str(record["evidence_id"]), claim.claim_id)
        if key not in support_cache:
            support_cache[key] = judge.entails(str(record["text"]), claim.text)
        return support_cache[key]

    supported_claim_ids = {
        claim.claim_id
        for claim in unique_claims
        if any(evidence_supports(record, claim) for record in presented.values())
    }

    valid_links: set[tuple[str, str]] = set()
    invalid_links: set[tuple[str, str]] = set()
    supporting_links: set[tuple[str, str]] = set()
    validly_cited_claims: set[str] = set()
    duplicate_links = 0
    for sentence_index, sentence in enumerate(_sentences(answer)):
        markers = re.findall(r"\[(\d+)\]", sentence)
        if not markers:
            continue
        bound_claim_ids = sentence_claims.get(sentence_index, [])
        for marker in markers:
            ordinal = int(marker)
            if not bound_claim_ids:
                key = (f"__sentence_{sentence_index}", marker)
                if key in invalid_links:
                    duplicate_links += 1
                else:
                    invalid_links.add(key)
                continue
            for claim_id in bound_claim_ids:
                record = presented.get(ordinal)
                if record is None:
                    key = (claim_id, marker)
                    if key in invalid_links:
                        duplicate_links += 1
                    else:
                        invalid_links.add(key)
                    continue
                key = (claim_id, str(record["evidence_id"]))
                if key in valid_links:
                    duplicate_links += 1
                    continue
                valid_links.add(key)
                claim = claim_by_id[claim_id]
                if evidence_supports(record, claim):
                    supporting_links.add(key)
                    validly_cited_claims.add(claim_id)

    matched_fact_ids: set[str] = set()
    matched_and_cited_fact_ids: set[str] = set()
    for group in fact_groups:
        fact_id = str(group["fact_id"])
        question = str(group["fact_question"])
        aliases = group["aliases"]
        for claim in unique_claims:
            premise = f"Question: {question}\nAnswer statement: {claim.text}"
            if any(judge.entails(premise, canonical_fact(question, str(alias))) for alias in aliases):
                matched_fact_ids.add(fact_id)
                if claim.claim_id in validly_cited_claims:
                    matched_and_cited_fact_ids.add(fact_id)
                break

    reference_count = len(fact_groups)
    claim_count = len(unique_claims)
    unsupported_count = claim_count - len(supported_claim_ids)
    link_count = len(valid_links) + len(invalid_links)
    supporting_link_count = len(supporting_links)
    return {
        "query_id": sidecar.get("query_id"),
        "metrics": {
            "rfc": len(matched_fact_ids) / reference_count,
            "vrfc": len(matched_and_cited_fact_ids) / reference_count,
            "ucr": unsupported_count / claim_count,
            "cp": supporting_link_count / link_count if link_count else 0.0,
            "cr": len(validly_cited_claims) / claim_count,
            "rr": 1.0,
        },
        "counts": {
            "reference_facts": reference_count,
            "matched_reference_facts": len(matched_fact_ids),
            "matched_and_validly_cited_reference_facts": len(matched_and_cited_fact_ids),
            "claims": claim_count,
            "unsupported_claims": unsupported_count,
            "citation_links": link_count,
            "supporting_citation_links": supporting_link_count,
            "duplicate_citation_links": duplicate_links,
            "invalid_citation_links": len(invalid_links),
        },
        "runtime_error": False,
        "scorer_error": False,
    }


def score_query_safely(
    sidecar: Mapping[str, Any],
    output: Mapping[str, Any],
    claims: Sequence[ExtractedClaim],
    judge: EntailmentJudge,
) -> dict[str, Any]:
    """Apply the frozen scorer-failure rule without removing the query."""

    try:
        return score_query(sidecar, output, claims, judge)
    except Exception as exc:
        scored = _zero_score(sidecar, output)
        scored["scorer_error"] = True
        scored["scorer_error_type"] = type(exc).__name__
        return scored


def aggregate_query_scores(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot aggregate an empty scorer bundle")
    n_total = len(materialized)
    macro_keys = ("rfc", "vrfc", "cp", "cr", "rr")
    metrics: dict[str, float | None] = {
        key: sum(float(row["metrics"][key]) for row in materialized) / n_total
        for key in macro_keys
    }
    n_claims = sum(int(row["counts"]["claims"]) for row in materialized)
    unsupported = sum(int(row["counts"]["unsupported_claims"]) for row in materialized)
    metrics["ucr"] = unsupported / n_claims if n_claims else None
    metrics = {
        "rfc": metrics["rfc"],
        "vrfc": metrics["vrfc"],
        "ucr": metrics["ucr"],
        "cp": metrics["cp"],
        "cr": metrics["cr"],
        "rr": metrics["rr"],
    }
    return {
        "metrics": metrics,
        "n_total": n_total,
        "n_answered": sum(int(float(row["metrics"]["rr"]) == 1.0) for row in materialized),
        "n_runtime_error": sum(bool(row.get("runtime_error")) for row in materialized),
        "n_scorer_error": sum(bool(row.get("scorer_error")) for row in materialized),
        "n_claims": n_claims,
        "n_undefined_ucr": sum(row["metrics"]["ucr"] is None for row in materialized),
    }


def compute_appendix_diagnostics(
    *,
    reference_fact_ids: Sequence[str],
    retrieved_support_units: Iterable[tuple[str, str, str]],
    selected_support_units: Iterable[tuple[str, str, str]],
    selector_input_tokens: int,
    selected_tokens: int,
    presented_tokens: int,
) -> dict[str, float | None]:
    """Compute ER@10, exact-text SELR, and the two context reductions."""

    facts = set(reference_fact_ids)
    if not facts:
        raise ValueError("reference_fact_ids cannot be empty")
    if not (0 <= presented_tokens <= selected_tokens <= selector_input_tokens):
        raise ValueError("token counts must satisfy 0 <= presented <= selected <= input")
    if selector_input_tokens == 0:
        raise ValueError("selector_input_tokens must be positive")
    retrieved = set(retrieved_support_units)
    selected = set(selected_support_units)
    if any(unit[0] not in facts for unit in retrieved | selected):
        raise ValueError("support unit references an unknown fact")

    retrieved_facts = {fact_id for fact_id, _evidence_id, _text_hash in retrieved}
    retained_units = retrieved & selected
    return {
        "er_at_10": len(retrieved_facts) / len(facts),
        "selr": (len(retrieved) - len(retained_units)) / len(retrieved)
        if retrieved
        else None,
        "selection_reduction_rate": 1.0 - (selected_tokens / selector_input_tokens),
        "presented_reduction_rate": 1.0 - (presented_tokens / selector_input_tokens),
    }
