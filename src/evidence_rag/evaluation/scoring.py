import hashlib
import json
import re
from collections.abc import Iterable, Mapping

from evidence_rag.contracts.models import EvidenceCandidate
from evidence_rag.evaluation.models import (
    AggregateMetric,
    MetricDirection,
    MetricValue,
)

CORE_DIRECTIONS: dict[str, MetricDirection] = {
    "retriever.core.document_recall": "higher",
    "retriever.core.document_mrr": "higher",
    "retriever.core.document_recall_at_5": "higher",
    "retriever.core.document_recall_at_10": "higher",
    "retriever.core.document_recall_at_20": "higher",
    "selector.core.conditional_document_recall": "higher",
    "selector.core.document_precision": "higher",
    "generator.core.conditional_answer_match": "higher",
    "generator.core.conditional_cited_document_precision": "higher",
    "generator.core.citation_validity": "higher",
    "system.core.final_document_recall": "higher",
    "system.core.cited_document_precision": "higher",
    "system.core.answer_match": "higher",
}
CORE_METRIC_VERSION = "1.2"
CORE_METRIC_VERSIONS = {key: "1.0" for key in CORE_DIRECTIONS}

# The rank cut-offs reported as retriever.core.document_recall_at_{k}.
RECALL_AT_K = (5, 10, 20)


def signature(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def unscored(reason: str) -> MetricValue:
    return MetricValue(value=None, reason=reason)


def recall(predicted: set[str], relevant: set[str] | None) -> MetricValue:
    if relevant is None:
        return unscored("relevant documents were not labelled")
    if not relevant:
        return unscored("no relevant documents were labelled")
    return MetricValue(value=len(predicted & relevant) / len(relevant))


def _rank_ordered(candidates: Iterable[EvidenceCandidate]) -> list[EvidenceCandidate]:
    return sorted(candidates, key=lambda candidate: candidate.retrieval_rank)


def reciprocal_rank(
    candidates: Iterable[EvidenceCandidate],
    relevant: set[str] | None,
) -> MetricValue:
    """Mean-reciprocal-rank contribution: 1 / rank of the first relevant document."""

    if relevant is None:
        return unscored("relevant documents were not labelled")
    if not relevant:
        return unscored("no relevant documents were labelled")
    for candidate in _rank_ordered(candidates):
        if candidate.document_id in relevant:
            return MetricValue(value=1.0 / candidate.retrieval_rank)
    return MetricValue(value=0.0)


def recall_at_k(
    candidates: Iterable[EvidenceCandidate],
    relevant: set[str] | None,
    k: int,
) -> MetricValue:
    """Document recall restricted to the top-``k`` candidates by retrieval rank."""

    if relevant is None:
        return unscored("relevant documents were not labelled")
    if not relevant:
        return unscored("no relevant documents were labelled")
    top_k_documents = {candidate.document_id for candidate in _rank_ordered(candidates)[:k]}
    return MetricValue(value=len(top_k_documents & relevant) / len(relevant))


def precision(predicted: set[str], relevant: set[str] | None) -> MetricValue:
    if relevant is None:
        return unscored("relevant documents were not labelled")
    if not relevant:
        return unscored("no relevant documents were labelled")
    if not predicted:
        return MetricValue(value=0.0)
    return MetricValue(value=len(predicted & relevant) / len(predicted))


def conditional_recall(
    selected: set[str],
    retrieved: set[str],
    relevant: set[str] | None,
) -> MetricValue:
    if relevant is None:
        return unscored("relevant documents were not labelled")
    if not relevant:
        return unscored("no relevant documents were labelled")
    reachable = retrieved & relevant
    if not reachable:
        return unscored("retriever supplied no labelled-relevant document")
    return MetricValue(value=len(selected & reachable) / len(reachable))


def _normalise(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def answer_match(answer: str, references: tuple[str, ...] | None) -> MetricValue:
    if references is None:
        return unscored("reference answers were not labelled")
    if not references:
        return unscored("no reference answers were labelled")
    normalised_answer = _normalise(answer)
    matched = any(_normalise(reference) in normalised_answer for reference in references)
    return MetricValue(value=float(matched))


def generator_ineligibility(
    selected: set[str],
    relevant: set[str] | None,
) -> MetricValue | None:
    if relevant is None:
        return unscored("relevant documents were not labelled")
    if not relevant:
        return unscored("no relevant documents were labelled")
    if not selected & relevant:
        return unscored("selector supplied no labelled-relevant document")
    return None


def citation_validity(
    cited_evidence_ids: set[str],
    selected_evidence_ids: set[str],
) -> MetricValue:
    if not cited_evidence_ids:
        return MetricValue(value=0.0)
    return MetricValue(
        value=len(cited_evidence_ids & selected_evidence_ids)
        / len(cited_evidence_ids)
    )


def compose_generator_metrics(
    *,
    selected_evidence_ids: set[str],
    cited_evidence_ids: set[str],
    selected_document_ids: set[str],
    cited_document_ids: set[str],
    relevant_document_ids: set[str] | None,
    answer: str,
    reference_answers: tuple[str, ...] | None,
) -> tuple[MetricValue, MetricValue, MetricValue]:
    validity = citation_validity(cited_evidence_ids, selected_evidence_ids)
    ineligible = generator_ineligibility(
        selected_document_ids,
        relevant_document_ids,
    )
    if ineligible is not None:
        return ineligible, ineligible, validity
    return (
        answer_match(answer, reference_answers),
        precision(cited_document_ids, relevant_document_ids),
        validity,
    )


def document_ids(items: Iterable[EvidenceCandidate]) -> set[str]:
    return {item.document_id for item in items}


def aggregate_metrics(
    per_case: tuple[Mapping[str, MetricValue], ...],
    directions: Mapping[str, MetricDirection],
) -> dict[str, AggregateMetric]:
    aggregate: dict[str, AggregateMetric] = {}
    for key in directions:
        values = [
            metric.value
            for metrics in per_case
            if (metric := metrics[key]).value is not None
        ]
        aggregate[key] = AggregateMetric(
            mean=None if not values else sum(values) / len(values),
            n_scored=len(values),
            n_total=len(per_case),
        )
    return aggregate
