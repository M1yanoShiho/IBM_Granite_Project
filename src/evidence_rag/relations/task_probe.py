"""Gate 0B-2: task-shaped relation probe from the counterfactual mutation log (design §3.2).

Four deterministic pair types per injected query, zero new human annotation:

    needle      x gold claim         -> SUPPORTS   (injector verified the alias occurs once)
    cf::needle  x replacement claim  -> SUPPORTS   (by definition of the mutation)
    cf::needle  x gold claim         -> REFUTES    (single-answer assumption + same-class swap)
    needle      x replacement claim  -> REFUTES    (same)

The third row is the twin discrimination that three rounds of extraction-layer work failed to
fix; here it becomes a plain CPU-scorable classification problem.

DISCIPLINE (design §3.3): this is an ISOLATED-PAIR probe. Results must never be extrapolated to
the retrieval pool — pool-level judgement belongs to cluster_eval. A previous round of work
failed in exactly that way, with an isolated probe showing a large gain that reversed once the
same configuration met a real 20-passage pool.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evidence_rag.materializer.provenance import MutationRecord
from evidence_rag.relations.claims import build_hypothesis
from evidence_rag.relations.models import RelationLabel

NEEDLE_GOLD = "needle_gold"
CF_REPLACEMENT = "cf_replacement"
CF_GOLD = "cf_gold"
NEEDLE_REPLACEMENT = "needle_replacement"

TWIN_REFUTES = (CF_GOLD, NEEDLE_REPLACEMENT)
GOLD_SUPPORTS = (NEEDLE_GOLD,)


@dataclass(frozen=True)
class ProbePair:
    premise: str
    hypothesis: str
    label: RelationLabel
    group: str
    kind: str
    query_id: str


def synthetic_family(record: MutationRecord) -> str:
    """Leakage-audit group. Two queries sharing a gold/replacement/class triple are the same
    synthetic family and must not be split across a train/test boundary."""
    return f"{record.gold_value}|{record.replacement_value}|{record.string_class}"


def build_probe_pairs(
    *,
    records: Sequence[MutationRecord],
    question_by_query: Mapping[str, str],
    text_by_document: Mapping[str, str],
) -> tuple[ProbePair, ...]:
    """Records whose query or either document is missing are skipped, not partially emitted —
    a half-built probe would silently change the denominators Gate 0B is judged on."""
    pairs: list[ProbePair] = []
    for record in records:
        question = question_by_query.get(record.query_id)
        needle_text = text_by_document.get(record.needle_document_id)
        cf_text = text_by_document.get(record.counterfactual_document_id)
        if question is None or needle_text is None or cf_text is None:
            continue
        gold_claim = build_hypothesis(question, record.gold_value)
        replacement_claim = build_hypothesis(question, record.replacement_value)
        family = synthetic_family(record)
        for premise, hypothesis, label, kind in (
            (needle_text, gold_claim, RelationLabel.SUPPORTS, NEEDLE_GOLD),
            (cf_text, replacement_claim, RelationLabel.SUPPORTS, CF_REPLACEMENT),
            (cf_text, gold_claim, RelationLabel.REFUTES, CF_GOLD),
            (needle_text, replacement_claim, RelationLabel.REFUTES, NEEDLE_REPLACEMENT),
        ):
            pairs.append(
                ProbePair(
                    premise=premise,
                    hypothesis=hypothesis,
                    label=label,
                    group=family,
                    kind=kind,
                    query_id=record.query_id,
                )
            )
    return tuple(pairs)
