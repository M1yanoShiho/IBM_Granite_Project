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
from typing import Literal

from evidence_rag.materializer.provenance import MutationRecord
from evidence_rag.relations.claims import HypothesisForm, build_hypothesis
from evidence_rag.relations.models import RelationLabel

NEEDLE_GOLD = "needle_gold"
CF_REPLACEMENT = "cf_replacement"
CF_GOLD = "cf_gold"
NEEDLE_REPLACEMENT = "needle_replacement"

# R012d. `gold_value` is `canonicalize_answer` output (lowercased); `gold_alias_used` is the
# raw string the injector verified in the needle document. Measured 2026-08-03: gold claims
# are 100% lowercase while replacement claims are 67.8% cased, so the two gate metrics sit on
# different casing distributions and a cased model eats the difference. "canonical" is the
# default because R012 and R012b all ran on it and must stay reproducible.
GoldAnswerSource = Literal["canonical", "surface"]

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
    hypothesis_form: HypothesisForm = build_hypothesis,
    gold_answer_source: GoldAnswerSource = "canonical",
) -> tuple[ProbePair, ...]:
    """Records whose query or either document is missing are skipped, not partially emitted —
    a half-built probe would silently change the denominators Gate 0B is judged on.

    `hypothesis_form` defaults to the frozen §2.4 template so that omitting it cannot silently
    move the pre-registered main arm; the other rungs exist for the R012b form ablation."""
    pairs: list[ProbePair] = []
    for record in records:
        question = question_by_query.get(record.query_id)
        needle_text = text_by_document.get(record.needle_document_id)
        cf_text = text_by_document.get(record.counterfactual_document_id)
        if question is None or needle_text is None or cf_text is None:
            continue
        gold_answer = (
            record.gold_value if gold_answer_source == "canonical" else record.gold_alias_used
        )
        gold_claim = hypothesis_form(question, gold_answer)
        replacement_claim = hypothesis_form(question, record.replacement_value)
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
