"""Faithful re-implementation of ALCE's citation precision / recall.

Ported from the reference implementation, `compute_autoais` in
princeton-nlp/ALCE `eval.py` (verified against the source, not reconstructed from
memory -- the precision definition involves a redundancy ablation that is easy to
get subtly wrong).

The reference algorithm, per example:

    entail = 0; entail_prec = 0; total_citations = 0
    for each sentence:
        ref = citations of this sentence
        if len(ref) == 0:                  joint_entail = 0
        elif any ref is out of range:      joint_entail = 0
        else:
            total_citations += len(ref)
            joint_entail = NLI(concat(cited docs), sentence)
        entail += joint_entail
        if joint_entail and len(ref) > 1:
            for each cited doc d:
                if NLI(d, sentence):                       entail_prec += 1
                elif NLI(concat(ref minus d), sentence):   pass   # overcite, +0
                else:                                      entail_prec += 1
        else:
            entail_prec += joint_entail
    ais_scores.append(entail / len(sents))
    ais_scores_prec.append(entail_prec / total_citations if total_citations else 0)

    citation_rec  = 100 * mean(ais_scores)
    citation_prec = 100 * mean(ais_scores_prec)

Load-bearing details that differ from a naive implementation:

* **Recall's denominator is every sentence**, including uncited ones, which score
  0. Adding unsupported prose therefore lowers recall.
* **Precision's denominator is the example's total citations**, and an example
  that produced no citations at all scores 0 rather than being skipped.
* **The redundancy ablation only runs when a sentence has more than one
  citation.** A lone citation is precise iff it entails the sentence.
* **Both metrics are macro-averaged over examples**, not pooled over sentences or
  citations.

Deviations from ALCE, both deliberate and both documented in the results:

* The entailment function is **MiniCheck**, not ALCE's TRUE/AutoAIS. TRUE selected
  the verified arms' citations, so using it as the judge here would be circular.
* Documents are passed as raw chunk text; ALCE's `_format_document` prepends a
  title, and our `EvidenceCandidate` has no title field.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScoredExample:
    """One answer, already split into sentences with citations attached.

    ``sentences`` must have citation markers removed (ALCE's ``remove_citations``)
    because the judge should see the prose, not the bracket markers.
    ``citations[i]`` lists the document ids cited by ``sentences[i]``.
    """

    example_id: str
    sentences: tuple[str, ...]
    citations: tuple[tuple[str, ...], ...]
    docs: Mapping[str, str]

    def __post_init__(self) -> None:
        if len(self.sentences) != len(self.citations):
            raise ValueError("each sentence needs exactly one citation list")


@dataclass
class CitationReport:
    citation_rec: float
    citation_prec: float
    n_examples: int
    n_sentences: int
    n_citations: int
    sent_mcite: int = 0
    """sentences carrying more than one citation"""
    sent_mcite_support: int = 0
    """of those, ones whose joint citation set entails the sentence"""
    sent_mcite_overcite: int = 0
    """of those, individual citations found redundant (ALCE's 'overcite')"""
    per_example: list[dict[str, Any]] = field(default_factory=list)


class CachedEntailment:
    """Memoises the judge: the ablation re-asks about the same (passage, sentence)
    pairs repeatedly, and MiniCheck calls are the cost of this whole pass."""

    def __init__(self, entails: Callable[[str, str], bool]) -> None:
        self._entails = entails
        self._cache: dict[tuple[str, str], bool] = {}
        self.calls = 0

    def __call__(self, premise: str, hypothesis: str) -> bool:
        key = (premise, hypothesis)
        hit = self._cache.get(key)
        if hit is None:
            self.calls += 1
            hit = bool(self._entails(premise, hypothesis))
            self._cache[key] = hit
        return hit


def compute_citation_metrics(
    examples: Sequence[ScoredExample],
    entails: Callable[[str, str], bool],
) -> CitationReport:
    """ALCE citation recall + precision. ``entails(premise, hypothesis) -> bool``."""
    judge = CachedEntailment(entails)
    recall_scores: list[float] = []
    precision_scores: list[float] = []
    report = CitationReport(
        citation_rec=0.0, citation_prec=0.0, n_examples=0, n_sentences=0, n_citations=0
    )

    for example in examples:
        if not example.sentences:
            continue  # ALCE: `if len(sents) == 0: continue`
        entail = 0
        entail_prec = 0
        total_citations = 0

        for sentence, refs in zip(example.sentences, example.citations, strict=True):
            valid = all(ref in example.docs for ref in refs)
            if not refs or not valid:
                joint_entail = 0
            else:
                total_citations += len(refs)
                joint_passage = "\n".join(example.docs[ref] for ref in refs)
                joint_entail = int(judge(joint_passage, sentence))

            entail += joint_entail
            if len(refs) > 1:
                report.sent_mcite += 1

            if joint_entail and len(refs) > 1:
                report.sent_mcite_support += 1
                for ref in refs:
                    if judge(example.docs[ref], sentence):
                        entail_prec += 1
                        continue
                    rest = [other for other in refs if other != ref]
                    if judge("\n".join(example.docs[other] for other in rest), sentence):
                        report.sent_mcite_overcite += 1  # redundant: contributes 0
                    else:
                        entail_prec += 1
            else:
                entail_prec += joint_entail

        example_recall = entail / len(example.sentences)
        example_precision = entail_prec / total_citations if total_citations > 0 else 0.0
        recall_scores.append(example_recall)
        precision_scores.append(example_precision)
        report.n_examples += 1
        report.n_sentences += len(example.sentences)
        report.n_citations += total_citations
        report.per_example.append(
            {
                "example_id": example.example_id,
                "citation_rec": example_recall,
                "citation_prec": example_precision,
                "sentences": len(example.sentences),
                "citations": total_citations,
            }
        )

    report.citation_rec = 100 * (sum(recall_scores) / len(recall_scores)) if recall_scores else 0.0
    report.citation_prec = (
        100 * (sum(precision_scores) / len(precision_scores)) if precision_scores else 0.0
    )
    return report
