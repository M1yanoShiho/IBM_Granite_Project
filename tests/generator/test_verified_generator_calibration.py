"""CI-safe tests for the G2 calibration runner's pure logic (no models loaded).

The runner itself lives in scripts/ and is out of CI's default path; these tests
cover the two pieces most likely to be wrong -- the per-case B5 bookkeeping and
the ASQA case construction -- with hand-built objects and fakes.
"""

import random
import sys
from pathlib import Path

from evidence_rag.contracts.models import GenerationResult
from evidence_rag.generator.completeness import fallback_gap_question
from evidence_rag.generator.models import (
    Claim,
    ClaimSpan,
    ClaimVerification,
    DraftAnswer,
    RequiredFactCoverage,
    VerificationReport,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import verified_generator_calibration as g  # noqa: E402


def _claim(claim_id: str, start: int, end: int, faithful: bool = True) -> Claim:
    return Claim(
        claim_id=claim_id,
        text="x",
        span=ClaimSpan(start=start, end=end),
        faithful_to_answer=faithful,
    )


def _draft(query_id: str, text: str, claims: tuple[Claim, ...]) -> DraftAnswer:
    return DraftAnswer(query_id=query_id, answer_text=text, claims=claims)


def _verification(claim_id: str, status: str, evidence: tuple[str, ...], *, contradicted: bool = False,
                  entity_consistent: bool = True) -> ClaimVerification:
    return ClaimVerification(
        claim_id=claim_id,
        status=status,
        supporting_evidence_ids=evidence,
        entity_consistent=entity_consistent,
        contradicted=contradicted,
    )


def test_summarize_case_counts_supported_and_repair() -> None:
    # draft has 3 claims; one is unfaithful so only 2 are verified
    draft = _draft("q1", "aaaa bbbb", (_claim("c1", 0, 4), _claim("c2", 5, 9), _claim("c3", 5, 9, faithful=False)))
    report = VerificationReport(
        query_id="q1",
        claims=(
            _verification("c1", "supported", ("e1",)),
            _verification("c2", "unsupported", ()),
        ),
        fact_coverage=(RequiredFactCoverage(required_fact="f1", covered=True),),
    )
    generation = GenerationResult(query_id="q1", answer="aaaa.", cited_evidence_ids=("e1",))

    stats = g.summarize_case(draft, report, generation, entity_mismatch=0, gaps_patched=0)

    assert stats.draft_claims == 3
    assert stats.faithful_claims == 2
    assert stats.supported == 1
    assert stats.unsupported == 1
    assert stats.repair_fired is True  # the unsupported claim was dropped
    assert stats.abstained is False
    assert stats.contract_ok is True
    assert stats.gaps_found == 0


def test_summarize_case_abstention_and_contradiction() -> None:
    draft = _draft("q2", "aaaa", (_claim("c1", 0, 4),))
    report = VerificationReport(
        query_id="q2",
        claims=(_verification("c1", "unsupported", (), contradicted=True),),
        fact_coverage=(RequiredFactCoverage(required_fact="f1", covered=False, gap_question="what?"),),
    )
    generation = GenerationResult(query_id="q2", answer="", cited_evidence_ids=())

    stats = g.summarize_case(draft, report, generation, entity_mismatch=2, gaps_patched=0)

    assert stats.abstained is True
    assert stats.contract_ok is True  # empty answer + no citations is valid
    assert stats.contradicted == 1
    assert stats.entity_mismatch == 2
    assert stats.gaps_found == 1
    assert stats.repair_fired is True


def test_summarize_case_clean_answer_no_repair() -> None:
    draft = _draft("q3", "aaaa bbbb", (_claim("c1", 0, 4), _claim("c2", 5, 9)))
    report = VerificationReport(
        query_id="q3",
        claims=(
            _verification("c1", "supported", ("e1",)),
            _verification("c2", "supported", ("e2",)),
        ),
        fact_coverage=(RequiredFactCoverage(required_fact="f1", covered=True),),
    )
    generation = GenerationResult(query_id="q3", answer="aaaa bbbb.", cited_evidence_ids=("e1", "e2"))

    stats = g.summarize_case(draft, report, generation, entity_mismatch=0, gaps_patched=0)

    assert stats.repair_fired is False
    assert stats.supported == 2
    assert stats.unsupported == 0


def _asqa_sample(sample_id: str, question: str, answer: str, facts: list[str]) -> dict:
    return {
        "sample_id": sample_id,
        "question": question,
        "answer": answer,
        "annotations": [{"knowledge": [{"content": f} for f in facts]}],
        "docs": [{"text": f"passage {i} {answer}"} for i in range(6)],
    }


def test_build_cases_extracts_facts_year_and_selected() -> None:
    data = [
        _asqa_sample("s1", "When did it debut in 1997?",
                     "The band formed in Seattle. It released its debut in 1997.",
                     ["The band formed in Seattle.", "It released its debut in 1997."]),
    ]
    cases = g.build_cases(data, limit=5, rng=random.Random(13), top_k=5)

    assert len(cases) == 1
    case = cases[0]
    assert case.year == "1997"
    assert case.constraints == ("as of 1997",)
    assert len(case.selected.evidence) == 5
    assert case.required_facts  # facts stated by the gold answer survived


def test_build_cases_drops_facts_not_in_answer() -> None:
    # the knowledge sentence is absent from the gold answer -> not a fair 'covered'
    data = [_asqa_sample("s1", "q", "A short unrelated answer.", ["Some entirely different unstated fact about physics."])]
    cases = g.build_cases(data, limit=5, rng=random.Random(13), top_k=5)
    assert cases == []


def test_generic_gap_detection() -> None:
    fact = "It released its debut in 1997"
    assert g._is_generic_gap(fallback_gap_question(fact), fact) is True
    assert g._is_generic_gap("What entirely unrelated thing?", fact) is True  # no shared content token
    assert g._is_generic_gap("When was the debut released?", fact) is False  # shares 'debut'/'released'


def test_run_b4_labels_and_constraint_survival() -> None:
    class FakeChecker:
        """Stand-in CompletenessChecker: marks own facts covered, foreign uncovered
        with a gap question that echoes the constraint."""

        def check(self, answer_text, checklist):
            results = []
            for fact in checklist.required_facts:
                covered = fact.lower() in answer_text.lower()
                gap = None if covered else f"What about {fact}? {' '.join(checklist.constraints)}"
                results.append(RequiredFactCoverage(required_fact=fact, covered=covered, gap_question=gap))
            return tuple(results)

    data = [
        _asqa_sample("s1", "q1 in 1997", "The band formed in Seattle. It released its debut in 1997.",
                     ["The band formed in Seattle.", "It released its debut in 1997."]),
        _asqa_sample("s2", "q2", "Ada founded the company in Paris.", ["Ada founded the company in Paris."]),
    ]
    cases = g.build_cases(data, limit=5, rng=random.Random(1), top_k=5)
    result = g.run_b4(cases, FakeChecker(), random.Random(1))

    assert result.own_facts == result.own_covered  # every own fact is literally in its answer
    assert result.foreign_uncovered == result.foreign_facts  # every foreign fact absent
    # at least one constrained case carried its year into the gap question
    assert result.constrained_gaps >= 1
    assert result.constrained_gaps_with_year == result.constrained_gaps
