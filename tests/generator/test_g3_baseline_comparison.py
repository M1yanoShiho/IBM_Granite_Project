"""CI-safe tests for the G3 runner's pure logic (no models loaded)."""

import random
import sys
from pathlib import Path

from evidence_rag.contracts.models import GenerationResult

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import g3_baseline_comparison as g3  # noqa: E402


class FakeJudge:
    """Supports the hypothesis iff the premise contains the trigger token."""

    def __init__(self, trigger: str = "Seattle") -> None:
        self.trigger = trigger

    def classify(self, premise: str, hypothesis: str) -> str:
        return "entailment" if self.trigger in premise else "neutral"


def _asqa(sample_id: str, question: str, answer: str, facts: list[str], gold: list[list[str]]) -> dict:
    return {
        "sample_id": sample_id,
        "question": question,
        "answer": answer,
        "annotations": [{"knowledge": [{"content": f} for f in facts]}],
        "qa_pairs": [{"question": "q", "short_answers": g} for g in gold],
        "docs": [{"text": f"passage {i} {answer}"} for i in range(6)],
    }


def test_str_em_full_partial_empty() -> None:
    gold = (("Seattle",), ("1997",))
    assert g3.str_em("Formed in Seattle, debuted 1997.", gold) == 1.0
    assert g3.str_em("Formed in Seattle.", gold) == 0.5
    assert g3.str_em("", gold) == 0.0  # abstention is not correct
    assert g3.str_em("anything", ()) is None


def test_build_cases_extracts_gold_and_requires_it() -> None:
    good = _asqa("s1", "when 1997?", "The band formed in Seattle. It debuted in 1997.",
                 ["The band formed in Seattle.", "It debuted in 1997."], [["Seattle"], ["1997"]])
    cases = g3.build_cases([good], limit=5, rng=random.Random(13), top_k=5)
    assert len(cases) == 1
    assert cases[0].gold_answers == (("Seattle",), ("1997",))
    assert len(cases[0].selected.evidence) == 5
    assert cases[0].constraints == ("as of 1997",)

    # no qa_pair short answers -> dropped (correctness axis would be undefined)
    no_gold = _asqa("s2", "q", "Ada founded it in Paris.", ["Ada founded it in Paris."], [])
    assert g3.build_cases([no_gold], limit=5, rng=random.Random(13), top_k=5) == []


def test_score_arm_precision_and_abstention() -> None:
    case = g3.build_cases(
        [_asqa("s1", "q", "The band formed in Seattle. It debuted in 1997.",
               ["The band formed in Seattle.", "It debuted in 1997."], [["Seattle"], ["1997"]])],
        limit=1, rng=random.Random(1), top_k=5,
    )[0]
    cases_by_id = {case.query_id: case}
    judge = FakeJudge("Seattle")

    # answered: one cited chunk supports (contains 'Seattle'), one does not
    d0, d1 = case.selected.evidence[0].evidence_id, case.selected.evidence[1].evidence_id
    answered = {case.query_id: GenerationResult(query_id=case.query_id, answer="Formed in Seattle.",
                                                cited_evidence_ids=(d0, d1))}
    rep = g3.score_arm(answered, cases_by_id, judge)
    m = rep["per_case"][0]["metrics"]
    assert m["coverage"]["value"] == 1.0
    assert m["citation_precision"]["value"] == 1.0  # both chunks contain the answer text incl 'Seattle'

    # abstention: citation_* is None (dropped from paired both-answered subset), coverage 0
    absent = {case.query_id: GenerationResult(query_id=case.query_id, answer="", cited_evidence_ids=())}
    rep2 = g3.score_arm(absent, cases_by_id, judge)
    m2 = rep2["per_case"][0]["metrics"]
    assert m2["coverage"]["value"] == 0.0
    assert m2["citation_precision"] is None
    assert m2["answer_correctness"]["value"] == 0.0


def test_score_arm_precision_below_one_when_a_citation_is_unsupported() -> None:
    case = g3.build_cases(
        [_asqa("s1", "q", "The band formed in Seattle. It debuted in 1997.",
               ["The band formed in Seattle.", "It debuted in 1997."], [["Seattle"]])],
        limit=1, rng=random.Random(1), top_k=5,
    )[0]
    # judge supports only chunks containing 'passage 0'; cite chunk0 (supported) + chunk3 (not)
    judge = FakeJudge("passage 0 ")
    d0, d3 = case.selected.evidence[0].evidence_id, case.selected.evidence[3].evidence_id
    res = {case.query_id: GenerationResult(query_id=case.query_id, answer="x Seattle x",
                                           cited_evidence_ids=(d0, d3))}
    rep = g3.score_arm(res, {case.query_id: case}, judge)
    assert rep["per_case"][0]["metrics"]["citation_precision"]["value"] == 0.5


def test_build_arms_constructs_all_three_without_loading() -> None:
    class FakeLLM:
        def generate(self, prompt: str) -> str:
            return "x"

    arms = g3.build_arms(FakeLLM(), FakeJudge())
    assert set(arms) == {"baseline", "verify-only", "verified-full"}
