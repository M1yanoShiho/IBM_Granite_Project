"""CI-safe tests for the G3 follow-up diagnosis analysis (no models loaded)."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import g3_diagnosis as d  # noqa: E402


def _record(**overrides):
    base = {
        "query_id": "q1",
        "question": "q?",
        "error": "",
        "draft_answer": "a draft",
        "draft_claims": [{"claim_id": "claim-1", "text": "c1", "faithful": True}],
        "verified_claims": [],
        "pairs": [],
        "required_facts": [],
        "gaps": [],
        "repaired_answer": "",
        "draft_origin_citations": [],
        "recheck_added_citations": [],
        "final_answer": "",
        "final_citations": [],
        "answered": False,
        "abstention_reason": "",
        "evidence": [],
        "judged_citations": [],
    }
    base.update(overrides)
    return base


def test_false_gap_and_citation_origin_counts() -> None:
    records = [
        _record(
            answered=True,
            gaps=[{"required_fact": "f1", "recheck_found": True, "false_gap": True},
                  {"required_fact": "f2", "recheck_found": True, "false_gap": False}],
            judged_citations=[
                {"evidence_id": "e1", "origin": "draft", "minicheck_supported": True},
                {"evidence_id": "e2", "origin": "recheck", "minicheck_supported": False},
                {"evidence_id": "e3", "origin": "recheck", "minicheck_supported": False},
            ],
        )
    ]
    a = d.analyse(records)
    assert a.gaps_total == 2 and a.gaps_false == 1
    assert a.cites_draft == 1 and a.cites_draft_supported == 1
    assert a.cites_recheck == 2 and a.cites_recheck_supported == 0


def test_abstention_attributed_to_false_gaps() -> None:
    records = [
        # abstained, both triggering gaps false -> fully attributable
        _record(answered=False, abstention_reason="missing_required_fact",
                gaps=[{"required_fact": "f1", "recheck_found": False, "false_gap": True},
                      {"required_fact": "f2", "recheck_found": False, "false_gap": True}]),
        # abstained, one triggering gap genuine -> counted in 'any' but not 'all'
        _record(answered=False, abstention_reason="missing_required_fact",
                gaps=[{"required_fact": "f3", "recheck_found": False, "false_gap": True},
                      {"required_fact": "f4", "recheck_found": False, "false_gap": False}]),
        # abstained for a different reason -> not in the completeness-path bucket
        _record(answered=False, abstention_reason="no_trusted_content"),
    ]
    a = d.analyse(records)
    assert a.abstentions == 3
    assert a.abstentions_missing_fact == 2
    assert a.abstentions_all_false_gaps == 1
    assert a.abstentions_any_false_gap == 2


def test_sweep_is_monotone_in_coverage_and_uses_entity_check() -> None:
    records = [
        _record(
            draft_claims=[{"claim_id": "claim-1", "text": "c1", "faithful": True}],
            pairs=[
                # entailed only at low thresholds, judged unsupported by MiniCheck
                {"evidence_id": "e1", "claim_text": "c1", "p_entail": 0.3,
                 "entity_consistent": True, "minicheck_supported": False},
                # strong pair, supported
                {"evidence_id": "e2", "claim_text": "c1", "p_entail": 0.95,
                 "entity_consistent": True, "minicheck_supported": True},
                # strong but entity-vetoed -> never cited at any threshold
                {"evidence_id": "e3", "claim_text": "c1", "p_entail": 0.99,
                 "entity_consistent": False, "minicheck_supported": True},
            ],
        )
    ]
    rows = {row["threshold"]: row for row in d.sweep_verify_only(records)}
    # at 0.20 both e1 and e2 qualify -> 2 cited pairs, precision 1/2
    assert rows[0.20]["cited_pairs"] == 2
    assert rows[0.20]["citation_precision"] == 0.5
    # at 0.50 only e2 qualifies -> precision 1.0
    assert rows[0.50]["cited_pairs"] == 1
    assert rows[0.50]["citation_precision"] == 1.0
    # the entity-vetoed pair is excluded everywhere
    assert rows[0.90]["cited_pairs"] == 1
    # coverage never increases as the threshold rises
    coverages = [rows[t]["coverage"] for t in sorted(rows)]
    assert coverages == sorted(coverages, reverse=True)


def test_sweep_ignores_unfaithful_claims_and_errored_cases() -> None:
    records = [
        _record(error="boom", pairs=[{"evidence_id": "e1", "claim_text": "c1", "p_entail": 0.9,
                                      "entity_consistent": True, "minicheck_supported": True}]),
        _record(
            draft_claims=[{"claim_id": "claim-1", "text": "c1", "faithful": False}],
            pairs=[{"evidence_id": "e1", "claim_text": "c1", "p_entail": 0.9,
                    "entity_consistent": True, "minicheck_supported": True}],
        ),
    ]
    rows = {row["threshold"]: row for row in d.sweep_verify_only(records)}
    # the errored case is skipped entirely; the unfaithful claim yields no citations
    assert rows[0.50]["cited_pairs"] == 0
    assert rows[0.50]["coverage"] == 0.0
