"""CI-safe tests for the audit export/score scripts (no models, no I/O deps).

Cover the pieces that must not be wrong: the stratified sample, packet blindness
(no verdict leaks), and the per-stratum scoring.
"""

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_export as ae  # noqa: E402
import audit_score as asc  # noqa: E402


def _record(query: str, claim_id: str, stratum: str, veto_pairs: int = 0) -> dict:
    status = "supported" if stratum == "supported" else "unsupported"
    supporting = [f"{query}::d0"] if stratum == "supported" else []
    return {
        "query_id": query,
        "question": f"Q for {query}?",
        "claim_id": claim_id,
        "claim_text": f"claim {claim_id} of {query}",
        "system_status": status,
        "stratum": stratum,
        "supporting_evidence_ids": supporting,
        "entity_veto_pairs": veto_pairs,
        "evidence": [
            {"evidence_id": f"{query}::d{i}", "text": f"evidence {i} for {query}"} for i in range(5)
        ],
    }


def _dataset() -> list[dict]:
    records = []
    for i in range(4):
        records.append(_record(f"qa{i}", f"claim-{i}", "A"))
    for i in range(20):
        records.append(_record(f"qb{i}", f"claim-{i}", "B", veto_pairs=2))
    for i in range(15):
        records.append(_record(f"qs{i}", f"claim-{i}", "supported"))
    return records


def test_sample_takes_all_of_smaller_stratum() -> None:
    records = _dataset()  # A=4, B=20, supported=15
    chosen, sizes = ae.sample_items(records, random.Random(13), dropped_target=10, controls_target=10)

    assert sizes["A_no_entailment"] == 4
    assert sizes["B_entity_vetoed"] == 20
    assert sizes["entity_veto_pairs_total"] == 40  # 20 B-claims x 2 pairs each
    chosen_a = [r for r in chosen if r["stratum"] == "A"]
    chosen_b = [r for r in chosen if r["stratum"] == "B"]
    chosen_sup = [r for r in chosen if r["stratum"] == "supported"]
    assert len(chosen_a) == 4  # all of the smaller stratum kept
    assert len(chosen_b) == 6  # filled up to the dropped target of 10
    assert len(chosen_sup) == 10


def test_sample_balances_when_both_strata_large() -> None:
    # both strata comfortably exceed half the target -> ~even split, not all-of-one
    records = []
    for i in range(27):
        records.append(_record(f"qa{i}", f"c{i}", "A"))
    for i in range(24):
        records.append(_record(f"qb{i}", f"c{i}", "B"))
    for i in range(26):
        records.append(_record(f"qs{i}", f"c{i}", "supported"))
    chosen, _ = ae.sample_items(records, random.Random(13), dropped_target=30, controls_target=10)
    a = sum(1 for r in chosen if r["stratum"] == "A")
    b = sum(1 for r in chosen if r["stratum"] == "B")
    assert a == 15 and b == 15  # balanced 15/15, not 24 B + 6 A


def test_sample_takes_everything_when_under_target() -> None:
    records = [_record("qa", "c1", "A"), _record("qb", "c2", "B"), _record("qs", "c3", "supported")]
    chosen, _ = ae.sample_items(records, random.Random(1), dropped_target=30, controls_target=10)
    assert len(chosen) == 3


def test_sample_is_deterministic_with_seed() -> None:
    records = _dataset()
    a, _ = ae.sample_items(records, random.Random(7), 10, 5)
    b, _ = ae.sample_items(records, random.Random(7), 10, 5)
    assert [r["claim_id"] + r["query_id"] for r in a] == [r["claim_id"] + r["query_id"] for r in b]


def test_packet_is_blind() -> None:
    records = _dataset()
    rng = random.Random(13)
    chosen, sizes = ae.sample_items(records, rng, 10, 5)
    items = ae.to_items(chosen, rng)
    packet = ae.render_packet(items, seed=13)

    # no verdict/stratum machinery may leak into what the adjudicator reads
    assert "system_status" not in packet
    assert "stratum" not in packet
    assert "(dropped)" not in packet
    assert "supporting_evidence_ids" not in packet
    # supporting evidence ids (which would reveal the answer) must not appear
    for item in items:
        for evidence_id in item.supporting_evidence_ids:
            assert evidence_id not in packet
    # but the key retains them
    key = ae.build_key(items, 13, sizes)
    assert all("stratum" in entry for entry in key["items"].values())


def test_packet_shows_all_evidence_per_item() -> None:
    records = _dataset()
    rng = random.Random(5)
    chosen, sizes = ae.sample_items(records, rng, 10, 5)
    items = ae.to_items(chosen, rng)
    packet = ae.render_packet(items, seed=5)
    # every item's five chunks are present
    for item in items:
        assert len(item.evidence) == 5
        for chunk in item.evidence:
            assert chunk["text"] in packet


def test_parse_packet_reads_adjudication_and_reason(tmp_path: Path) -> None:
    filled = (
        "### item-01\n"
        "**Question:** q\n**Claim:** c\n**Evidence:**\n1. e\n"
        "adjudication: not supported\nreason: evidence is about a different year\n"
        "---\n"
        "### item-02\n"
        "adjudication: Supported\nreason: stated verbatim\n"
    )
    packet = tmp_path / "packet.md"
    packet.write_text(filled, encoding="utf-8")

    parsed = asc.parse_packet(packet)

    assert parsed["item-01"]["adjudication"] == "not_supported"
    assert parsed["item-01"]["reason"] == "evidence is about a different year"
    assert parsed["item-02"]["adjudication"] == "supported"  # case-insensitive


def test_score_stratum_false_veto_and_agreement() -> None:
    key_items = {
        "item-01": {"system_status": "unsupported", "stratum": "A"},
        "item-02": {"system_status": "unsupported", "stratum": "A"},
        "item-03": {"system_status": "unsupported", "stratum": "A"},
    }
    verdicts = {
        "item-01": {"adjudication": "not_supported"},  # agrees with system
        "item-02": {"adjudication": "supported"},  # false veto
        "item-03": {"adjudication": "unclear"},  # counted apart
    }
    result = asc.score_stratum(["item-01", "item-02", "item-03"], key_items, verdicts)
    assert result["n"] == 3
    assert result["adjudicated"] == 3
    assert result["false_veto"].startswith("1/3")
    assert result["unclear"].startswith("1/3")
    # agreement is over the non-unclear adjudicated items (2): item-01 agrees
    assert result["agreement"].startswith("1/2")


def test_score_controls_precision() -> None:
    verdicts = {
        "item-10": {"adjudication": "supported"},
        "item-11": {"adjudication": "not_supported"},
        "item-12": {"adjudication": "unclear"},
    }
    result = asc.score_controls(["item-10", "item-11", "item-12"], verdicts)
    # of the 2 non-unclear controls, 1 human-supported
    assert result["in_chain_citation_precision"].startswith("1/2")
    assert result["unclear"].startswith("1/3")
