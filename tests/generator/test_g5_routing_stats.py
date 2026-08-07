"""Pin the G7 runner's reporting tail.

It exists as a test because the previous round's scoring job died in an
equivalent tail on a stale arm name -- *after* every number had been computed.
The arm outputs survive that; the routing report does not, and the routing report
is where the observe-only measurement lives.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from g5_verify_annotate import (  # noqa: E402
    ANNOTATE_ARMS,
    CONTROL_ARM,
    SUBJECT_ARM,
    build_routing_stats,
)

from evidence_rag.contracts.models import UNVERIFIED_ANNOTATION  # noqa: E402
from evidence_rag.generator.verify_annotate import RoutingStats  # noqa: E402


def _record(answer: str, outcomes: tuple[str, ...] = ()) -> dict:
    return {"answer": answer, "routing": [{"outcome": o} for o in outcomes]}


def _fixture() -> tuple[dict, dict, dict]:
    stats_by_arm = {arm: RoutingStats() for arm in ANNOTATE_ARMS}
    subject = stats_by_arm[SUBJECT_ARM]
    subject.claims, subject.verified, subject.unverified = 10, 7, 3
    subject.gate_would_drop = 4
    subject.gate_would_drop_now_cited = 3
    subject.gate_would_drop_now_annotated = 1
    stats_by_arm[CONTROL_ARM].dropped_entity_conflict = 4
    stats_by_arm[CONTROL_ARM].gate_would_drop = 4
    results = {
        arm: {
            "q1": _record(f"Alpha holds. Beta holds. {UNVERIFIED_ANNOTATION}", ("unverified",)),
            "q2": _record(""),
        }
        for arm in ANNOTATE_ARMS
    }
    return stats_by_arm, results, dict.fromkeys(ANNOTATE_ARMS, 0)


def test_the_tail_assembles_and_serialises() -> None:
    report = build_routing_stats(*_fixture())
    json.dumps(report)  # the failure mode being guarded against is a raise, not a value
    assert report["arm"] == SUBJECT_ARM


def test_every_annotate_arm_reports_its_annotation_reach() -> None:
    """Named per arm, so adding or renaming an arm cannot silently drop one -- the
    G6 tail crashed on exactly that."""
    report = build_routing_stats(*_fixture())
    for arm in ANNOTATE_ARMS:
        key = f"annotated_claims_reaching_an_answer_{arm.rsplit('-', 1)[1]}"
        assert report[key] == 1, key


def test_the_observe_only_counters_are_reported() -> None:
    report = build_routing_stats(*_fixture())
    assert report["gate_would_drop"] == 4
    assert report["gate_would_drop_now_cited"] == 3
    assert report["gate_would_drop_now_annotated"] == 1


def test_the_control_arm_self_check_is_present() -> None:
    """With the gate engaged, what it *would* drop and what it *did* drop must be
    the same number. Reporting both is how a broken observe-only log gets caught."""
    report = build_routing_stats(*_fixture())
    assert report["control_gate_would_drop"] == report["control_dropped_entity_conflict"]


def test_an_answer_with_no_claims_is_not_counted_as_annotated() -> None:
    stats_by_arm, results, errors = _fixture()
    results[SUBJECT_ARM]["q3"] = _record("", ("unverified",))
    report = build_routing_stats(stats_by_arm, results, errors)
    assert report["annotated_claims_reaching_an_answer_nogate"] == 1
