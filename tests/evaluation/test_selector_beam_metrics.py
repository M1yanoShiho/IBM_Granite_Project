from evidence_rag.evaluation.selector_beam_metrics import (
    choose_threshold,
    compare_selector_arms,
)


def _row(
    query_id: str, *, harm: bool, recall: float, precision: float, count: int
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "harmful_pool_hit": True,
        "harmful_selected": harm,
        "required_evidence_recall": recall,
        "evidence_precision": precision,
        "selected_evidence_count": count,
    }


def test_comparison_uses_paired_query_differences() -> None:
    top = [
        _row("q1", harm=True, recall=1.0, precision=0.2, count=10),
        _row("q2", harm=True, recall=1.0, precision=0.2, count=10),
    ]
    beam = [
        _row("q1", harm=False, recall=1.0, precision=0.5, count=8),
        _row("q2", harm=True, recall=0.5, precision=0.4, count=7),
    ]
    report = compare_selector_arms(top, beam, seed=7, iterations=100)
    paired = report["paired"]
    assert isinstance(paired, dict)
    assert paired["harmful_in_context"]["delta_beam_minus_top_k"] == -0.5
    assert paired["required_recall"]["delta_beam_minus_top_k"] == -0.25


def test_threshold_policy_prioritizes_clear_recall_then_minimax_loss() -> None:
    rows = [
        {
            "required_threshold": 0.4,
            "reject_threshold": 0.6,
            "harm_delta": -0.5,
            "harm_ci_high": -0.4,
            "niah_recall_loss": 0.04,
            "twowiki_recall_loss": 0.04,
        },
        {
            "required_threshold": 0.5,
            "reject_threshold": 0.8,
            "harm_delta": -0.04,
            "harm_ci_high": -0.01,
            "niah_recall_loss": 0.01,
            "twowiki_recall_loss": 0.02,
        },
        {
            "required_threshold": 0.6,
            "reject_threshold": 0.9,
            "harm_delta": -0.06,
            "harm_ci_high": -0.02,
            "niah_recall_loss": 0.02,
            "twowiki_recall_loss": 0.02,
        },
    ]
    decision = choose_threshold(rows)
    assert decision["status"] == "CLEAR_PASS"
    assert decision["selected"] == rows[2]


def test_threshold_policy_fails_without_credible_harm_reduction() -> None:
    decision = choose_threshold(
        [
            {
                "required_threshold": 0.5,
                "reject_threshold": 0.8,
                "harm_delta": -0.02,
                "harm_ci_high": 0.01,
                "niah_recall_loss": 0.0,
                "twowiki_recall_loss": 0.0,
            }
        ]
    )
    assert decision["status"] == "FAIL"
    assert decision["selected"] is None
