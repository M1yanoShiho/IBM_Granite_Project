import math

import pytest

from evidence_rag.evaluation.selector_risk import (
    COMPONENT_REPRESENTATIVE_SEED,
    NIAH_CHAIN_RISK,
    NIAH_RECALL_RISK,
    REQUIRED_RISKS,
    TWOWIKI_CHAIN_RISK,
    TWOWIKI_RECALL_RISK,
    ComponentCandidate,
    RiskLossTable,
    conditional_chain_loss,
    corrected_risk,
    deletion_precision,
    document_recall,
    harm_reduction,
    minimum_calibration_size,
    nonzero_policy_can_qualify,
    relative_recall_loss,
    representative_digest,
    required_deletion_rate,
    select_component_representatives,
    select_crc_policy,
    validate_monotone_losses,
    validate_nested_policy_sets,
)


def test_relative_recall_loss_uses_document_id_sets_and_topk_minus_selector() -> None:
    baseline = {"doc-a", "doc-b", "doc-noise"}
    selector = {"doc-a", "doc-noise"}
    gold = {"doc-a", "doc-b", "doc-c"}

    assert document_recall(baseline, gold) == pytest.approx(2 / 3)
    assert document_recall(selector, gold) == pytest.approx(1 / 3)
    assert relative_recall_loss(
        baseline_document_ids=baseline,
        selector_document_ids=selector,
        gold_document_ids=gold,
    ) == pytest.approx(1 / 3)


def test_relative_recall_loss_rejects_rank_replacement_and_empty_gold() -> None:
    with pytest.raises(ValueError, match="subset"):
        relative_recall_loss(
            baseline_document_ids={"doc-a"},
            selector_document_ids={"doc-new"},
            gold_document_ids={"doc-a"},
        )
    with pytest.raises(ValueError, match="must not be empty"):
        relative_recall_loss(
            baseline_document_ids={"doc-a"},
            selector_document_ids={"doc-a"},
            gold_document_ids=set(),
        )


def test_conditional_chain_loss_only_scores_topk_complete_chains() -> None:
    gold = {"doc-a", "doc-b"}
    assert (
        conditional_chain_loss(
            baseline_document_ids={"doc-a", "doc-b", "noise"},
            selector_document_ids={"doc-a", "noise"},
            gold_document_ids=gold,
        )
        == 1.0
    )
    assert (
        conditional_chain_loss(
            baseline_document_ids={"doc-a", "doc-b", "noise"},
            selector_document_ids={"doc-a", "doc-b"},
            gold_document_ids=gold,
        )
        == 0.0
    )
    assert (
        conditional_chain_loss(
            baseline_document_ids={"doc-a", "noise"},
            selector_document_ids={"doc-a"},
            gold_document_ids=gold,
        )
        is None
    )


def test_harm_reduction_sign_is_baseline_minus_selector() -> None:
    assert harm_reduction(baseline_harm=0.40, selector_harm=0.38) == pytest.approx(0.02)
    assert harm_reduction(baseline_harm=0.20, selector_harm=0.30) == pytest.approx(-0.10)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        harm_reduction(baseline_harm=1.1, selector_harm=0.3)


def test_deletion_metrics_use_all_drop_actions_as_denominator() -> None:
    dropped = ("doc-harm", "doc-required", "doc-neutral", "doc-harm")
    assert deletion_precision(
        dropped_document_ids=dropped, harmful_document_ids={"doc-harm"}
    ) == pytest.approx(0.5)
    assert required_deletion_rate(
        dropped_document_ids=dropped, required_document_ids={"doc-required"}
    ) == pytest.approx(0.25)


def test_deletion_metrics_are_unscored_when_nothing_was_deleted() -> None:
    assert deletion_precision(dropped_document_ids=(), harmful_document_ids={"harm"}) is None
    assert required_deletion_rate(
        dropped_document_ids=(), required_document_ids={"required"}
    ) is None


def test_component_representative_is_minimum_frozen_hash_and_order_independent() -> None:
    candidates = (
        ComponentCandidate("2wiki", "recall", "component-a", "q3"),
        ComponentCandidate("2wiki", "recall", "component-a", "q1"),
        ComponentCandidate("2wiki", "recall", "component-a", "q2"),
        ComponentCandidate("2wiki", "chain", "component-a", "q4"),
        ComponentCandidate("2wiki", "recall", "component-b", "q5"),
    )
    expected = min(
        candidates[:3],
        key=lambda item: (
            representative_digest(item, seed=COMPONENT_REPRESENTATIVE_SEED),
            item.query_id,
        ),
    )

    selected = select_component_representatives(candidates)
    reversed_selected = select_component_representatives(reversed(candidates))

    assert selected == reversed_selected
    assert len(selected) == 3
    assert expected in selected


def test_component_representative_rejects_duplicate_join_rows() -> None:
    candidate = ComponentCandidate("niah", "recall", "component-a", "q1")
    with pytest.raises(ValueError, match="duplicate"):
        select_component_representatives((candidate, candidate))


def test_component_representative_rejects_one_query_in_two_components() -> None:
    with pytest.raises(ValueError, match="cannot belong to two components"):
        select_component_representatives(
            (
                ComponentCandidate("niah", "recall", "component-a", "q1"),
                ComponentCandidate("niah", "recall", "component-b", "q1"),
            )
        )


def test_component_candidate_requires_nonblank_string_ids() -> None:
    with pytest.raises(ValueError, match="non-blank string"):
        ComponentCandidate("niah", "recall", "component-a", 1)  # type: ignore[arg-type]


def test_representative_digest_has_a_literal_golden_value() -> None:
    candidate = ComponentCandidate(
        "niah-dev", "niah_required_recall", "component-a", "q1"
    )
    assert representative_digest(candidate) == (
        "99f6c3fe4b4bfc2223b84be0a3d3c2ad6ba7d7f026542352bd12bac097011efb"
    )


def test_nested_policy_sets_accept_deletion_only_p0_through_p6() -> None:
    policies: tuple[set[str], ...] = (
        {"a", "b", "c", "d", "e", "f"},
        {"a", "b", "c", "d", "e"},
        {"a", "b", "c", "d"},
        {"a", "b", "c"},
        {"a", "b"},
        {"a"},
        set(),
    )
    assert validate_nested_policy_sets(policies, baseline_ids=policies[0])[0] == frozenset(
        policies[0]
    )


def test_nested_policy_sets_reject_an_id_reintroduced_by_a_stronger_policy() -> None:
    policies: tuple[set[str], ...] = (
        {"a", "b"},
        {"a"},
        {"a", "new"},
        {"a"},
        {"a"},
        {"a"},
        {"a"},
    )
    with pytest.raises(ValueError, match="not a subset"):
        validate_nested_policy_sets(policies, baseline_ids={"a", "b"})


def test_nested_policy_sets_require_p0_to_equal_the_frozen_top_k_baseline() -> None:
    policies = tuple({"a"} for _ in range(7))
    with pytest.raises(ValueError, match="P0 must be exactly"):
        validate_nested_policy_sets(policies, baseline_ids={"a", "b"})


def test_monotone_losses_are_checked_per_representative() -> None:
    valid = (
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (1.0, 1.0),
        (1.0, 1.0),
        (1.0, 1.0),
    )
    assert validate_monotone_losses(valid) == valid

    invalid = list(valid)
    invalid[3] = (0.0, 0.0)
    with pytest.raises(ValueError, match="not monotone"):
        validate_monotone_losses(tuple(invalid))


def test_crc_correction_and_one_percent_sample_floor() -> None:
    assert corrected_risk((0.0,) * 99) == pytest.approx(0.01)
    assert corrected_risk((0.0,) * 98) > 0.01
    assert minimum_calibration_size(alpha=0.01) == 99
    assert not nonzero_policy_can_qualify(98, alpha=0.01)
    assert nonzero_policy_can_qualify(99, alpha=0.01)


def _risk_table(risk_name: str, *, strongest_qualified: int, n: int = 199) -> RiskLossTable:
    representatives = tuple(f"{risk_name}-q{index}" for index in range(n))
    components = tuple(f"{risk_name}-component-{index}" for index in range(n))
    passing = (0.0,) * n
    failing = (1.0, 1.0) + (0.0,) * (n - 2)
    losses = tuple(
        passing if policy_index <= strongest_qualified else failing
        for policy_index in range(7)
    )
    return RiskLossTable(
        risk_name=risk_name,
        representative_ids=representatives,
        representative_component_ids=components,
        losses_by_policy=losses,
    )


def test_crc_selects_max_per_risk_then_min_across_four_risks() -> None:
    tables = (
        _risk_table(NIAH_RECALL_RISK, strongest_qualified=5),
        _risk_table(NIAH_CHAIN_RISK, strongest_qualified=3),
        _risk_table(TWOWIKI_RECALL_RISK, strongest_qualified=4),
        _risk_table(TWOWIKI_CHAIN_RISK, strongest_qualified=2),
    )

    selection = select_crc_policy(reversed(tables), alpha=0.01)

    assert selection.final_policy == 2
    assert not selection.structural_fallback
    assert selection.decision_for(NIAH_RECALL_RISK).strongest_qualified_policy == 5
    assert selection.decision_for(NIAH_CHAIN_RISK).strongest_qualified_policy == 3
    assert selection.decision_for(TWOWIKI_RECALL_RISK).strongest_qualified_policy == 4
    assert selection.decision_for(TWOWIKI_CHAIN_RISK).strongest_qualified_policy == 2
    assert selection.decision_for(NIAH_RECALL_RISK).corrected_risks[0] is None


def test_crc_uses_structural_p0_when_one_risk_has_too_few_components() -> None:
    tables = [_risk_table(risk_name, strongest_qualified=6) for risk_name in REQUIRED_RISKS]
    too_small = _risk_table(TWOWIKI_CHAIN_RISK, strongest_qualified=6, n=98)
    tables[-1] = too_small

    selection = select_crc_policy(tables, alpha=0.01)

    blocked = selection.decision_for(TWOWIKI_CHAIN_RISK)
    assert blocked.strongest_qualified_policy == 0
    assert not blocked.nonzero_possible_by_size
    assert blocked.status == "BLOCKED_BY_SAMPLE_SIZE"
    assert selection.final_policy == 0
    assert selection.structural_fallback
    assert selection.blocked_by_sample_size


def test_crc_records_no_calibration_evidence_explicitly() -> None:
    tables = [_risk_table(risk_name, strongest_qualified=6) for risk_name in REQUIRED_RISKS]
    tables[-1] = _risk_table(TWOWIKI_CHAIN_RISK, strongest_qualified=6, n=0)

    selection = select_crc_policy(tables, alpha=0.01)

    missing = selection.decision_for(TWOWIKI_CHAIN_RISK)
    assert missing.status == "NO_CALIBRATION_EVIDENCE"
    assert selection.has_no_calibration_evidence
    assert selection.final_policy == 0


def test_crc_requires_exactly_the_four_pre_registered_risks() -> None:
    with pytest.raises(ValueError, match="exactly the four"):
        select_crc_policy((_risk_table(NIAH_RECALL_RISK, strongest_qualified=1),))


def test_risk_table_rejects_nonzero_p0_and_nonmonotone_policy_loss() -> None:
    representatives = ("q1",)
    components = ("component-1",)
    nonzero_p0 = tuple(((1.0,) if index == 0 else (1.0,)) for index in range(7))
    with pytest.raises(ValueError, match="P0"):
        RiskLossTable(NIAH_RECALL_RISK, representatives, components, nonzero_p0)

    nonmonotone = ((0.0,), (1.0,), (0.0,), (1.0,), (1.0,), (1.0,), (1.0,))
    with pytest.raises(ValueError, match="not monotone"):
        RiskLossTable(NIAH_RECALL_RISK, representatives, components, nonmonotone)


def test_risk_table_rejects_duplicate_representative_components() -> None:
    losses = tuple((0.0, 0.0) for _ in range(7))
    with pytest.raises(ValueError, match="at most one representative"):
        RiskLossTable(
            NIAH_RECALL_RISK,
            ("q1", "q2"),
            ("same-component", "same-component"),
            losses,
        )


def test_risk_table_normalizes_mutable_inputs_to_tuples() -> None:
    table = RiskLossTable(
        NIAH_RECALL_RISK,
        ["q1"],  # type: ignore[arg-type]
        ["component-1"],  # type: ignore[arg-type]
        [[0.0] for _ in range(7)],  # type: ignore[arg-type]
    )
    assert isinstance(table.representative_ids, tuple)
    assert isinstance(table.representative_component_ids, tuple)
    assert all(isinstance(row, tuple) for row in table.losses_by_policy)


def test_corrected_risk_rejects_unbounded_or_nonfinite_losses() -> None:
    with pytest.raises(ValueError, match="loss must"):
        corrected_risk((1.1,))
    with pytest.raises(ValueError, match="loss must"):
        corrected_risk((math.nan,))
