import math

import pytest

from evidence_rag.evaluation.paired_metric import compare_paired


def test_compare_paired_basic() -> None:
    on = {"q1": 1.0, "q2": 1.0, "q3": 0.0}
    off = {"q1": 0.0, "q2": 1.0, "q3": 0.0}
    c = compare_paired(on, off, iterations=2000)
    assert c.n_paired == 3
    assert c.n_queries == 3
    assert c.n_clusters == 3
    assert math.isclose(c.mean_on, 2 / 3)
    assert math.isclose(c.mean_off, 1 / 3)
    assert math.isclose(c.delta, 1 / 3)
    assert 0.0 <= c.p_value <= 1.0
    assert c.ci_low <= c.delta <= c.ci_high


def test_compare_paired_skips_unscored() -> None:
    on = {"q1": 1.0, "q2": None, "q3": 0.0}
    off = {"q1": 0.0, "q2": None, "q3": 0.0}
    c = compare_paired(on, off, iterations=1000)
    assert c.n_paired == 2
    assert c.n_total == 3
    assert c.n_unscored == 1


def test_compare_paired_identical_is_nonsignificant() -> None:
    on = {"q1": 1.0, "q2": 0.0}
    off = {"q1": 1.0, "q2": 0.0}
    c = compare_paired(on, off, iterations=2000)
    assert c.delta == 0.0
    assert c.p_value == 1.0


def test_compare_paired_deterministic() -> None:
    on = {"q1": 1.0, "q2": 0.0, "q3": 1.0}
    off = {"q1": 0.0, "q2": 0.0, "q3": 0.0}
    assert compare_paired(on, off, seed=13, iterations=1000) == compare_paired(
        on, off, seed=13, iterations=1000
    )


def test_compare_paired_no_pairs_raises() -> None:
    try:
        compare_paired({"q1": None}, {"q1": 1.0})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_compare_paired_rejects_query_key_mismatch_instead_of_taking_intersection() -> None:
    with pytest.raises(ValueError, match="on/off query keys differ"):
        compare_paired(
            {"shared": 1.0, "only-on": 0.0},
            {"shared": 0.0, "only-off": 1.0},
        )


def test_none_is_an_explicit_unscored_value_not_a_missing_query() -> None:
    comparison = compare_paired(
        {"scored": 1.0, "unscored": None},
        {"scored": 0.0, "unscored": None},
        iterations=100,
    )

    assert comparison.n_queries == 1
    assert comparison.n_clusters == 1
    assert comparison.n_total == 2
    assert comparison.n_unscored == 1


@pytest.mark.parametrize(
    ("on_value", "off_value"),
    [(None, 1.0), (1.0, None)],
)
def test_scoring_masks_must_match(on_value: float | None, off_value: float | None) -> None:
    with pytest.raises(ValueError, match="scoring masks differ"):
        compare_paired({"q": on_value}, {"q": off_value})


def test_constant_distribution_has_degenerate_bootstrap_interval() -> None:
    on = {f"q-{index}": 0.75 for index in range(10)}
    off = {f"q-{index}": 0.25 for index in range(10)}

    comparison = compare_paired(on, off, iterations=500)

    assert comparison.delta == 0.5
    assert comparison.ci_low == 0.5
    assert comparison.ci_high == 0.5


def test_monte_carlo_p_value_uses_plus_one_correction() -> None:
    # With 25 independent +1 differences, an equally extreme sign flip requires all
    # 25 signs to agree. Seed 13 draws none in this short deterministic run, so the
    # corrected value is 1/(iterations+1), never the impossible p=0.
    on = {f"q-{index}": 1.0 for index in range(25)}
    off = {query_id: 0.0 for query_id in on}

    comparison = compare_paired(on, off, seed=13, iterations=31)

    assert comparison.p_value == 1 / 32
    assert comparison.p_value > 0.0


def test_component_keys_must_cover_the_same_queries() -> None:
    with pytest.raises(ValueError, match="metric/component query keys differ"):
        compare_paired(
            {"q1": 1.0, "q2": 0.0},
            {"q1": 0.0, "q2": 0.0},
            component_ids={"q1": "component-a"},
        )


def test_clustered_comparison_is_deterministic_and_order_invariant() -> None:
    on = {"q3": 1.0, "q1": 1.0, "q2": 0.0, "q4": 0.0}
    off = {"q1": 0.0, "q2": 1.0, "q3": 0.0, "q4": 0.0}
    components = {"q4": "c3", "q2": "c1", "q1": "c1", "q3": "c2"}

    first = compare_paired(on, off, component_ids=components, seed=73, iterations=1000)
    second = compare_paired(
        dict(reversed(tuple(on.items()))),
        dict(reversed(tuple(off.items()))),
        component_ids=dict(reversed(tuple(components.items()))),
        seed=73,
        iterations=1000,
    )

    assert first == second
    assert first.n_queries == 4
    assert first.n_clusters == 3


def test_one_giant_component_cannot_masquerade_as_many_independent_queries() -> None:
    on = {f"giant-{index}": 1.0 for index in range(100)}
    on["small"] = 0.0
    off = {query_id: 0.0 for query_id in on}
    off["small"] = 1.0
    components = {
        query_id: ("giant" if query_id.startswith("giant-") else "small")
        for query_id in on
    }

    query_level = compare_paired(on, off, seed=13, iterations=2000)
    clustered = compare_paired(
        on,
        off,
        component_ids=components,
        seed=13,
        iterations=2000,
    )

    assert clustered.n_queries == 101
    assert clustered.n_clusters == 2
    assert clustered.delta == query_level.delta  # point estimate remains query-weighted
    assert clustered.p_value > query_level.p_value
    assert clustered.ci_low < 0.0 < clustered.ci_high


def test_iterations_must_be_positive() -> None:
    with pytest.raises(ValueError, match="iterations must be positive"):
        compare_paired({"q": 1.0}, {"q": 0.0}, iterations=0)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_metric_values_are_rejected(invalid: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        compare_paired({"q": invalid}, {"q": 0.0})
