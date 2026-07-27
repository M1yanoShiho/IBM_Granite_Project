import math

from evidence_rag.evaluation.paired_metric import compare_paired


def test_compare_paired_basic() -> None:
    on = {"q1": 1.0, "q2": 1.0, "q3": 0.0}
    off = {"q1": 0.0, "q2": 1.0, "q3": 0.0}
    c = compare_paired(on, off, iterations=2000)
    assert c.n_paired == 3
    assert math.isclose(c.mean_on, 2 / 3)
    assert math.isclose(c.mean_off, 1 / 3)
    assert math.isclose(c.delta, 1 / 3)
    assert 0.0 <= c.p_value <= 1.0
    assert c.ci_low <= c.delta <= c.ci_high


def test_compare_paired_skips_unscored() -> None:
    on = {"q1": 1.0, "q2": None, "q3": 0.0}
    off = {"q1": 0.0, "q2": 0.0, "q3": None}
    c = compare_paired(on, off, iterations=1000)
    assert c.n_paired == 1  # only q1 is scored in both arms


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
