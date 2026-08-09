"""The tiers have to be decidable by machine, or they are the -1% guard again with more words."""

import pytest

from evidence_rag.evaluation.acceptance import classify


def test_a_clean_win_is_a_clean_win() -> None:
    verdict = classify(
        harm_delta=-0.105,
        harm_ci_high=-0.088,
        recall_losses={"required_recall": 0.012, "twowiki_supporting_recall": 0.004},
    )
    assert verdict.tier == "clear-pass"
    assert verdict.passed


def test_a_negative_point_estimate_whose_ci_straddles_zero_is_not_a_reduction() -> None:
    """S8's two axes both had negative-looking point estimates and CIs across zero. Reading
    those as "no cost" was the specific mistake the S3 wording had to be corrected for, so the
    distinction is mechanical here rather than left to whoever writes the summary."""

    verdict = classify(
        harm_delta=-0.0018,
        harm_ci_high=+0.0040,
        recall_losses={"required_recall": 0.001},
    )
    assert verdict.tier == "fail"
    assert "does not clear zero" in verdict.reasons[0]


def test_a_loss_inside_the_middle_band_is_reported_as_a_trade_not_a_win() -> None:
    verdict = classify(
        harm_delta=-0.112,
        harm_ci_high=-0.093,
        recall_losses={"required_recall": 0.048},
    )
    assert verdict.tier == "tradeoff"
    assert verdict.passed, "a trade is still an accepting tier -- it is just not a clean one"
    assert "between" in verdict.reasons[0]


def test_a_loss_past_the_outer_band_fails_however_good_the_harm_number_is() -> None:
    """Reliability-MIS is the case in the file: a very large harm reduction bought with a recall
    collapse. No harm number rescues that, which is why the recall test is not a tie-break."""

    verdict = classify(
        harm_delta=-0.604,
        harm_ci_high=-0.560,
        recall_losses={"required_recall": 0.372},
    )
    assert verdict.tier == "fail"


def test_every_offending_metric_is_named_not_just_the_first() -> None:
    """Fixing whichever one sorted first would only surface the next a run later."""

    verdict = classify(
        harm_delta=+0.010,
        harm_ci_high=+0.030,
        recall_losses={"a_recall": 0.09, "b_recall": 0.07},
    )
    assert verdict.tier == "fail"
    joined = " ".join(verdict.reasons)
    assert "harm not credibly reduced" in joined
    assert "a_recall" in joined and "b_recall" in joined


def test_an_arm_is_judged_on_every_recall_it_could_damage() -> None:
    """One clean axis does not carry the verdict: E2 watched required recall while nothing
    watched 2Wiki supporting recall, and an arm trading one for the other would have read
    clean."""

    verdict = classify(
        harm_delta=-0.080,
        harm_ci_high=-0.060,
        recall_losses={"required_recall": 0.005, "twowiki_supporting_recall": 0.061},
    )
    assert verdict.tier == "fail"


def test_judging_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one recall metric"):
        classify(harm_delta=-0.1, harm_ci_high=-0.05, recall_losses={})


def test_incoherent_bands_are_refused() -> None:
    with pytest.raises(ValueError, match="clean_loss"):
        classify(
            harm_delta=-0.1,
            harm_ci_high=-0.05,
            recall_losses={"required_recall": 0.01},
            clean_loss=0.09,
            max_loss=0.05,
        )
