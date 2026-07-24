from evidence_rag.evaluation.needle_probe import (
    NONE,
    RECOVERED,
    WRONG,
    ProbeOutcome,
    classify_extraction,
    summarize_probe,
)


def test_classify_recovered():
    assert classify_extraction(["Kennedy"], "Kennedy") == RECOVERED


def test_classify_wrong_when_valid_but_not_gold():
    assert classify_extraction(["Nixon"], "Kennedy") == WRONG


def test_classify_none_when_abstained_or_invalid():
    assert classify_extraction(["NONE"], "Kennedy") == NONE
    assert classify_extraction([""], "Kennedy") == NONE
    assert classify_extraction(["the"], "Kennedy") == NONE  # stopword -> invalid


def test_classify_recovered_via_canonicalization():
    # number canonicalization makes these one answer
    assert classify_extraction(["1,200"], "1200") == RECOVERED


def test_recovered_wins_over_wrong_across_chunks():
    assert classify_extraction(["Nixon", "Kennedy"], "Kennedy") == RECOVERED


def test_summarize_splits_by_visibility():
    outcomes = [
        ProbeOutcome("q1", True, RECOVERED, "Kennedy", "Kennedy"),
        ProbeOutcome("q2", True, NONE, "NONE", "Kennedy"),
        ProbeOutcome("q3", True, WRONG, "Nixon", "Kennedy"),
        ProbeOutcome("q4", False, NONE, "NONE", "Kennedy"),
    ]
    summary = summarize_probe(outcomes)
    assert summary.n == 4
    assert summary.recovered == 1
    assert summary.wrong == 1
    assert summary.none == 2
    assert summary.n_visible == 3
    assert summary.visible_recovered == 1
    assert summary.visible_wrong == 1
    assert summary.visible_none == 1
