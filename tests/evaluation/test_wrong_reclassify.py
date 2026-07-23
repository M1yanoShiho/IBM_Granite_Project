from evidence_rag.evaluation.wrong_reclassify import (
    DIFFERENT,
    EQUAL_AFTER_NORM,
    EXTRACTED_IN_GOLD,
    GOLD_IN_EXTRACTED,
    is_sublist,
    normalize_tokens,
    reclassify,
    summarize_reclass,
)


def test_is_sublist():
    assert is_sublist(("paul",), ("apostle", "paul")) is True
    assert is_sublist(("trade",), ("trade", "with", "chinese", "markets")) is True
    assert is_sublist(("6",), ("1960", "s")) is False  # word-level, no spurious substring
    assert is_sublist((), ("anything",)) is False


def test_normalize_strips_leading_function_words_and_number_words():
    assert normalize_tokens("in 2009") == ("2009",)
    assert normalize_tokens("thirty minutes") == ("30", "minutes")
    assert normalize_tokens("to trade with chinese markets") == (
        "trade", "with", "chinese", "markets",
    )


def test_reclassify_equal_after_norm_leading_prep():
    assert reclassify("2009", "in 2009") == EQUAL_AFTER_NORM


def test_reclassify_equal_after_norm_number_word():
    assert reclassify("30 minutes", "thirty minutes") == EQUAL_AFTER_NORM


def test_reclassify_gold_in_extracted():
    assert reclassify("Apostle Paul", "paul") == GOLD_IN_EXTRACTED
    assert reclassify("Moldovan leu", "leu") == GOLD_IN_EXTRACTED


def test_reclassify_extracted_in_gold():
    assert reclassify("trade", "to trade with chinese markets") == EXTRACTED_IN_GOLD


def test_reclassify_different_for_synonyms_and_real_errors():
    assert reclassify("chemical bonds", "intramolecular force") == DIFFERENT
    assert reclassify("Reba McEntire", "linda davis") == DIFFERENT


def test_summarize_counts_and_recoverable_rate():
    rows = [
        {"extracted": "Apostle Paul", "gold_value": "paul"},
        {"extracted": "2009", "gold_value": "in 2009"},
        {"extracted": "trade", "gold_value": "to trade with chinese markets"},
        {"extracted": "Reba McEntire", "gold_value": "linda davis"},
    ]
    summary = summarize_reclass(rows)
    assert summary.n == 4
    assert summary.gold_in_extracted == 1
    assert summary.equal_after_norm == 1
    assert summary.extracted_in_gold == 1
    assert summary.different == 1
    assert summary.recoverable == 3
    assert summary.recoverable_rate == 0.75
