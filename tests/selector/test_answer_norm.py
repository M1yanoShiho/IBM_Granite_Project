from evidence_rag.selector.answer_norm import canonicalize_answer, is_valid_answer, normalize_answer


def test_text_normalization_is_preserved() -> None:
    assert normalize_answer("The Acme.") == "acme"
    assert canonicalize_answer("Acme Corp.") == "acme corp"


def test_thousands_separators_collapse() -> None:
    assert canonicalize_answer("1,200") == "1200"
    assert canonicalize_answer("1200") == "1200"


def test_currency_and_magnitude_aliases_merge() -> None:
    assert canonicalize_answer("$1.2B") == "1200000000"
    assert canonicalize_answer("1,200 million") == "1200000000"
    assert canonicalize_answer("1.2 billion") == "1200000000"
    assert canonicalize_answer("US$500m") == "500000000"


def test_percent_forms_merge() -> None:
    assert canonicalize_answer("18%") == "18"
    assert canonicalize_answer("18 percent") == "18"


def test_trailing_zero_trim() -> None:
    assert canonicalize_answer("$45.30") == "45.3"


def test_date_whitelist_merges() -> None:
    assert canonicalize_answer("January 5, 2019") == "2019-01-05"
    assert canonicalize_answer("5 January 2019") == "2019-01-05"
    assert canonicalize_answer("2019-01-05") == "2019-01-05"


def test_month_year_is_not_parsed() -> None:
    assert canonicalize_answer("March 2019") == "march 2019"


def test_mixed_tokens_fall_back_to_text() -> None:
    assert canonicalize_answer("covid-19") == "covid-19"
    assert canonicalize_answer("three") == "three"


def test_idempotent() -> None:
    for value in ("$1.2B", "January 5, 2019", "Acme Corp.", "18%"):
        once = canonicalize_answer(value)
        assert canonicalize_answer(once) == once


def test_validity_rules_unchanged() -> None:
    assert is_valid_answer("Acme")
    assert not is_valid_answer("none")
    assert not is_valid_answer("it")
    assert is_valid_answer("42")
