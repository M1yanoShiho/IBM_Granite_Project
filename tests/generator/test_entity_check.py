import pytest

from evidence_rag.generator.entity_check import (
    Entity,
    EntityConsistencyChecker,
    RuleBasedEntityExtractor,
)


def normalized(text: str, entity_type: str) -> set[str]:
    return {
        entity.normalized
        for entity in RuleBasedEntityExtractor().extract(text)
        if entity.entity_type == entity_type
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("$1.2B", "usd:1200000000"),
        ("1,200 million dollars", "usd:1200000000"),
        ("12亿美元", "usd:1200000000"),
        ("3.5 million euros", "eur:3500000"),
        ("8%", "8%"),
        ("8 percent", "8%"),
    ),
)
def test_amounts_normalize_across_unit_and_magnitude(text: str, expected: str) -> None:
    assert expected in normalized(text, "number")


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("2024-03-15", "2024-03-15"),
        ("March 15, 2024", "2024-03-15"),
        ("15 March 2024", "2024-03-15"),
        ("2024年3月15日", "2024-03-15"),
        ("March 2024", "2024-03"),
        ("Q1 2024", "2024-Q1"),
    ),
)
def test_dates_normalize_to_a_common_form(text: str, expected: str) -> None:
    assert expected in normalized(text, "date")


def test_year_range_also_yields_its_endpoints() -> None:
    assert normalized("from 2020-2024", "date") == {"2020..2024", "2020", "2024"}


def test_bare_year_is_a_date_not_a_quantity() -> None:
    assert normalized("in 2024", "date") == {"2024"}
    assert normalized("in 2024", "number") == set()


def test_organization_aliases_and_legal_suffixes_collapse() -> None:
    assert normalized("Pfizer Inc. and 辉瑞", "name") == {"pfizer"}


def test_swapped_organization_is_a_mismatch() -> None:
    result = EntityConsistencyChecker().check(
        "Acme reported revenue of $1.2B in 2024.",
        "Globex reported revenue of $1.2B in 2024.",
    )

    assert result.consistent is False
    mismatch = result.mismatches[0]
    assert mismatch.normalized == "acme"
    assert mismatch.evidence_values == ("globex",)


def test_same_facts_written_differently_are_consistent() -> None:
    result = EntityConsistencyChecker().check(
        "Acme reported revenue of $1.2B in 2024.",
        "Acme Inc. reported revenue of 1,200 million dollars in 2024.",
    )

    assert result.consistent is True
    assert result.mismatches == ()


def test_swapped_year_is_a_mismatch() -> None:
    result = EntityConsistencyChecker().check(
        "Revenue was $1.2B in 2024.",
        "Revenue was $1.2B in 2023.",
    )

    assert result.consistent is False
    assert [mismatch.normalized for mismatch in result.mismatches] == ["2024"]


def test_swapped_currency_is_a_mismatch() -> None:
    result = EntityConsistencyChecker().check(
        "Revenue was $1.2 billion.",
        "Revenue was 1.2 billion euros.",
    )

    assert result.consistent is False
    assert [mismatch.normalized for mismatch in result.mismatches] == ["usd:1200000000"]


def test_unqualified_amount_matches_a_currency_qualified_one() -> None:
    result = EntityConsistencyChecker().check(
        "Revenue was 1.2 billion.",
        "Revenue was $1.2 billion.",
    )

    assert result.consistent is True


def test_missing_date_is_a_mismatch_but_an_unnamed_evidence_is_not() -> None:
    checker = EntityConsistencyChecker()

    assert checker.check("Revenue rose in 2024.", "Revenue rose sharply.").consistent is False
    assert checker.check("Acme grew.", "The business grew.").consistent is True


def test_benign_name_variants_match_but_swaps_still_fail() -> None:
    # G3: tolerant name matching -- possessive/plural and partial-vs-full variants
    # are the same entity, while a genuine swap (disjoint tokens) still mismatches.
    checker = EntityConsistencyChecker()

    # possessive vs plural surface form: "King's Mountain" ~ "Kings Mountain"
    assert checker.check(
        "The battle of King's Mountain was a Patriot victory.",
        "The Battle of Kings Mountain was a decisive victory for the Patriots.",
    ).consistent is True
    # partial vs full: "Patriots" ~ "Patriot militia"
    assert checker.check(
        "The Patriots won.",
        "The Patriot militia won the engagement.",
    ).consistent is True
    # a real swap is NOT masked by the tolerance
    assert checker.check(
        "Sir Garfield Sobers scored the runs.",
        "Graham Gooch scored the runs.",
    ).consistent is False


def test_extractor_is_injectable() -> None:
    class StubExtractor:
        def extract(self, text: str) -> tuple[Entity, ...]:
            return (Entity(entity_type="organization", text=text, normalized=text),)

    checker = EntityConsistencyChecker(StubExtractor())

    assert checker.check("acme", "acme").consistent is True
    assert checker.check("acme", "globex").consistent is False
