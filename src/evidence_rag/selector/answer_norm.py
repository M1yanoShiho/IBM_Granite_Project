"""Deterministic answer canonicalization (spec §6.2).

Cluster keys and corroboration votes compare answers AFTER canonicalization so that
numeric aliases ("$1.2B" vs "1,200 million") land in one cluster. Rules are
deterministic and unit-tested; known tradeoff: bare magnitude suffixes ("19b")
are read as numbers — the §12 component eval (false-conflict rate) measures the
cost of such merges on real pools.
"""

import re
from decimal import Decimal, InvalidOperation

MIN_ANSWER_LENGTH = 3
STOPWORDS = {"the", "a", "an", "none", "n/a", "unknown", "it", "yes", "no"}

_CURRENCY_PREFIX = re.compile(r"^(?:us\$|\$|£|€)\s*")
_PERCENT_SUFFIX = re.compile(r"\s*(?:%|percent|per cent)$")
_NUMBER_PATTERN = re.compile(
    r"^(?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(?P<magnitude>k|m|b|thousand|million|billion|trillion)?$"
)
_MAGNITUDES = {
    "k": Decimal(1_000),
    "thousand": Decimal(1_000),
    "m": Decimal(1_000_000),
    "million": Decimal(1_000_000),
    "b": Decimal(1_000_000_000),
    "billion": Decimal(1_000_000_000),
    "trillion": Decimal(10**12),
}
_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_DATE_ISO = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_DATE_DMY = re.compile(r"^(\d{1,2})\s+([a-z]+)\s+(\d{4})$")
_DATE_MDY = re.compile(r"^([a-z]+)\s+(\d{1,2}),?\s+(\d{4})$")


def normalize_answer(answer: str) -> str:
    normalized = answer.strip().lower()
    normalized = re.sub(r"^(the|a|an)\s+", "", normalized)
    return normalized.strip(" \t\n.,;:!?\"'()[]")


def is_valid_answer(answer: str) -> bool:
    normalized = normalize_answer(answer)
    if not normalized or normalized in STOPWORDS:
        return False
    if len(normalized) < MIN_ANSWER_LENGTH and not normalized.isdigit():
        return False
    return True


def _format_decimal(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _canonical_number(text: str) -> str | None:
    stripped = _PERCENT_SUFFIX.sub("", _CURRENCY_PREFIX.sub("", text))
    match = _NUMBER_PATTERN.match(stripped)
    if match is None:
        return None
    try:
        value = Decimal(match.group("number").replace(",", ""))
    except InvalidOperation:
        return None
    magnitude = match.group("magnitude")
    if magnitude is not None:
        value *= _MAGNITUDES[magnitude]
    return _format_decimal(value)


def _build_date(year: str, month: int, day: str) -> str | None:
    day_number = int(day)
    if not 1 <= day_number <= 31:
        return None
    return f"{int(year):04d}-{month:02d}-{day_number:02d}"


def _canonical_date(text: str) -> str | None:
    if match := _DATE_ISO.match(text):
        year, month, day = match.groups()
        if 1 <= int(month) <= 12:
            return _build_date(year, int(month), day)
        return None
    if match := _DATE_DMY.match(text):
        day, month_name, year = match.groups()
        if month_name in _MONTHS:
            return _build_date(year, _MONTHS[month_name], day)
        return None
    if match := _DATE_MDY.match(text):
        month_name, day, year = match.groups()
        if month_name in _MONTHS:
            return _build_date(year, _MONTHS[month_name], day)
    return None


def canonicalize_answer(answer: str) -> str:
    text = normalize_answer(answer)
    if (date := _canonical_date(text)) is not None:
        return date
    if (number := _canonical_number(text)) is not None:
        return number
    return text
