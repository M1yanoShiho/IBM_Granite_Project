"""B3 -- entity consistency between a claim and one piece of evidence.

Catches the counterfactual-injection case the project is built around (plan
section 4): the evidence *reads* like it supports the claim -- NLI happily says
entailment -- but an entity was swapped, so the claim is about a different
company, year, or amount than the evidence is. An entity mismatch downgrades an
entailed claim to "unsupported".

Extraction is injectable. The default ``RuleBasedEntityExtractor`` adds no
dependency and deliberately puts its accuracy budget into dates, numbers and
money -- the entities that are easiest to tamper with and easiest to extract
exactly. ``SpacyEntityExtractor`` upgrades only the proper-noun side (org /
person / location / product), reusing the same date and number regexes.
"""

import importlib
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, Protocol

EntityType = Literal["organization", "person", "location", "product", "name", "date", "number"]

NAME_TYPES: frozenset[str] = frozenset(
    {"organization", "person", "location", "product", "name"}
)
"""Types that compare against each other. ``name`` is what the rule-based
extractor emits when it recognises a proper noun but cannot tell an org from a
person; letting the name-ish types match cross-type keeps a spaCy-extracted
evidence comparable with a rule-extracted claim instead of failing every pair on
a label difference."""


@dataclass(frozen=True)
class Entity:
    entity_type: EntityType
    text: str
    """Surface form, kept for mismatch reporting."""
    normalized: str
    """Canonical form entities are actually compared on."""


@dataclass(frozen=True)
class EntityMismatch:
    """One claim entity with no consistent counterpart in the evidence."""

    entity_type: EntityType
    claim_text: str
    normalized: str
    evidence_values: tuple[str, ...]
    """Normalized same-role values the evidence did carry -- non-empty means the
    entity conflicts (the counterfactual swap), empty means it is simply absent."""


@dataclass(frozen=True)
class EntityConsistency:
    consistent: bool
    mismatches: tuple[EntityMismatch, ...]


class EntityExtractor(Protocol):
    def extract(self, text: str) -> tuple[Entity, ...]: ...


class EntityChecker(Protocol):
    def check(self, claim_text: str, evidence_text: str) -> EntityConsistency: ...


# --------------------------------------------------------------------------
# normalization tables
# --------------------------------------------------------------------------

MAGNITUDES: Mapping[str, int] = {
    "k": 10**3,
    "thousand": 10**3,
    "千": 10**3,
    "m": 10**6,
    "mn": 10**6,
    "million": 10**6,
    "百万": 10**6,
    "b": 10**9,
    "bn": 10**9,
    "billion": 10**9,
    "t": 10**12,
    "trillion": 10**12,
    "万": 10**4,
    "千万": 10**7,
    "亿": 10**8,
    "万亿": 10**12,
}

CURRENCIES: Mapping[str, str] = {
    "$": "usd",
    "us$": "usd",
    "usd": "usd",
    "dollar": "usd",
    "dollars": "usd",
    "美元": "usd",
    "€": "eur",
    "eur": "eur",
    "euro": "eur",
    "euros": "eur",
    "欧元": "eur",
    "£": "gbp",
    "gbp": "gbp",
    "pound": "gbp",
    "pounds": "gbp",
    "英镑": "gbp",
    "¥": "cny",
    "cny": "cny",
    "rmb": "cny",
    "yuan": "cny",
    "人民币": "cny",
    "jpy": "jpy",
    "日元": "jpy",
}

PERCENT_UNITS: frozenset[str] = frozenset({"%", "percent", "percentage points"})

MONTHS: Mapping[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

LEGAL_SUFFIXES: tuple[str, ...] = (
    "incorporated",
    "inc",
    "corporation",
    "corp",
    "company",
    "co",
    "limited",
    "ltd",
    "llc",
    "plc",
    "gmbh",
    "ag",
    "nv",
    "sa",
    "holdings",
    "group",
)

LEADING_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "he",
        "his",
        "she",
        "her",
        "they",
        "their",
        "we",
        "our",
        "i",
        "in",
        "on",
        "at",
        "by",
        "for",
        "from",
        "to",
        "of",
        "and",
        "but",
        "or",
        "if",
        "when",
        "while",
        "after",
        "before",
        "however",
        "therefore",
        "there",
        "here",
        "what",
        "which",
        "who",
        "why",
        "how",
        "as",
        "is",
        "was",
        "were",
        "are",
        "be",
        "been",
        "no",
        "not",
        "also",
        "both",
        "each",
        "per",
        "according",
        "during",
        "between",
        "over",
        "under",
        "about",
    }
)

DEFAULT_ALIASES: Mapping[str, str] = {
    "国际商业机器": "ibm",
    "international business machines": "ibm",
    "辉瑞": "pfizer",
    "微软": "microsoft",
    "谷歌": "google",
    "亚马逊": "amazon",
    "阿里巴巴": "alibaba",
}
"""Seed alias map -- extensible, not exhaustive. CJK keys are matched against the
raw text directly, since the proper-noun regex is capitalization-based and
cannot see them."""


# --------------------------------------------------------------------------
# date / number extraction (shared by both extractors)
# --------------------------------------------------------------------------

_MONTH_PATTERN = "|".join(sorted(MONTHS, key=len, reverse=True))
_CURRENCY_PREFIX = r"US\$|A\$|C\$|\$|€|£|¥|USD|EUR|GBP|CNY|RMB|JPY|人民币|美元|欧元|英镑|日元"
_CURRENCY_SUFFIX = (
    r"美元|欧元|英镑|日元|人民币|dollars|dollar|euros|euro|pounds|pound|yuan|USD|EUR|GBP|JPY|CNY"
)
_MAGNITUDE_PATTERN = r"trillion|billion|million|thousand|bn|mn|万亿|千万|百万|亿|万|千|[KMBT]\b"

YEAR_RANGE_RE = re.compile(
    r"\b(?P<start>(?:19|20)\d{2})\s*(?:-|–|—|~|to|至|到)\s*(?P<end>(?:19|20)\d{2})\b",
    re.IGNORECASE,
)
ISO_DATE_RE = re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\b")
CN_DATE_RE = re.compile(r"(?P<year>\d{4})年(?:(?P<month>\d{1,2})月(?:(?P<day>\d{1,2})日)?)?")
MDY_DATE_RE = re.compile(
    rf"\b(?P<month>{_MONTH_PATTERN})\.?\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<year>\d{{4}})\b",
    re.IGNORECASE,
)
DMY_DATE_RE = re.compile(
    rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month>{_MONTH_PATTERN})\.?,?\s+(?P<year>\d{{4}})\b",
    re.IGNORECASE,
)
MONTH_YEAR_RE = re.compile(
    rf"\b(?P<month>{_MONTH_PATTERN})\.?,?\s+(?P<year>\d{{4}})\b",
    re.IGNORECASE,
)
QUARTER_RE = re.compile(
    r"\b(?:Q(?P<q>[1-4])\s*(?:of\s+)?(?:FY)?\s*(?P<qyear>\d{4})"
    r"|(?:FY)?(?P<year>\d{4})\s*Q(?P<q2>[1-4]))\b",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
NUMBER_RE = re.compile(
    rf"(?P<prefix>{_CURRENCY_PREFIX})?\s*"
    r"(?P<value>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    rf"\s*(?P<magnitude>{_MAGNITUDE_PATTERN})?"
    rf"\s*(?P<suffix>%|percent|percentage points|{_CURRENCY_SUFFIX})?",
    re.IGNORECASE,
)

Handler = Callable[["re.Match[str]"], list[Entity]]


def format_amount(amount: Decimal) -> str:
    """Render a magnitude-normalized amount without exponent or trailing zeros,
    so ``$1.2B`` and ``1,200 million`` land on the same string."""
    normalized = amount.normalize()
    if normalized == normalized.to_integral_value():
        normalized = normalized.to_integral_value()
    return format(normalized, "f")


def _date_entity(text: str, normalized: str) -> Entity:
    return Entity(entity_type="date", text=text, normalized=normalized)


def _handle_year_range(match: "re.Match[str]") -> list[Entity]:
    start, end = match.group("start"), match.group("end")
    # Endpoints are emitted alongside the range so a claim about a single year
    # still matches evidence that states the year only as part of a range.
    return [
        _date_entity(match.group(0), f"{start}..{end}"),
        _date_entity(start, start),
        _date_entity(end, end),
    ]


def _handle_ymd(match: "re.Match[str]") -> list[Entity]:
    year, month, day = match.group("year"), match.group("month"), match.group("day")
    if month is None:
        return [_date_entity(match.group(0), year)]
    normalized = f"{year}-{int(month):02d}"
    if day is not None:
        normalized = f"{normalized}-{int(day):02d}"
    return [_date_entity(match.group(0), normalized), _date_entity(year, year)]


def _handle_named_month(match: "re.Match[str]") -> list[Entity]:
    groups = match.groupdict()
    month = MONTHS.get(groups["month"].lower())
    if month is None:  # pragma: no cover - pattern is built from MONTHS
        return []
    year = groups["year"]
    normalized = f"{year}-{month:02d}"
    day = groups.get("day")
    if day is not None:
        normalized = f"{normalized}-{int(day):02d}"
    return [_date_entity(match.group(0), normalized), _date_entity(year, year)]


def _handle_quarter(match: "re.Match[str]") -> list[Entity]:
    groups = match.groupdict()
    year = groups["qyear"] or groups["year"]
    quarter = groups["q"] or groups["q2"]
    return [_date_entity(match.group(0), f"{year}-Q{quarter}"), _date_entity(year, year)]


def _handle_year(match: "re.Match[str]") -> list[Entity]:
    year = match.group(0)
    return [_date_entity(year, year)]


def _handle_number(match: "re.Match[str]") -> list[Entity]:
    try:
        amount = Decimal(match.group("value").replace(",", ""))
    except InvalidOperation:  # pragma: no cover - pattern only admits decimals
        return []
    magnitude = match.group("magnitude")
    if magnitude:
        factor = MAGNITUDES.get(magnitude.strip().lower())
        if factor is None:  # pragma: no cover - pattern is built from MAGNITUDES
            return []
        amount *= factor
    prefix = (match.group("prefix") or "").strip().lower()
    suffix = (match.group("suffix") or "").strip().lower()
    if suffix in PERCENT_UNITS:
        normalized = f"{format_amount(amount)}%"
    else:
        currency = CURRENCIES.get(prefix) or CURRENCIES.get(suffix)
        normalized = format_amount(amount)
        if currency is not None:
            normalized = f"{currency}:{normalized}"
    return [Entity(entity_type="number", text=match.group(0).strip(), normalized=normalized)]


DATE_PATTERNS: tuple[tuple["re.Pattern[str]", Handler], ...] = (
    (YEAR_RANGE_RE, _handle_year_range),
    (ISO_DATE_RE, _handle_ymd),
    (CN_DATE_RE, _handle_ymd),
    (MDY_DATE_RE, _handle_named_month),
    (DMY_DATE_RE, _handle_named_month),
    (QUARTER_RE, _handle_quarter),
    (MONTH_YEAR_RE, _handle_named_month),
    (YEAR_RE, _handle_year),
)


def _overlaps(span: tuple[int, int], occupied: Sequence[tuple[int, int]]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in occupied)


def _scan(
    text: str,
    patterns: Iterable[tuple["re.Pattern[str]", Handler]],
    occupied: list[tuple[int, int]],
) -> list[Entity]:
    """Run patterns in priority order, letting an earlier (more specific)
    pattern claim characters so ``2024-03-15`` is one date and not three."""
    found: list[Entity] = []
    for pattern, handler in patterns:
        for match in pattern.finditer(text):
            span = match.span()
            if _overlaps(span, occupied):
                continue
            entities = handler(match)
            if not entities:
                continue
            occupied.append(span)
            found.extend(entities)
    return found


def extract_dates_and_numbers(text: str) -> tuple[Entity, ...]:
    """Dates first, then numbers on the characters dates did not claim -- a bare
    ``2024`` is a year, not the quantity two thousand and twenty-four."""
    occupied: list[tuple[int, int]] = []
    dates = _scan(text, DATE_PATTERNS, occupied)
    numbers = _scan(text, ((NUMBER_RE, _handle_number),), occupied)
    return tuple(dates + numbers)


# --------------------------------------------------------------------------
# proper-noun extraction
# --------------------------------------------------------------------------

_NOUN_TOKEN = r"[A-Z][A-Za-z0-9&.'’\-]*"
PROPER_NOUN_RE = re.compile(
    rf"\b{_NOUN_TOKEN}(?:\s+(?:of|the|de|van|von|der|for|and)\s+{_NOUN_TOKEN}|\s+{_NOUN_TOKEN})*"
)


def _strip_possessive(token: str) -> str:
    """``Acme's`` and ``Acme`` name the same company."""
    for suffix in ("'s", "’s", "'", "’"):
        if len(token) > len(suffix) and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def normalize_name(text: str, aliases: Mapping[str, str] = DEFAULT_ALIASES) -> str:
    """Lowercase, drop leading stopwords and legal suffixes, then apply aliases,
    so ``The Pfizer Inc.`` and ``Pfizer`` compare equal."""
    tokens = [_strip_possessive(token).strip(".,;:'’\"") for token in text.split()]
    tokens = [token for token in tokens if token]
    while tokens and tokens[0].lower() in LEADING_STOPWORDS:
        tokens.pop(0)
    while len(tokens) > 1 and tokens[-1].lower() in LEGAL_SUFFIXES:
        tokens.pop()
    normalized = " ".join(tokens).lower()
    return aliases.get(normalized, normalized)


def dedupe(entities: Iterable[Entity]) -> tuple[Entity, ...]:
    seen: set[tuple[str, str]] = set()
    unique: list[Entity] = []
    for entity in entities:
        key = (entity.entity_type, entity.normalized)
        if key in seen:
            continue
        seen.add(key)
        unique.append(entity)
    return tuple(unique)


class RuleBasedEntityExtractor:
    """Zero-dependency default: regex dates/numbers plus capitalization-based
    proper nouns, which are emitted as the generic ``name`` type because no
    org/person/location distinction is available without a real NER model."""

    def __init__(self, aliases: Mapping[str, str] | None = None) -> None:
        self.aliases = dict(DEFAULT_ALIASES if aliases is None else aliases)

    def extract(self, text: str) -> tuple[Entity, ...]:
        occupied: list[tuple[int, int]] = []
        dates = _scan(text, DATE_PATTERNS, occupied)
        numbers = _scan(text, ((NUMBER_RE, _handle_number),), occupied)
        return dedupe([*dates, *numbers, *self._names(text, occupied)])

    def _names(self, text: str, occupied: Sequence[tuple[int, int]]) -> list[Entity]:
        found: list[Entity] = []
        for match in PROPER_NOUN_RE.finditer(text):
            if _overlaps(match.span(), occupied):
                continue
            normalized = normalize_name(match.group(0), self.aliases)
            if not normalized or normalized in MONTHS:
                continue
            found.append(Entity(entity_type="name", text=match.group(0), normalized=normalized))
        lowered = text.lower()
        for alias, canonical in self.aliases.items():
            if alias in lowered:
                found.append(Entity(entity_type="name", text=alias, normalized=canonical))
        return found


SPACY_LABELS: Mapping[str, EntityType] = {
    "ORG": "organization",
    "PERSON": "person",
    "GPE": "location",
    "LOC": "location",
    "FAC": "location",
    "NORP": "organization",
    "PRODUCT": "product",
    "WORK_OF_ART": "product",
    "EVENT": "product",
}


class SpacyEntityExtractor:
    """Optional upgrade: spaCy NER for the proper-noun types, still using the
    regex path for dates and numbers (spaCy's DATE/MONEY spans are not
    normalized, and those are the entities worth being exact about).

    Requires the ``verification`` extra plus a downloaded spaCy model.
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        nlp: Any = None,
        aliases: Mapping[str, str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.aliases = dict(DEFAULT_ALIASES if aliases is None else aliases)
        self._nlp = nlp

    def _ensure_loaded(self) -> Any:
        if self._nlp is None:
            try:
                spacy = importlib.import_module("spacy")
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "SpacyEntityExtractor requires the optional 'spacy' package."
                ) from exc
            self._nlp = spacy.load(self.model_name)
        return self._nlp

    def extract(self, text: str) -> tuple[Entity, ...]:
        document = self._ensure_loaded()(text)
        names: list[Entity] = []
        for span in document.ents:
            entity_type = SPACY_LABELS.get(str(span.label_))
            if entity_type is None:
                continue
            normalized = normalize_name(str(span.text), self.aliases)
            if not normalized:
                continue
            names.append(Entity(entity_type=entity_type, text=str(span.text), normalized=normalized))
        return dedupe([*extract_dates_and_numbers(text), *names])


# --------------------------------------------------------------------------
# the check itself
# --------------------------------------------------------------------------


REQUIRE_PRESENCE: frozenset[str] = frozenset({"date", "number"})
"""Types where a claim entity missing from the evidence counts as a mismatch on
its own, not just when it conflicts with a same-role value.

Dates and numbers are extracted exactly, so "the claim states a year/amount the
evidence never states" is a real attribution failure. Proper nouns are not: the
default extractor recognises them by capitalization, so ``Revenue rose 8%`` also
yields a "name". Demanding presence there would flag every sentence-initial
common noun, so names only fail when the evidence names something else in the
same role -- which is exactly the counterfactual-swap shape (``Acme`` in the
claim, ``Globex`` in the evidence).

The residual risk -- a claim's common noun conflicting with a real name in the
evidence -- is bounded by the fact that this check only ever runs on pairs NLI
already labelled entailment, so the two texts are already close in meaning. Swap
in ``SpacyEntityExtractor`` to remove it."""


class EntityConsistencyChecker:
    """Compare the entities a claim asserts against the ones its candidate
    supporting evidence actually contains (plan section 4, steps 1-4)."""

    def __init__(
        self,
        extractor: EntityExtractor | None = None,
        require_presence: Iterable[str] = REQUIRE_PRESENCE,
    ) -> None:
        self.extractor = extractor or RuleBasedEntityExtractor()
        self.require_presence = frozenset(require_presence)

    def check(self, claim_text: str, evidence_text: str) -> EntityConsistency:
        claim_entities = self.extractor.extract(claim_text)
        evidence_entities = self.extractor.extract(evidence_text)
        mismatches: list[EntityMismatch] = []
        seen: set[tuple[str, str]] = set()
        for entity in claim_entities:
            comparable = self._comparable_values(entity.entity_type, evidence_entities)
            if self._is_consistent(entity, comparable):
                continue
            if not comparable and entity.entity_type not in self.require_presence:
                continue
            key = (entity.entity_type, entity.normalized)
            if key in seen:
                continue
            seen.add(key)
            mismatches.append(
                EntityMismatch(
                    entity_type=entity.entity_type,
                    claim_text=entity.text,
                    normalized=entity.normalized,
                    evidence_values=tuple(sorted(comparable)),
                )
            )
        return EntityConsistency(consistent=not mismatches, mismatches=tuple(mismatches))

    @staticmethod
    def _is_consistent(entity: Entity, comparable: set[str]) -> bool:
        if entity.normalized in comparable:
            return True
        if entity.entity_type == "number" and ":" not in entity.normalized:
            # An unqualified amount matches a currency-qualified one ("1.2 billion"
            # vs "$1.2 billion"); the reverse stays a mismatch so a swapped
            # currency is still caught.
            return entity.normalized in {value.split(":", 1)[-1] for value in comparable}
        return False

    @staticmethod
    def _comparable_values(
        entity_type: EntityType,
        evidence_entities: Iterable[Entity],
    ) -> set[str]:
        if entity_type in NAME_TYPES:
            return {
                item.normalized for item in evidence_entities if item.entity_type in NAME_TYPES
            }
        return {
            item.normalized for item in evidence_entities if item.entity_type == entity_type
        }
