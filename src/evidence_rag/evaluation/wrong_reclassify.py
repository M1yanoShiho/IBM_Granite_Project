"""Re-classify probe `wrong` cases into matching-artifact tiers (systematic-debugging).

The needle probe marks an extraction WRONG whenever exact-string canonicalization differs from
gold. Many of those are the model being right in different words. `summarize_reclass` buckets WRONG
into the deterministic, recoverable tiers versus genuinely DIFFERENT (real QA error or synonym). The
equivalence primitives now live in `evidence_rag.selector.answer_equivalence`; they are re-exported
here for backwards compatibility. `recoverable_rate` sizes how much needle-gold-recovery a better
answer-matcher buys for free.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from evidence_rag.selector.answer_equivalence import (
    DIFFERENT as DIFFERENT,
)
from evidence_rag.selector.answer_equivalence import (
    EQUAL_AFTER_NORM as EQUAL_AFTER_NORM,
)
from evidence_rag.selector.answer_equivalence import (
    EXTRACTED_IN_GOLD as EXTRACTED_IN_GOLD,
)
from evidence_rag.selector.answer_equivalence import (
    GOLD_IN_EXTRACTED as GOLD_IN_EXTRACTED,
)
from evidence_rag.selector.answer_equivalence import (
    is_sublist as is_sublist,
)
from evidence_rag.selector.answer_equivalence import (
    lenient_equivalent as lenient_equivalent,
)
from evidence_rag.selector.answer_equivalence import (
    normalize_tokens as normalize_tokens,
)
from evidence_rag.selector.answer_equivalence import (
    reclassify,
)


@dataclass(frozen=True)
class ReclassSummary:
    n: int
    equal_after_norm: int
    gold_in_extracted: int
    extracted_in_gold: int
    different: int
    recoverable: int
    recoverable_rate: float | None


def summarize_reclass(rows: Iterable[Mapping[str, object]]) -> ReclassSummary:
    labels = [reclassify(str(row["extracted"]), str(row["gold_value"])) for row in rows]
    n = len(labels)
    equal = labels.count(EQUAL_AFTER_NORM)
    gold_in = labels.count(GOLD_IN_EXTRACTED)
    extracted_in = labels.count(EXTRACTED_IN_GOLD)
    different = labels.count(DIFFERENT)
    recoverable = equal + gold_in + extracted_in
    return ReclassSummary(
        n=n,
        equal_after_norm=equal,
        gold_in_extracted=gold_in,
        extracted_in_gold=extracted_in,
        different=different,
        recoverable=recoverable,
        recoverable_rate=(recoverable / n) if n else None,
    )
