"""VitaminC adapter + revision-family decontamination (Graph 2.0 design §3.0, §3.6).

Fields verified 2026-07-30 against HF `tals/vitaminc`: claim / evidence / label / page /
revision_type, labels in {SUPPORTS, REFUTES, NOT ENOUGH INFO}, splits 371k / 63.1k / 55.2k.

Why this dataset and not ContractNLI: VitaminC's contrastive structure — near-identical evidence
pairs differing in one changed fact — is the same shape as the counterfactual injector's twins.
ContractNLI is 17 fixed hypotheses over 607 NDAs, a legal-domain document-level task with no
structural similarity to "does this passage support that the answer is X", and v1 already
measured -0.134 transfer from it.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from evidence_rag.materializer.source_parent import normalize_parent
from evidence_rag.relations.models import RelationLabel, RelationPair

LABELS = {
    "SUPPORTS": RelationLabel.SUPPORTS,
    "REFUTES": RelationLabel.REFUTES,
    "NOT ENOUGH INFO": RelationLabel.UNKNOWN,
}


def to_pairs(rows: Iterable[Mapping[str, str]]) -> tuple[RelationPair, ...]:
    """Adapt official rows. Evidence is the premise, claim is the hypothesis — reversing them
    would invert every REFUTES edge while still producing plausible-looking numbers."""
    pairs: list[RelationPair] = []
    for row in rows:
        raw = row["label"]
        if raw not in LABELS:
            raise ValueError(f"unknown VitaminC label: {raw!r}")
        pairs.append(
            RelationPair(
                premise=row["evidence"],
                hypothesis=row["claim"],
                label=LABELS[raw],
                group=row["page"],
            )
        )
    return tuple(pairs)


@dataclass(frozen=True)
class DecontaminatedSplits:
    train: tuple[RelationPair, ...]
    dev: tuple[RelationPair, ...]
    test: tuple[RelationPair, ...]
    removed_train_groups: tuple[str, ...]
    removed_dev_groups: tuple[str, ...]


def decontaminate(
    *,
    train: Sequence[RelationPair],
    dev: Sequence[RelationPair],
    test: Sequence[RelationPair],
) -> DecontaminatedSplits:
    """Test-preserving revision-family decontamination.

    Official test is never moved or filtered. Dev loses any page that appears in test; train then
    loses any page appearing in test or in dev.

    "Same page" is `normalize_parent` — casefold plus whitespace collapse — NOT raw string
    equality, because that is what `training.page_key` uses and therefore what every leakage
    audit downstream will compare. Matching raw here removed nothing while the audit found a
    collision, which is how `XXx-COLON- Return of Xander Cage` (train) and
    `XXX-COLON- Return of Xander Cage` (dev) both survived a pass that was supposed to separate
    them; measured on bp1 2026-08-08, one page on this axis and zero against test. An exporter
    that partitions more finely than the audit reads cannot remove what the audit will find.

    Note on `kept_dev` below: using the kept dev rather than the raw dev makes no difference to
    the result, because the pages that differ between them are exactly `dev & test`, which
    `test_keys` already forbids. It is written this way only because "train must avoid every
    page that survives downstream" is the property being expressed. Do not add a test claiming
    the two formulations differ — they are provably equivalent (the argument survives
    normalisation unchanged, since both sides are compared through the same key), and such a
    test passes under both, which is worse than no test.

    `removed_*_groups` stay RAW spellings: they are an audit trail of what left the data, and
    reporting the normalised key would hide the very difference that caused the removal.
    """
    test_keys = {normalize_parent(pair.group) for pair in test}
    removed_dev = sorted(
        {pair.group for pair in dev if normalize_parent(pair.group) in test_keys}
    )
    kept_dev = tuple(pair for pair in dev if normalize_parent(pair.group) not in test_keys)
    forbidden_for_train = test_keys | {normalize_parent(pair.group) for pair in kept_dev}
    removed_train = sorted(
        {pair.group for pair in train if normalize_parent(pair.group) in forbidden_for_train}
    )
    kept_train = tuple(
        pair for pair in train if normalize_parent(pair.group) not in forbidden_for_train
    )
    return DecontaminatedSplits(
        train=kept_train,
        dev=kept_dev,
        test=tuple(test),
        removed_train_groups=tuple(removed_train),
        removed_dev_groups=tuple(removed_dev),
    )
