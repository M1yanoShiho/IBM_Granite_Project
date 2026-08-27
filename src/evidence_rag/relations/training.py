"""The M0 §3.8 training recipe for the relation model (R013-R015).

§3.8 is PRE-REGISTERED and is reproduced here in full, because every constant below is one of
its clauses rather than a tuning knob:

    base `cross-encoder/nli-deberta-v3-base` (A3 / §11); sentence-transformers CrossEncoder
    three-class scaffold; VitaminC main training (revision-family decontaminated, official test
    untouched, deletion list archived) plus NIAH train-split mutation-log pairs for domain
    adaptation; 5-fold OOF over the whole chain grouped by parent page + synthetic family;
    three seeds 13 / 42 / 73; official test runs once only. Hard constraint: the NIAH
    adaptation pairs' parent pages must have ZERO overlap with sealed-600.

The path is unlocked because Gate 0B failed in all nine cells across three zero-training arms,
which is what makes the family-level claim ("no zero-training model suffices") stand.

WHAT THIS MODULE IS AND IS NOT. It holds the recipe's invariants and its data plumbing, and it
is deliberately torch-free so all of it is unit-testable on a laptop. The five actual fits are a
seam (`FoldFitter`), supplied by `evidence_rag.cli.train_relations`, exactly as `cli/gate0b.py`
keeps `load_score_fn` as its seam.

WHY THESE ARE RAISES AND NOT WARNINGS. Each guard below names one way to finish a training run
and get a Gate 0B reading that is in range and wrong: a head saved in a different class order, a
fold that reused another fold's weights, a split that moved because it was keyed on a salted
hash, dev rows in the training chain. None of them makes any downstream number look unusual.
This project has already paid for that failure mode three times (f63e905's warm cache, A1's
sum-versus-max collapse, MiniCheck's label-token lookup), so nothing here is advisory.

THREE GAPS THIS MODULE ONCE CARRIED HAVE BEEN CLOSED BY THE PROTOCOL, not by code choosing for
it: §3.8(a) rules the NIAH twin rows are trained as REFUTES (the flag stays required, so the
choice is still visible in every manifest), §3.8(b) freezes the optimisation hyperparameters,
and §3.8(d) requires that a head whose `id2label` cannot be mapped by name be REFUSED — never
guessed, never re-attached, never fallen back from.

WHAT §3.8 STILL DOES NOT SPECIFY, AND IS THEREFORE NOT DECIDED HERE:

  * whether the 5-fold split is re-drawn per seed. It is NOT: the split here is a deterministic
    function of the data alone, so the three seeds vary training randomness only. A seed-varying
    split would make §5.4's three-seed reading a mixture of initialisation variance and split
    variance, and the protocol reinstates the clause for the model, not for the data.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from evidence_rag.materializer.source_parent import normalize_parent
from evidence_rag.relations.models import (
    PREDICTED_LABELS,
    RelationLabel,
    RelationPair,
)
from evidence_rag.relations.predictor import require_fingerprinted_version

PROTOCOL_VERSION = "g2-proto-5"

# A3 §11.5. The emergency arm is pre-registered too (§3.8, §11.5 裁决): it may be used ONLY
# after all three seeds have finished with `gold_supports_recall` still under .85, and both
# arms must then be reported side by side. That trigger is a fact about results, so no library
# function can enforce it; what code can do is refuse any base that is neither of the two.
BASE_MODEL = "cross-encoder/nli-deberta-v3-base"
EMERGENCY_ARM_MODEL = "cross-encoder/nli-deberta-v3-large"
PRE_REGISTERED_BASES = (BASE_MODEL, EMERGENCY_ARM_MODEL)

# §5.4's three-seed clause, reinstated for the training path only.
SEEDS = (13, 42, 73)
N_FOLDS = 5

# Measured on bp1 2026-08-06 by `scripts/a3_preflight.py` (§11.9 item 4) and registered in
# `cli/gate0b.py::LABEL_ORDER`. Kept here as well because the trainer must verify that the
# FINE-TUNED checkpoint still saves this exact head: `AutoConfig(..., num_labels=3)` is capable
# of replacing a named id2label with LABEL_0/1/2, and sentence-transformers builds its config
# that way. A head that came back re-ordered rather than anonymised would not be visible in any
# metric — it just relabels every edge.
BASE_ID2LABEL: Mapping[int, str] = {0: "contradiction", 1: "entailment", 2: "neutral"}

# By NAME, never by position. §11.9 records that this base is the first LABEL_ORDER entry whose
# position 0 is not SUPPORTS; the two older entries start with SUPPORTS, and copying either
# positionally swaps SUPPORTS and REFUTES on every edge without raising.
RELATION_BY_NLI_NAME: Mapping[str, str] = {
    "entailment": RelationLabel.SUPPORTS.value,
    "contradiction": RelationLabel.REFUTES.value,
    "neutral": RelationLabel.UNKNOWN.value,
}

VITAMINC_SOURCE = "vitaminc"
NIAH_SOURCE = "niah"

_PAGE_PREFIX = "page:"
_FAMILY_PREFIX = "family:"


def derive_label_order(id2label: Mapping[int, str]) -> tuple[str, ...]:
    """Turn a checkpoint's `id2label` into a `LABEL_ORDER` tuple, by name.

    Every rejection below corresponds to a config that would otherwise yield a three-element
    tuple that zips cleanly against a softmax row and labels it wrongly.
    """
    if len(id2label) != 3:
        raise ValueError(
            f"id2label {dict(id2label)} does not describe three classes. A2 (§10.2) restored the"
            " three-class output space; a two-class head cannot be certified on Gate 0B-1 at all"
            " (§10.6 cost 3), and a four-class one has no place in this recipe."
        )
    if set(id2label) != {0, 1, 2}:
        raise ValueError(
            f"id2label keys {sorted(id2label)} are not exactly 0, 1, 2. The order is POSITIONAL"
            " against the logit vector, so any other id set produces a three-name tuple that is"
            " misaligned with the row it labels."
        )
    order: list[str] = []
    for position in (0, 1, 2):
        name = str(id2label[position]).strip().lower()
        if name not in RELATION_BY_NLI_NAME:
            raise ValueError(
                f"id2label[{position}] = {id2label[position]!r} is not a verifiable NLI label"
                f" name (expected one of {sorted(RELATION_BY_NLI_NAME)}). This is H6 of §11.2:"
                " a head labelled LABEL_0/1/2 has an order that cannot be decided from its"
                " config, and guessing it does not raise — it relabels every edge."
            )
        order.append(RELATION_BY_NLI_NAME[name])
    if len(set(order)) != 3:
        raise ValueError(
            f"id2label {dict(id2label)} gives two positions the same duplicate class: {order}."
            " One relation label then has no position, and the tuple is still length three."
        )
    return tuple(order)


BASE_LABEL_ORDER = derive_label_order(BASE_ID2LABEL)


def require_base_model(model_id: str) -> str:
    if model_id not in PRE_REGISTERED_BASES:
        raise ValueError(
            f"{model_id!r} is not a pre-registered §3.8 base. Only {BASE_MODEL!r} (A3 §11.5) and"
            f" the emergency arm {EMERGENCY_ARM_MODEL!r} are permitted, and the emergency arm"
            " only once all three seeds have finished with gold_supports_recall still under .85,"
            " with both arms reported side by side. Any other base needs its own amendment."
        )
    return model_id


def require_seed(seed: int) -> int:
    if seed not in SEEDS:
        raise ValueError(
            f"seed {seed} is not one of the pre-registered seeds {SEEDS} (§5.4, reinstated for"
            " the training path by §3.8). Reporting a different seed as one of the three, or"
            " reporting two of three, is the cheapest possible way to under-state variance."
        )
    return seed


def page_key(title: str) -> str:
    """Parent-page leakage key, normalised the same way on both sides of the chain.

    VitaminC pages and dpr-w100 article titles are both Wikipedia titles with inconsistent
    casing and whitespace. Comparing them raw would report no shared article for two spellings
    of one — an audit that passes because it cannot see.
    """
    return f"{_PAGE_PREFIX}{normalize_parent(title)}"


def family_key(family: str) -> str:
    """Synthetic-family leakage key: `(gold_value, replacement_value, string_class)` (§4)."""
    return f"{_FAMILY_PREFIX}{family}"


@dataclass(frozen=True)
class Hyperparameters:
    """PRE-REGISTERED by M0 §3.8(b). Not defaults — frozen values.

    §3.8(b) also records that no legal tuning surface exists for them: dev is reserved by §2.4
    and §11.2a, and this path trains on train only. There is therefore no run that could
    legitimately produce a reason to change any number below, which is why the CLI exposes no
    flag for them and they are written into every manifest instead.
    """

    optimizer: str = "adamw"
    schedule: str = "linear_decay"
    epochs: int = 2
    learning_rate: float = 2e-5
    batch_size: int = 32
    warmup_ratio: float = 0.06
    max_length: int = 256
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    bf16: bool = True


DEFAULT_HYPERPARAMETERS = Hyperparameters()


@dataclass(frozen=True)
class TrainingExample:
    """One row of the training chain.

    `group_keys` is the set of leakage keys this row belongs to — a VitaminC row carries its
    page, a NIAH row carries its parent page AND its synthetic family. They are never features.
    """

    premise: str
    hypothesis: str
    label: RelationLabel
    group_keys: tuple[str, ...]
    source: str

    def __post_init__(self) -> None:
        if self.label not in PREDICTED_LABELS:
            raise ValueError(
                f"{self.label} is not in the three-class output space"
                f" {[label.value for label in PREDICTED_LABELS]}, so a CrossEncoder head has no"
                " logit for it. NOT_SUPPORTED in particular is DERIVED from A2 onward, not"
                " predicted (relations/models.py): training on it needs either a fourth class or"
                " a silent remap, and a silent remap is how the probe's twin rows would become"
                " REFUTES without an amendment."
            )
        if not self.group_keys:
            raise ValueError(
                "a training row with no leakage group key cannot be assigned to a fold without"
                " being treated as its own group, which is the assumption §3.8's grouping exists"
                " to refuse."
            )


def vitaminc_examples(pairs: Sequence[RelationPair]) -> tuple[TrainingExample, ...]:
    """Adapt decontaminated VitaminC pairs. `RelationPair.group` is the page (see vitaminc.py)."""
    return tuple(
        TrainingExample(
            premise=pair.premise,
            hypothesis=pair.hypothesis,
            label=pair.label,
            group_keys=(page_key(pair.group),),
            source=VITAMINC_SOURCE,
        )
        for pair in pairs
    )


def assert_decontaminated(
    *, train: Sequence[TrainingExample], dev: Sequence[TrainingExample]
) -> None:
    """Refuse train and dev sharing a page.

    `vitaminc.decontaminate` removes from train every page that survives in dev, so a non-empty
    intersection can only mean the raw official splits were exported. That trains perfectly well
    and reports perfectly well. What it destroys is dev: §2.4 makes VitaminC official dev the
    one legal surface for freezing a threshold, and §11.2a makes it the one legal surface for a
    base-selection measurement, both on the strength of it being disjoint from everything else.
    """
    shared = sorted(_keys(train) & _keys(dev))
    if shared:
        raise ValueError(
            f"{len(shared)} page(s) appear in BOTH the training chain and the dev split"
            f" (e.g. {shared[:5]}), so revision-family decontamination did not run. Export the"
            " splits with `evidence_rag.cli.export_vitaminc`, which removes them, rather than"
            " loading tals/vitaminc directly. Training on dev burns the only calibration surface"
            " §2.4 and §11.2a permit, and nothing downstream would report it."
        )


def _keys(examples: Sequence[TrainingExample]) -> set[str]:
    return {key for example in examples for key in example.group_keys}


@dataclass(frozen=True)
class Fold:
    index: int
    train: tuple[TrainingExample, ...]
    held_out: tuple[TrainingExample, ...]
    held_out_indices: tuple[int, ...]


def _leakage_components(
    examples: Sequence[TrainingExample],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Connected components over the leakage keys, as (root key, row indices).

    §3.8 groups the whole chain "by parent page + synthetic family" — one grouping over two key
    kinds, not two independent groupings. A NIAH needle drawn from an article that VitaminC also
    covers shares that article with the VitaminC rows, so the two must move together; grouping
    each corpus on its own key would place them in different folds and still call the result
    out-of-fold. Components can only merge groups, never split them, so this reading is the
    conservative one in the only direction that matters.

    The root is the lexicographically smallest key in the component, which makes the result a
    function of the data alone. Python's built-in `hash()` is salted per process and would make
    the split differ between two runs of the same command.
    """
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != root:
            parent[key], key = root, parent[key]
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root

    for example in examples:
        for key in example.group_keys:
            parent.setdefault(key, key)
        for key in example.group_keys[1:]:
            union(example.group_keys[0], key)

    members: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        members.setdefault(find(example.group_keys[0]), []).append(index)
    return tuple((root, tuple(indices)) for root, indices in sorted(members.items()))


def oof_folds(
    examples: Sequence[TrainingExample], *, n_folds: int = N_FOLDS
) -> tuple[Fold, ...]:
    """Grouped 5-fold out-of-fold split over the whole chain (§3.8).

    Assignment is greedy largest-component-first into the currently smallest fold, with ties
    broken by key and then by fold index. That makes it deterministic and independent of the
    order the rows arrive in — a split that moves when the input is reordered is a random seed
    with extra steps, and the three-seed clause would then be reporting split variance too.
    """
    if not examples:
        raise ValueError("no training examples: an empty chain cannot be split into folds")
    components = _leakage_components(examples)
    if len(components) < n_folds:
        raise ValueError(
            f"{len(components)} leakage group(s) is fewer leakage groups than the {n_folds}"
            " pre-registered folds, so at least one fold would hold out nothing. A fold scored"
            " on zero rows contributes a silent 0.0 to any average over folds rather than an"
            " error."
        )
    sizes = [0] * n_folds
    buckets: list[list[int]] = [[] for _ in range(n_folds)]
    for _root, indices in sorted(components, key=lambda item: (-len(item[1]), item[0])):
        target = min(range(n_folds), key=lambda index: (sizes[index], index))
        buckets[target].extend(indices)
        sizes[target] += len(indices)

    folds: list[Fold] = []
    for index, bucket in enumerate(buckets):
        held_out_indices = tuple(sorted(bucket))
        if not held_out_indices:
            raise ValueError(f"fold {index} holds out no rows; the split is degenerate")
        held_out = frozenset(held_out_indices)
        folds.append(
            Fold(
                index=index,
                train=tuple(row for i, row in enumerate(examples) if i not in held_out),
                held_out=tuple(examples[i] for i in held_out_indices),
                held_out_indices=held_out_indices,
            )
        )
    return tuple(folds)


@dataclass(frozen=True)
class FoldFit:
    """What one fitted fold hands back.

    `label_order` is read from the SAVED checkpoint, not from the base: verifying the base and
    then trusting the artefact would miss exactly the case this field exists for.
    """

    model_version: str
    label_order: tuple[str, ...]
    predictions: tuple[RelationLabel, ...]


class FoldFitter(Protocol):
    def __call__(self, *, fold: Fold, seed: int, base_model: str) -> FoldFit: ...


@dataclass(frozen=True)
class FoldOutcome:
    index: int
    model_version: str
    n_train: int
    n_held_out: int


@dataclass(frozen=True)
class OofRun:
    seed: int
    base_model: str
    label_order: tuple[str, ...]
    folds: tuple[FoldOutcome, ...]
    predictions: tuple[RelationLabel, ...]
    hyperparameters: Hyperparameters = DEFAULT_HYPERPARAMETERS


def run_oof(
    *,
    examples: Sequence[TrainingExample],
    seed: int,
    base_model: str,
    expected_label_order: tuple[str, ...],
    fit_fold: FoldFitter,
    n_folds: int = N_FOLDS,
    hyperparameters: Hyperparameters = DEFAULT_HYPERPARAMETERS,
) -> OofRun:
    """Fit every fold and reassemble out-of-fold predictions in the input's own order."""
    require_seed(seed)
    require_base_model(base_model)
    folds = oof_folds(examples, n_folds=n_folds)

    predictions: list[RelationLabel | None] = [None] * len(examples)
    outcomes: list[FoldOutcome] = []
    versions: dict[str, int] = {}
    for fold in folds:
        fit = fit_fold(fold=fold, seed=seed, base_model=base_model)
        if fit.label_order != expected_label_order:
            raise ValueError(
                f"fold {fold.index} saved its head with label order {fit.label_order}, but the"
                f" base registers {expected_label_order}. One fold's fifth of the out-of-fold"
                " predictions would carry two classes swapped, and every count downstream would"
                " still be in range. Re-verify the saved config.id2label; do NOT reorder the"
                " tuple to match."
            )
        require_fingerprinted_version(fit.model_version)
        if fit.model_version in versions:
            raise ValueError(
                f"folds {versions[fit.model_version]} and {fold.index} report an identical"
                f" weight fingerprint ({fit.model_version}). Five folds train on five different"
                " subsets, so identical weights mean no fitting happened or a cached checkpoint"
                " was served twice — the f63e905 shape. Either way the out-of-fold report would"
                " be complete and meaningless."
            )
        versions[fit.model_version] = fold.index
        if len(fit.predictions) != len(fold.held_out):
            raise ValueError(
                f"fold {fold.index} returned {len(fit.predictions)} predictions for"
                f" {len(fold.held_out)} held-out rows. Zipped without this check the surplus"
                " rows would take another row's label."
            )
        for label in fit.predictions:
            if label not in PREDICTED_LABELS:
                raise ValueError(
                    f"fold {fold.index} predicted {label}, which is outside the three-class"
                    " output space A2 restored. A natively-binary head cannot be certified on"
                    " Gate 0B-1 (§10.6 cost 3), and this recipe trains a three-class one."
                )
        for index, label in zip(fold.held_out_indices, fit.predictions, strict=True):
            predictions[index] = label
        outcomes.append(
            FoldOutcome(
                index=fold.index,
                model_version=fit.model_version,
                n_train=len(fold.train),
                n_held_out=len(fold.held_out),
            )
        )

    missing = [index for index, label in enumerate(predictions) if label is None]
    if missing:
        raise ValueError(
            f"{len(missing)} row(s) received no out-of-fold prediction (e.g. {missing[:5]});"
            " the folds do not cover the chain."
        )
    return OofRun(
        seed=seed,
        base_model=base_model,
        label_order=expected_label_order,
        folds=tuple(outcomes),
        predictions=tuple(label for label in predictions if label is not None),
        hyperparameters=hyperparameters,
    )


def registry_line(model_id: str, label_order: tuple[str, ...]) -> str:
    """The exact line to paste into `cli/gate0b.py::LABEL_ORDER` to score a trained checkpoint.

    §11.10 item 7 requires that a base id and its `id2label` reach the code through the registry
    rather than being hardcoded at a call site. A trained checkpoint's id is a run-specific path
    that cannot be pre-registered, so the trainer emits the line and a human adds it — the
    registry stays the single place a label order is written down, and adding one is a visible,
    reviewable act rather than a CLI flag that accepts any order.
    """
    return f'    "{model_id}": {label_order!r},'
