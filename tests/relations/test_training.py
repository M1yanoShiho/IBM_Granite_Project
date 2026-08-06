"""Tests for the M0 §3.8 training recipe (R013-R015).

Every assertion here exists because the corresponding mistake would produce a trained model and
a plausible Gate 0B number rather than a crash. That is this project's characteristic defect
(f63e905's warm cache, A1's sum-instead-of-max collapse, the MiniCheck label-token lookup), so
the recipe's invariants are guards, not comments.
"""

import pytest

from evidence_rag.cli.gate0b import LABEL_ORDER
from evidence_rag.relations.models import RelationLabel, RelationPair
from evidence_rag.relations.training import (
    BASE_ID2LABEL,
    BASE_LABEL_ORDER,
    BASE_MODEL,
    EMERGENCY_ARM_MODEL,
    N_FOLDS,
    SEEDS,
    Fold,
    FoldFit,
    TrainingExample,
    assert_decontaminated,
    derive_label_order,
    family_key,
    oof_folds,
    page_key,
    require_base_model,
    require_seed,
    run_oof,
    vitaminc_examples,
)


def _example(
    group: str, label: RelationLabel = RelationLabel.SUPPORTS, source: str = "vitaminc"
) -> TrainingExample:
    return TrainingExample(
        premise=f"premise for {group}",
        hypothesis=f"hypothesis for {group}",
        label=label,
        group_keys=(group,),
        source=source,
    )


def _pair(page: str, label: RelationLabel = RelationLabel.SUPPORTS) -> RelationPair:
    return RelationPair(premise="e", hypothesis="c", label=label, group=page)


def _fit_fold_returning(
    *,
    label_order: tuple[str, ...] = BASE_LABEL_ORDER,
    versions: dict[int, str] | None = None,
    n_predictions: int | None = None,
    predicted: RelationLabel | None = None,
):  # type: ignore[no-untyped-def]
    """A fake fitter: perfect out-of-fold predictions unless a test asks for a broken shape."""

    def fit(*, fold: Fold, seed: int, base_model: str) -> FoldFit:
        rows = fold.held_out if n_predictions is None else fold.held_out[:n_predictions]
        version = (versions or {}).get(fold.index, f"local/fold-{fold.index}@{'a' * 16}")
        return FoldFit(
            model_version=version,
            label_order=label_order,
            predictions=tuple(predicted or row.label for row in rows),
        )

    return fit


# --- the pre-registered constants -------------------------------------------------------------


def test_the_three_seeds_are_the_ones_section_5_4_reinstated() -> None:
    """§5.4 abolished the 3-seed ensemble for the zero-training path and reinstated it, for the
    relation model only, if §3.8 is started. Two of the three landing and being reported as
    'the three seeds' is the cheapest way to under-report variance, so the tuple is pinned."""
    assert SEEDS == (13, 42, 73)


def test_five_folds_and_the_two_pre_registered_bases_are_pinned() -> None:
    assert N_FOLDS == 5
    assert BASE_MODEL == "cross-encoder/nli-deberta-v3-base"
    assert EMERGENCY_ARM_MODEL == "cross-encoder/nli-deberta-v3-large"


def test_require_base_model_refuses_a_base_that_is_not_pre_registered() -> None:
    """A3 (§11) spent a whole amendment choosing this base under rules written before any
    candidate was looked at. Swapping it at the call site would undo that silently."""
    with pytest.raises(ValueError, match="not a pre-registered §3.8 base"):
        require_base_model("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")


def test_require_base_model_allows_the_large_emergency_arm() -> None:
    """§3.8's emergency arm is pre-registered, so the code must not block it — but its trigger
    (three seeds finished, gold_supports_recall still under .85) is a fact about results that no
    library function can check. It is allowed here and recorded in the run manifest instead."""
    assert require_base_model(EMERGENCY_ARM_MODEL) == EMERGENCY_ARM_MODEL


def test_require_seed_refuses_a_seed_outside_the_pre_registered_three() -> None:
    with pytest.raises(ValueError, match="not one of the pre-registered seeds"):
        require_seed(7)


# --- label order: the failure mode §11.9 item 4 exists for ------------------------------------


def test_the_base_label_order_is_derived_by_name_and_does_not_start_with_supports() -> None:
    """§11.9 item 4, measured on bp1: id2label = {0: contradiction, 1: entailment, 2: neutral}.
    Both older entries in LABEL_ORDER start with SUPPORTS; this one starts with REFUTES. Copying
    a neighbour positionally swaps SUPPORTS and REFUTES on every edge and raises nothing."""
    assert derive_label_order(BASE_ID2LABEL) == ("REFUTES", "SUPPORTS", "UNKNOWN")
    assert BASE_LABEL_ORDER[0] != "SUPPORTS"


def test_the_derived_order_agrees_with_the_gate_0b_registry() -> None:
    """Two tables now name this checkpoint's classes: cli.gate0b.LABEL_ORDER (which scores the
    gate) and this module (which trains and then verifies the saved head). If they ever drift,
    the model is trained under one reading and scored under another, and every metric stays in
    range. Cross-checked here rather than trusted."""
    assert BASE_LABEL_ORDER == LABEL_ORDER[BASE_MODEL]


def test_derive_label_order_refuses_an_anonymous_head() -> None:
    """`{LABEL_0, LABEL_1, LABEL_2}` is exactly H6's exclusion (§11.2): the order cannot be
    decided from the config, and guessing it does not raise."""
    with pytest.raises(ValueError, match="not a verifiable NLI label"):
        derive_label_order({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})


def test_derive_label_order_refuses_a_two_class_head() -> None:
    """A2 restored the three-class output space; a two-class head has no UNKNOWN logit and
    cannot be certified on 0B-1 at all (§10.6 cost 3)."""
    with pytest.raises(ValueError, match="three classes"):
        derive_label_order({0: "entailment", 1: "contradiction"})


def test_derive_label_order_refuses_duplicate_class_names() -> None:
    """Two positions claiming 'entailment' means one relation label has no position. The tuple
    would still have length three and still zip cleanly against a softmax row."""
    with pytest.raises(ValueError, match="duplicate"):
        derive_label_order({0: "entailment", 1: "entailment", 2: "neutral"})


def test_derive_label_order_refuses_ids_that_are_not_0_1_2() -> None:
    """The order is positional against the logit vector. Ids {0, 1, 3} still produce a
    three-name tuple, and it would be misaligned with the softmax row it labels."""
    with pytest.raises(ValueError, match="0, 1, 2"):
        derive_label_order({0: "entailment", 1: "neutral", 3: "contradiction"})


# --- turning VitaminC into training rows -------------------------------------------------------


def test_vitaminc_examples_carry_the_page_as_a_normalized_group_key() -> None:
    """The parent-page axis has to join across two corpora, so both sides must normalize the
    same way; 'Barack Obama' and 'barack  obama' are one page."""
    examples = vitaminc_examples([_pair("Barack Obama"), _pair("barack  obama")])
    assert examples[0].group_keys == examples[1].group_keys == (page_key("Barack Obama"),)


def test_vitaminc_examples_refuse_the_derived_not_supported_label() -> None:
    """NOT_SUPPORTED is derived from A2 onward, not predicted: a three-class head has no logit
    for it. Training on it needs either a fourth class or a silent remap, and a silent remap is
    how the twin rows would quietly become REFUTES without an amendment."""
    with pytest.raises(ValueError, match="NOT_SUPPORTED"):
        vitaminc_examples([_pair("page", RelationLabel.NOT_SUPPORTED)])


def test_assert_decontaminated_refuses_train_and_dev_sharing_a_page() -> None:
    """`vitaminc.decontaminate` removes from train every page kept in dev, so a non-empty
    intersection means the operator exported the raw splits. That trains fine and reports fine
    — and burns dev, which §2.4 and §11.2a reserve as the only legal calibration surface."""
    with pytest.raises(ValueError, match="decontamination"):
        assert_decontaminated(
            train=vitaminc_examples([_pair("shared")]),
            dev=vitaminc_examples([_pair("shared")]),
        )


def test_assert_decontaminated_accepts_disjoint_pages() -> None:
    assert_decontaminated(
        train=vitaminc_examples([_pair("train-only")]),
        dev=vitaminc_examples([_pair("dev-only")]),
    )


# --- 5-fold OOF, grouped by parent page + synthetic family ------------------------------------


def test_every_example_is_held_out_exactly_once() -> None:
    """That property is what makes the concatenated predictions out-of-fold. If a row were held
    out twice the OOF set would double-count it; if never, the report covers fewer rows than the
    denominator says."""
    examples = [_example(f"g{i}") for i in range(20)]
    folds = oof_folds(examples)
    held_out = sorted(index for fold in folds for index in fold.held_out_indices)
    assert held_out == list(range(20))


def test_a_group_is_never_split_across_two_folds() -> None:
    """This is the whole point of grouping. VitaminC's contrastive pairs differ in one changed
    fact, so the same page on both sides of a fold boundary is near-verbatim leakage."""
    examples = [_example("shared") for _ in range(6)] + [_example(f"g{i}") for i in range(10)]
    folds = oof_folds(examples)
    fold_of_shared = {
        fold.index for fold in folds for row in fold.held_out if row.group_keys == ("shared",)
    }
    assert len(fold_of_shared) == 1


def test_a_niah_family_and_a_vitaminc_page_from_one_article_land_in_the_same_fold() -> None:
    """§3.8 groups the WHOLE chain by parent page + synthetic family. A NIAH needle drawn from
    the 'Barack Obama' article and a VitaminC pair from the 'Barack Obama' page are the same
    article; grouping the two corpora independently would put them on opposite sides of a fold
    boundary and call it out-of-fold."""
    niah = TrainingExample(
        premise="p",
        hypothesis="h",
        label=RelationLabel.SUPPORTS,
        group_keys=(page_key("Barack Obama"), family_key("a|b|numeric")),
        source="niah",
    )
    examples = [niah, *vitaminc_examples([_pair("Barack Obama")])]
    examples += [_example(f"filler{i}") for i in range(10)]
    folds = oof_folds(examples)
    joined = {
        fold.index
        for fold in folds
        for index in fold.held_out_indices
        if index in (0, 1)
    }
    assert len(joined) == 1


def test_fold_assignment_does_not_depend_on_the_order_the_examples_arrive_in() -> None:
    """A split that moves when the input is reordered is not a split, it is a random seed with
    extra steps — and the three-seed clause would then be measuring split variance too. Python's
    built-in hash() is salted per process and would do exactly this."""
    examples = [_example(f"g{i % 7}") for i in range(21)]
    forward = oof_folds(examples)
    backward = oof_folds(list(reversed(examples)))

    def signature(folds: tuple[Fold, ...]) -> list[set[str]]:
        return [{key for row in fold.held_out for key in row.group_keys} for fold in folds]

    assert signature(forward) == signature(backward)


def test_oof_folds_refuse_fewer_groups_than_folds() -> None:
    """Four groups over five folds leaves one fold with nothing held out. Its 'model' would then
    be scored on zero rows, and a zero-row fold contributes a silent 0.0 to any average."""
    with pytest.raises(ValueError, match="fewer leakage groups"):
        oof_folds([_example(f"g{i}") for i in range(4)])


def test_oof_folds_refuse_an_empty_chain() -> None:
    with pytest.raises(ValueError, match="no training examples"):
        oof_folds([])


# --- running the five fits ---------------------------------------------------------------------


def test_run_oof_returns_predictions_aligned_with_the_input_examples() -> None:
    examples = [
        _example(f"g{i}", RelationLabel.SUPPORTS if i % 2 else RelationLabel.REFUTES)
        for i in range(15)
    ]
    run = run_oof(
        examples=examples,
        seed=13,
        base_model=BASE_MODEL,
        expected_label_order=BASE_LABEL_ORDER,
        fit_fold=_fit_fold_returning(),
    )
    assert run.predictions == tuple(row.label for row in examples)
    assert len(run.folds) == N_FOLDS


def test_run_oof_refuses_a_fold_whose_head_came_back_in_a_different_order() -> None:
    """A fold that saves its head as (SUPPORTS, REFUTES, UNKNOWN) while the others save
    (REFUTES, SUPPORTS, UNKNOWN) contributes a fifth of the OOF predictions with SUPPORTS and
    REFUTES swapped. Nothing downstream can see it: the counts stay in range."""
    with pytest.raises(ValueError, match="label order"):
        run_oof(
            examples=[_example(f"g{i}") for i in range(15)],
            seed=13,
            base_model=BASE_MODEL,
            expected_label_order=BASE_LABEL_ORDER,
            fit_fold=_fit_fold_returning(label_order=("SUPPORTS", "REFUTES", "UNKNOWN")),
        )


def test_run_oof_refuses_a_prediction_count_that_does_not_match_the_held_out_rows() -> None:
    """Short predictions would otherwise zip silently against the held-out rows and label the
    wrong ones — strict=True is what turns that into an error, and this pins it."""
    with pytest.raises(ValueError, match="predictions"):
        run_oof(
            examples=[_example(f"g{i}") for i in range(15)],
            seed=13,
            base_model=BASE_MODEL,
            expected_label_order=BASE_LABEL_ORDER,
            fit_fold=_fit_fold_returning(n_predictions=1),
        )


def test_run_oof_refuses_two_folds_that_produced_identical_weights() -> None:
    """Five folds train on five different subsets; identical fingerprints mean no training
    happened, or a warm cache served fold 0's checkpoint five times — the f63e905 shape. Both
    yield a complete, plausible OOF report."""
    with pytest.raises(ValueError, match="identical weight fingerprint"):
        run_oof(
            examples=[_example(f"g{i}") for i in range(15)],
            seed=13,
            base_model=BASE_MODEL,
            expected_label_order=BASE_LABEL_ORDER,
            fit_fold=_fit_fold_returning(versions=dict.fromkeys(range(N_FOLDS), "x@" + "b" * 16)),
        )


def test_run_oof_refuses_a_model_version_that_is_not_weight_fingerprinted() -> None:
    """Same rule the edge cache already enforces: a hand-written version lets two checkpoints
    share one cache key (relations.predictor.require_fingerprinted_version)."""
    with pytest.raises(ValueError, match="fingerprinted"):
        run_oof(
            examples=[_example(f"g{i}") for i in range(15)],
            seed=13,
            base_model=BASE_MODEL,
            expected_label_order=BASE_LABEL_ORDER,
            fit_fold=_fit_fold_returning(versions={0: "seed-13-fold-0"}),
        )


def test_run_oof_refuses_a_prediction_outside_the_three_class_space() -> None:
    """A fitter handing back NOT_SUPPORTED means a two-class head was trained; scored on 0B-1 it
    would read three structural failures and one vacuous pass even if perfect (§10.6 cost 3)."""
    with pytest.raises(ValueError, match="three-class"):
        run_oof(
            examples=[_example(f"g{i}") for i in range(15)],
            seed=13,
            base_model=BASE_MODEL,
            expected_label_order=BASE_LABEL_ORDER,
            fit_fold=_fit_fold_returning(predicted=RelationLabel.NOT_SUPPORTED),
        )


def test_run_oof_refuses_a_seed_that_is_not_pre_registered() -> None:
    with pytest.raises(ValueError, match="not one of the pre-registered seeds"):
        run_oof(
            examples=[_example(f"g{i}") for i in range(15)],
            seed=1,
            base_model=BASE_MODEL,
            expected_label_order=BASE_LABEL_ORDER,
            fit_fold=_fit_fold_returning(),
        )
