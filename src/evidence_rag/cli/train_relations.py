"""CLI: R013-R015, the M0 §3.8 relation-model fine-tuning run (one seed per invocation).

One seed per job on purpose. §5.4's three-seed clause is reinstated for this path, and three
seeds submitted as three jobs leave three manifests, three log files and three sets of
checkpoints — a run that silently produced two of three would otherwise be reported as "the
three seeds" with nothing on disk to contradict it.

    python -m evidence_rag.cli.train_relations \
      --vitaminc-train data/gate0b/vitaminc_train.jsonl \
      --vitaminc-dev data/gate0b/vitaminc_dev.jsonl \
      --decontamination-log data/gate0b/vitaminc_decontamination.json \
      --seed 13 --output-dir runs/r013/seed-13

WHAT RUNS TODAY AND WHAT DOES NOT. The VitaminC half runs. The NIAH domain-adaptation half is
implemented and REFUSES to run until it is given a sealed-600 parent-page set that the pairs
actually clear (§3.8's hard constraint; sealed-600 is not built yet, §8 item 2). There is no
flag to skip the check. Omitting the NIAH flags entirely runs the VitaminC half alone, which is
legitimate and is recorded as such in the manifest; there is no way to run the NIAH half with
the constraint switched off.

WHAT THIS DELIBERATELY DOES NOT COMPUTE. Gate 0B numbers. The out-of-fold predictions here are
over TRAINING data, and `relations.gate0b.external_report` would happily turn them into the five
frozen 0B-1 threshold names on the wrong data. 0B-1 is VitaminC official test, run once only
(§3.8), and it is scored by `evidence_rag.cli.gate0b`. This CLI writes per-row predictions and
counts and stops there.

MODES. `--plan-only` builds the whole chain, the folds and the manifest without importing torch,
so the data path can be checked on a login node before a GPU is spent. `--scaffold-check-only`
constructs the CrossEncoder and reports what the construction did to the head's `id2label` —
§11.9 item 6 verified that the scaffold ACCEPTS the checkpoint, but not that it preserves the
named three-class head that item 4 measured, and those are different facts.
"""

import argparse
import dataclasses
import importlib
import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.materializer.provenance import read_provenance
from evidence_rag.materializer.source_parent import read_parent_index
from evidence_rag.relations.models import RelationLabel, RelationPair
from evidence_rag.relations.niah_adaptation import (
    PERMITTED_TWIN_LABELS,
    DevEvaluationQueries,
    SealedCorpus,
    build_niah_examples,
    load_dev_queries,
    load_sealed_parents,
    partition_dev_overlap,
)
from evidence_rag.relations.predictor import (
    NLIRelationPredictor,
    fingerprinted_version,
    weight_fingerprint,
)
from evidence_rag.relations.training import (
    BASE_LABEL_ORDER,
    BASE_MODEL,
    N_FOLDS,
    PRE_REGISTERED_BASES,
    PROTOCOL_VERSION,
    Fold,
    FoldFit,
    FoldFitter,
    Hyperparameters,
    OofRun,
    TrainingExample,
    assert_decontaminated,
    derive_label_order,
    oof_folds,
    registry_line,
    require_base_model,
    require_seed,
    run_oof,
    vitaminc_examples,
)

NIAH_FLAGS = (
    "niah_dev_manifest",
    "niah_manifest",
    "niah_parents",
    "niah_provenance",
    "niah_twin_label",
    "sealed_dir",
)


def _read_pairs(path: Path) -> tuple[RelationPair, ...]:
    """Read an `export_vitaminc` pairs file."""
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return tuple(
        RelationPair(
            premise=row["premise"],
            hypothesis=row["hypothesis"],
            label=RelationLabel(row["label"]),
            group=row["group"],
        )
        for row in rows
    )


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _check_against_decontamination_log(
    *, log_path: Path, n_train: int, n_dev: int
) -> Mapping[str, Any]:
    """Cross-check the loaded files against the log `export_vitaminc` wrote beside them.

    This is the guard against the single most likely operator slip on this path: pointing
    `--vitaminc-train` at `vitaminc_test.jsonl`. Every downstream step would accept it — the
    schema is identical — and the run would train on the official test set that §3.8 says is
    touched exactly once, at the end, by a different command. The counts are the cheapest
    evidence that these two files came out of the same decontamination pass.
    """
    log = json.loads(log_path.read_text(encoding="utf-8"))
    mismatches = [
        f"{name}: log says {log[name]}, file has {actual}"
        for name, actual in (("n_train", n_train), ("n_dev", n_dev))
        if log[name] != actual
    ]
    if mismatches:
        raise ValueError(
            f"the pairs files do not match {log_path}: {'; '.join(mismatches)}. They are"
            " therefore not the files that revision-family decontamination produced. The most"
            " likely cause is a path slip onto vitaminc_test.jsonl, which has the same schema and"
            " would train silently on the split §3.8 reserves for a single final run."
        )
    return cast(Mapping[str, Any], log)


def _load_niah_examples(
    arguments: argparse.Namespace,
) -> tuple[
    tuple[TrainingExample, ...],
    SealedCorpus | None,
    DevEvaluationQueries | None,
    tuple[str, ...],
]:
    """Build the domain-adaptation half, or return nothing if it was not asked for."""
    supplied = [name for name in NIAH_FLAGS if getattr(arguments, name) is not None]
    if not supplied:
        return (), None, None, ()
    incomplete = _incomplete_niah_flags(arguments)
    if incomplete:
        raise ValueError(incomplete)
    sealed = load_sealed_parents(arguments.sealed_dir)
    dev_queries = load_dev_queries(arguments.niah_dev_manifest)
    bundle = JsonlDatasetAdapter.load(arguments.niah_manifest)
    parent_index = read_parent_index(arguments.niah_parents)
    kept, excluded = partition_dev_overlap(
        tuple(read_provenance(arguments.niah_provenance)), dev_queries
    )
    examples = build_niah_examples(
        records=kept,
        question_by_query={query.query_id: query.text for query in bundle.queries},
        text_by_document={
            document.document_id: document.text for document in bundle.documents
        },
        # The raw mapping, not `ParentIndex.parent_of`: its fallback would hand back the
        # document_id for an unresolved document, and a synthetic parent cannot collide with a
        # sealed title. See relations/niah_adaptation.py.
        parent_by_document=parent_index.parent_by_document,
        sealed=sealed,
        dev_queries=dev_queries,
        twin_label=RelationLabel(arguments.niah_twin_label),
    )
    return examples, sealed, dev_queries, excluded


def load_fold_fitter(
    *, base_model: str, output_dir: Path, hyperparameters: Hyperparameters
) -> FoldFitter:
    """Build the real sentence-transformers fitter. Seam: tests replace this whole function.

    Everything torch-shaped lives below this line, exactly as `cli/gate0b.py` keeps its model
    loading behind `load_score_fn`, so `relations/training.py` stays importable and testable
    without a GPU.

    THE PREDICTION PATH REUSES `NLIRelationPredictor` RATHER THAN CALLING argmax. The frozen
    §2.4 tie-break resolves UNKNOWN -> REFUTES -> SUPPORTS and must never land on SUPPORTS;
    `numpy.argmax` takes the first maximum, which under this base's order
    (REFUTES, SUPPORTS, UNKNOWN) resolves a SUPPORTS/UNKNOWN tie the wrong way. Reusing the
    predictor also means the out-of-fold labels come out of the same code the gate scores with.
    """
    transformers = importlib.import_module("transformers")
    torch = importlib.import_module("torch")
    cross_encoder = importlib.import_module("sentence_transformers.cross_encoder")

    def fit(*, fold: Fold, seed: int, base_model: str = base_model) -> FoldFit:
        transformers.set_seed(seed)
        model = _construct_scaffold(cross_encoder, base_model, hyperparameters)
        _train(model, fold=fold, seed=seed, hyperparameters=hyperparameters)

        fold_dir = output_dir / f"fold-{fold.index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        _save(model, fold_dir)

        # From the SAVED artefact, not from the object in memory: the checkpoint on disk is what
        # `cli/gate0b.py` will later load, and a config that changed during save is exactly the
        # kind of difference no metric can show.
        saved = transformers.AutoConfig.from_pretrained(str(fold_dir))
        label_order = derive_label_order(
            {int(key): str(value) for key, value in saved.id2label.items()}
        )

        inner = model.model if hasattr(model, "model") else model
        version = fingerprinted_version(
            f"{base_model}#seed-{seed}#fold-{fold.index}",
            weight_fingerprint(_weight_buffers(inner)),
        )
        predictor = NLIRelationPredictor(
            score_fn=_score_fn(model, torch, label_order), model_version=version
        )
        rows = [(row.premise, row.hypothesis) for row in fold.held_out]
        predictions = predictor.predict(rows)
        return FoldFit(
            model_version=version,
            label_order=label_order,
            predictions=tuple(prediction.label for prediction in predictions),
        )

    return fit


def _construct_scaffold(
    cross_encoder: Any, base_model: str, hyperparameters: Hyperparameters
) -> Any:
    """`CrossEncoder(num_labels=3)`, then check what the construction did to the head's names.

    §11.9 item 6 verified only that the scaffold accepts this checkpoint. `num_labels` reaches
    `PretrainedConfig`, which replaces a named `id2label` with LABEL_0/1/2 in some
    configurations — and §11.6 rests on the base already carrying a semantically correct
    three-class head that does NOT need re-initialising. If the names are gone, that premise no
    longer holds for the object being trained, so this stops rather than re-attaching names we
    would then be asserting rather than reading.
    """
    model = cross_encoder.CrossEncoder(
        base_model, num_labels=3, max_length=hyperparameters.max_length
    )
    inner = model.model if hasattr(model, "model") else model
    id2label = {int(key): str(value) for key, value in inner.config.id2label.items()}
    try:
        derive_label_order(id2label)
    except ValueError as error:
        raise ValueError(
            f"CrossEncoder({base_model!r}, num_labels=3) produced a head whose id2label is"
            f" {id2label}, which cannot be mapped by name: {error}. §11.9 item 4 measured"
            " {0: contradiction, 1: entailment, 2: neutral} on the checkpoint itself, so the"
            " names were lost in construction. Do NOT re-attach them here — a re-attached order"
            " is asserted rather than read, and if the head was re-initialised as well then"
            " §11.6's 'no need to re-initialise' premise has failed and that is a protocol"
            " question, not a code one."
        ) from error
    return model


def _train(
    model: Any, *, fold: Fold, seed: int, hyperparameters: Hyperparameters
) -> None:
    """Fit one fold, on whichever sentence-transformers training API is installed.

    The extra pins `sentence-transformers>=3,<6` and the two majors do not share a training
    API. Which one ran is printed rather than swallowed, the same discipline `cli/gate0b.py`
    applies to fast-versus-slow tokenizers: two routes that produce different numbers must not
    be indistinguishable in the log.
    """
    label_index = {label: position for position, label in enumerate(BASE_LABEL_ORDER)}
    premises = [row.premise for row in fold.train]
    hypotheses = [row.hypothesis for row in fold.train]
    labels = [label_index[row.label.value] for row in fold.train]

    trainer_module = _optional_module("sentence_transformers.cross_encoder.trainer")
    if trainer_module is not None:
        print(f"[train_relations] fold {fold.index}: CrossEncoderTrainer API", flush=True)
        datasets = importlib.import_module("datasets")
        losses = importlib.import_module("sentence_transformers.cross_encoder.losses")
        arguments_module = importlib.import_module(
            "sentence_transformers.cross_encoder.training_args"
        )
        trainer_module.CrossEncoderTrainer(
            model=model,
            args=arguments_module.CrossEncoderTrainingArguments(
                output_dir=f"build/train_relations/seed-{seed}/fold-{fold.index}",
                num_train_epochs=hyperparameters.epochs,
                per_device_train_batch_size=hyperparameters.batch_size,
                learning_rate=hyperparameters.learning_rate,
                warmup_ratio=hyperparameters.warmup_ratio,
                weight_decay=hyperparameters.weight_decay,
                max_grad_norm=hyperparameters.max_grad_norm,
                lr_scheduler_type=hyperparameters.schedule.replace("_decay", ""),
                bf16=hyperparameters.bf16,
                seed=seed,
                save_strategy="no",
                report_to=[],
            ),
            train_dataset=datasets.Dataset.from_dict(
                {"premise": premises, "hypothesis": hypotheses, "label": labels}
            ),
            loss=losses.CrossEntropyLoss(model),
        ).train()
        return

    if hyperparameters.bf16:
        raise RuntimeError(
            "§3.8(b) freezes bf16, and the installed sentence-transformers is a 3.x whose"
            " CrossEncoder.fit exposes fp16 autocast (use_amp) but no bf16 switch. Substituting"
            " fp16 would run a different recipe than the one on record while every log line and"
            " every metric stayed in range, so this stops instead. Install"
            " sentence-transformers >= 4, which has CrossEncoderTrainer and a bf16 flag."
        )
    print(f"[train_relations] fold {fold.index}: CrossEncoder.fit API", flush=True)
    sentence_transformers = importlib.import_module("sentence_transformers")
    torch_data = importlib.import_module("torch.utils.data")
    samples = [
        sentence_transformers.InputExample(texts=[premise, hypothesis], label=label)
        for premise, hypothesis, label in zip(premises, hypotheses, labels, strict=True)
    ]
    loader = torch_data.DataLoader(
        samples, shuffle=True, batch_size=hyperparameters.batch_size
    )
    steps = max(1, len(loader) * hyperparameters.epochs)
    model.fit(
        train_dataloader=loader,
        epochs=hyperparameters.epochs,
        warmup_steps=int(steps * hyperparameters.warmup_ratio),
        optimizer_params={"lr": hyperparameters.learning_rate},
        weight_decay=hyperparameters.weight_decay,
        max_grad_norm=hyperparameters.max_grad_norm,
        show_progress_bar=False,
    )


def _optional_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _save(model: Any, path: Path) -> None:
    saver = getattr(model, "save_pretrained", None) or getattr(model, "save", None)
    if saver is None:  # pragma: no cover - depends on the installed sentence-transformers
        raise RuntimeError(
            "the CrossEncoder object exposes neither save_pretrained nor save; the installed"
            " sentence-transformers is outside the >=3,<6 pin this recipe was written against."
        )
    saver(str(path))


def _weight_buffers(model: Any) -> Any:
    for name, tensor in model.state_dict().items():
        spec = f"{name}|{tuple(tensor.shape)}|{tensor.dtype}"
        yield spec, tensor.detach().cpu().contiguous().numpy().tobytes()


def _score_fn(model: Any, torch: Any, label_order: tuple[str, ...]) -> Any:
    def score(pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        raw = model.predict(list(pairs), convert_to_numpy=True, show_progress_bar=False)
        probabilities = torch.softmax(torch.as_tensor(raw).float(), dim=-1).tolist()
        return [
            {name: float(value) for name, value in zip(label_order, row, strict=True)}
            for row in probabilities
        ]

    return score


def _label_counts(examples: Sequence[TrainingExample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.label.value] = counts.get(example.label.value, 0) + 1
    return dict(sorted(counts.items()))


def _manifest(
    *,
    arguments: argparse.Namespace,
    chain: Sequence[TrainingExample],
    folds: Sequence[Fold],
    n_vitaminc: int,
    n_niah: int,
    sealed: SealedCorpus | None,
    dev_queries: DevEvaluationQueries | None,
    excluded_dev_overlap: tuple[str, ...],
    decontamination: Mapping[str, Any],
    hyperparameters: Hyperparameters,
    run: OofRun | None,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "recipe": "M0 §3.8 (base set by A3 §11.5)",
        "base_model": arguments.base_model,
        "seed": arguments.seed,
        "n_folds": arguments.n_folds,
        "hyperparameters": dataclasses.asdict(hyperparameters),
        "hyperparameters_note": (
            "Pre-registered and frozen by M0 §3.8(b). No legal tuning surface exists: dev is"
            " reserved by §2.4 and §11.2a, and this path trains on train only."
        ),
        "label_order": list(BASE_LABEL_ORDER),
        "gate0b_registry_line": registry_line(
            str(arguments.output_dir), BASE_LABEL_ORDER
        ),
        "chain": {
            "n_examples": len(chain),
            "n_vitaminc": n_vitaminc,
            "n_niah": n_niah,
            "n_leakage_groups": len({key for row in chain for key in row.group_keys}),
            "labels": _label_counts(chain),
            "fold_sizes": [len(fold.held_out) for fold in folds],
        },
        "vitaminc": {
            "train_sha256": _sha256(arguments.vitaminc_train),
            "dev_sha256": _sha256(arguments.vitaminc_dev),
            "decontamination_log": dict(decontamination),
        },
        "niah_domain_adaptation": (
            {
                "status": "enforced",
                "twin_label": arguments.niah_twin_label,
                "twin_label_ruling": "M0 §3.8(a)",
                "sealed_dir": sealed.source,
                "sealed_parent_pages_sha256": sealed.parent_pages_sha256,
                "sealed_n_parent_pages": sealed.n_parent_pages,
                "dev_eval_manifest": dev_queries.source,
                "dev_eval_query_ids_sha256": dev_queries.query_ids_sha256,
                "dev_eval_n_queries": dev_queries.n_queries,
                "n_families_excluded_dev_overlap": len(excluded_dev_overlap),
                "dev_overlap_ruling": (
                    "2026-08-09 (A4): dev-overlapping families excluded; the guard compares"
                    " query-id sets and never reads a manifest's split label"
                ),
            }
            if sealed is not None and dev_queries is not None
            else {
                "status": "not run",
                "reason": (
                    "§3.8 requires zero parent-page overlap with sealed-600, which does not"
                    " exist yet (§8 item 2). This run trained on the VitaminC half only."
                ),
            }
        ),
        "is_smoke_run": arguments.max_examples is not None,
        "max_examples": arguments.max_examples,
        "folds": [dataclasses.asdict(outcome) for outcome in (run.folds if run else ())],
        "is_plan_only": run is None,
    }


# Every Path-typed flag is either an input that must already exist or an output this run
# creates. `_missing_input_paths` checks the first group and ignores the second; the test
# `test_no_path_flag_escapes_the_input_output_split` fails if a new Path flag joins neither, so
# the split cannot drift away from the parser it describes.
INPUT_PATHS = frozenset(
    {
        "vitaminc_train",
        "vitaminc_dev",
        "decontamination_log",
        "niah_manifest",
        "niah_provenance",
        "niah_parents",
        "niah_dev_manifest",
        "sealed_dir",
    }
)
OUTPUT_PATHS = frozenset({"output_dir"})


def _incomplete_niah_flags(arguments: argparse.Namespace) -> str | None:
    """The all-or-nothing check on the six domain-adaptation flags, or None if it passes.

    Hoisted so `main` can run it BEFORE the path check: the shape of the request is more
    actionable than the state of the disk. Told "these three files are absent" first, an
    operator fixes the paths, resubmits, and only then learns a flag was missing -- two queue
    waits for one mistake, which is exactly what the path check exists to stop.
    """

    if not any(getattr(arguments, name) is not None for name in NIAH_FLAGS):
        return None
    missing = [name for name in NIAH_FLAGS if getattr(arguments, name) is None]
    if not missing:
        return None
    return (
        f"the NIAH domain-adaptation half needs all of {sorted(NIAH_FLAGS)};"
        f" missing {sorted(missing)}. --sealed-dir in particular is not optional:"
        " §3.8's hard constraint is zero parent-page overlap with sealed-600, and a run"
        " without it is indistinguishable afterwards from one that passed the check."
        " --niah-dev-manifest is likewise not optional (ruling 2026-08-09 / A4): the"
        " adaptation pool shares queries with the dev evaluation run, and the excluded"
        " families are only auditable if the dev set is named here."
    )


def _missing_input_paths(arguments: argparse.Namespace) -> tuple[str, ...]:
    """Every declared input that is not on disk -- all of them, not the first one.

    Three submissions in this series died because a path named on the command line did not
    exist, and 18322821 waited seventeen hours in the queue to be told about one file. Stopping
    at the first would make a second absent file cost a second wait, which is the whole reason
    this is reported as a list and runnable before sbatch (`--check-paths-only`).
    """

    return tuple(
        f"--{dest.replace('_', '-')} {getattr(arguments, dest)}"
        for dest in sorted(INPUT_PATHS)
        if getattr(arguments, dest, None) is not None
        and not Path(getattr(arguments, dest)).exists()
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="R013-R015: M0 §3.8 relation-model training")
    parser.add_argument("--vitaminc-train", required=True, type=Path)
    parser.add_argument("--vitaminc-dev", required=True, type=Path)
    parser.add_argument(
        "--decontamination-log",
        required=True,
        type=Path,
        help="the JSON `export_vitaminc --removed-log` wrote; cross-checked against the pairs",
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", default=BASE_MODEL, choices=PRE_REGISTERED_BASES)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    # No --epochs / --learning-rate / --batch-size. §3.8(b) freezes them and records that no
    # legal tuning surface exists (dev is reserved by §2.4 and §11.2a; this path trains on
    # train only), so a flag here could only ever be used to drift off the recipe.
    parser.add_argument(
        "--max-examples",
        type=int,
        help="smoke runs only; recorded in the manifest as is_smoke_run",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="build the chain, the folds and the manifest without importing torch",
    )
    parser.add_argument(
        "--scaffold-check-only",
        action="store_true",
        help="construct the CrossEncoder and report the head's id2label, then stop",
    )
    parser.add_argument(
        "--check-paths-only",
        action="store_true",
        help="stat every input path and stop. Run it on the LOGIN NODE before sbatch: an "
        "absent file is decidable in a second there, and costs a queue wait from inside a job",
    )
    parser.add_argument("--niah-manifest", type=Path)
    parser.add_argument("--niah-provenance", type=Path)
    parser.add_argument("--niah-parents", type=Path, help="source_parent sidecar for NIAH train")
    parser.add_argument(
        "--niah-dev-manifest",
        type=Path,
        help=(
            "manifest.json of the dev evaluation run the adaptation set must stay disjoint"
            " from (ruling 2026-08-09 / A4): its query-id set drives partition_dev_overlap,"
            " and the guard compares ID sets, never split labels"
        ),
    )
    parser.add_argument(
        "--sealed-dir",
        type=Path,
        help="sealed-600 build directory; its frozen split fingerprint is what §3.8's "
        "zero-overlap check reads. Not optional once any --niah-* flag is given.",
    )
    parser.add_argument(
        "--niah-twin-label",
        choices=[label.value for label in PERMITTED_TWIN_LABELS],
        help="three-class target for the mutation-log twin rows. Ruled REFUTES by §3.8(a); "
        "still NO DEFAULT, so the ruling stays visible in every manifest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    require_seed(arguments.seed)
    require_base_model(arguments.base_model)

    if arguments.scaffold_check_only:
        # Deliberately before the path check: this mode constructs the model and touches none
        # of the data, so requiring the corpus to be present would only make it harder to run.
        return _scaffold_check(arguments.base_model)

    incomplete = _incomplete_niah_flags(arguments)
    if incomplete:
        raise ValueError(incomplete)

    missing = _missing_input_paths(arguments)
    if missing:
        raise ValueError(
            "these input paths do not exist:\n  "
            + "\n  ".join(missing)
            + "\n(--check-paths-only runs this same check on a login node, before sbatch)"
        )
    if arguments.check_paths_only:
        print(json.dumps({"check_paths_only": True, "inputs_present": sorted(INPUT_PATHS)}))
        return 0

    train_pairs = _read_pairs(arguments.vitaminc_train)
    dev_pairs = _read_pairs(arguments.vitaminc_dev)
    decontamination = _check_against_decontamination_log(
        log_path=arguments.decontamination_log,
        n_train=len(train_pairs),
        n_dev=len(dev_pairs),
    )
    vitaminc = vitaminc_examples(train_pairs)
    # Dev never enters the chain: §2.4 and §11.2a reserve VitaminC official dev as the one legal
    # surface for freezing a threshold and for a base-selection measurement. It is loaded only
    # so this guard can run.
    assert_decontaminated(train=vitaminc, dev=vitaminc_examples(dev_pairs))

    niah, sealed, dev_queries, excluded_dev_overlap = _load_niah_examples(arguments)
    chain: tuple[TrainingExample, ...] = (*vitaminc, *niah)
    if arguments.max_examples is not None:
        chain = chain[: arguments.max_examples]
    folds = oof_folds(chain, n_folds=arguments.n_folds)
    hyperparameters = Hyperparameters()  # §3.8(b), frozen

    run: OofRun | None = None
    if not arguments.plan_only:
        run = run_oof(
            examples=chain,
            seed=arguments.seed,
            base_model=arguments.base_model,
            expected_label_order=BASE_LABEL_ORDER,
            fit_fold=load_fold_fitter(
                base_model=arguments.base_model,
                output_dir=arguments.output_dir,
                hyperparameters=hyperparameters,
            ),
            n_folds=arguments.n_folds,
            hyperparameters=hyperparameters,
        )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(
        arguments=arguments,
        chain=chain,
        folds=folds,
        n_vitaminc=len(vitaminc),
        n_niah=len(niah),
        sealed=sealed,
        dev_queries=dev_queries,
        excluded_dev_overlap=excluded_dev_overlap,
        decontamination=decontamination,
        hyperparameters=hyperparameters,
        run=run,
    )
    (arguments.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    if run is not None:
        _write_oof(arguments.output_dir / "oof_predictions.jsonl", chain, folds, run)
    print(json.dumps(manifest, sort_keys=True))
    return 0


def _write_oof(
    path: Path, chain: Sequence[TrainingExample], folds: Sequence[Fold], run: OofRun
) -> None:
    """Per-row out-of-fold predictions (G-PQ §3.7: rescoring must stay a CPU-level operation).

    These are predictions over TRAINING data. They are a training diagnostic and must never be
    presented as a Gate 0B reading — 0B-1 is VitaminC official test, run once, by
    `evidence_rag.cli.gate0b`.
    """
    fold_of_row = {
        index: fold.index for fold in folds for index in fold.held_out_indices
    }
    path.write_text(
        "".join(
            json.dumps(
                {
                    "fold": fold_of_row[index],
                    "gold": row.label.value,
                    "group_keys": list(row.group_keys),
                    "predicted": prediction.value,
                    "source": row.source,
                },
                sort_keys=True,
            )
            + "\n"
            for index, (row, prediction) in enumerate(
                zip(chain, run.predictions, strict=True)
            )
        ),
        encoding="utf-8",
    )


def _scaffold_check(base_model: str) -> int:
    """§11.9 item 6, one step further: does the scaffold KEEP the head item 4 measured?"""
    cross_encoder = importlib.import_module("sentence_transformers.cross_encoder")
    model = _construct_scaffold(cross_encoder, base_model, Hyperparameters())
    inner = model.model if hasattr(model, "model") else model
    id2label = {int(key): str(value) for key, value in inner.config.id2label.items()}
    order = derive_label_order(id2label)
    print(json.dumps({"base_model": base_model, "id2label": id2label, "label_order": order}))
    if order != BASE_LABEL_ORDER:
        raise ValueError(
            f"scaffold head order {order} differs from the registered {BASE_LABEL_ORDER}"
            " (§11.9 item 4, cli/gate0b.py::LABEL_ORDER). The checkpoint changed or the record"
            " was wrong; resolve it in the protocol before training."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
