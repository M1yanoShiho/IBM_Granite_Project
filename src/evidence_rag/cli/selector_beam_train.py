"""Train the frozen three-class Beam Selector recipe, starting with M1 sanity."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from evidence_rag.contracts.models import CandidateSet, Query
from evidence_rag.materializer.selector_beam_data import (
    BeamSelectorCase,
    BeamTrainingExample,
    load_niah_cases,
    load_twowiki_cases,
    training_examples,
    write_example_manifest,
)
from evidence_rag.materializer.selector_beam_split import sha256_file
from evidence_rag.selector.beam_three_class import ThreeClassBeamSelector
from evidence_rag.selector.beam_torch import TorchBeamNetwork, TorchBeamTextScorer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the three-class Beam Selector")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--m0-report", required=True, type=Path)
    parser.add_argument("--niah-manifest", required=True, type=Path)
    parser.add_argument("--niah-candidates", required=True, type=Path)
    parser.add_argument("--niah-assignments", required=True, type=Path)
    parser.add_argument("--twowiki-manifest", required=True, type=Path)
    parser.add_argument("--twowiki-candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir")
    parser.add_argument("--sanity", action="store_true")
    return parser


def _section(config: Mapping[str, object], name: str) -> Mapping[str, object]:
    section = config.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"config has no [{name}] section")
    return section


def _integer(section: Mapping[str, object], name: str) -> int:
    value = section.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"config value {name} must be an integer")
    return value


def _number(section: Mapping[str, object], name: str) -> float:
    value = section.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"config value {name} must be numeric")
    return float(value)


def _text(section: Mapping[str, object], name: str) -> str:
    value = section.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"config value {name} must be non-empty text")
    return value


def _integer_list(section: Mapping[str, object], name: str) -> tuple[int, ...]:
    value = section.get(name)
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise ValueError(f"config value {name} must be an integer list")
    return tuple(value)


def _minimum_class_recall(metrics: Mapping[str, object]) -> float:
    value = metrics.get("class_recall")
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in value)
    ):
        raise ValueError("classification metrics must contain three class recalls")
    return min(float(item) for item in value)


def _balanced_class_weights(
    sources: Sequence[Sequence[BeamTrainingExample]],
) -> tuple[float, float, float]:
    """Balance labels after giving every dataset source equal probability mass."""

    if not sources or any(not source for source in sources):
        raise ValueError("class weighting requires non-empty dataset sources")
    probabilities = [0.0, 0.0, 0.0]
    for source in sources:
        for label in range(3):
            probabilities[label] += (
                sum(int(item.label) == label for item in source) / len(source) / len(sources)
            )
    if any(probability <= 0.0 for probability in probabilities):
        raise ValueError("class weighting requires all three Selector labels")
    weights = [1.0 / math.sqrt(3.0 * probability) for probability in probabilities]
    return weights[0], weights[1], weights[2]


def _json_object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _validate_m0(arguments: argparse.Namespace) -> tuple[str, str]:
    report = _json_object(arguments.m0_report)
    if report.get("status") != "PASS":
        raise ValueError("training requires a passing M0 report")
    model = _section(report, "model")
    if model.get("config_sha256") != sha256_file(arguments.config):
        raise ValueError("Selector config is not the version frozen by M0")
    niah = _section(report, "niah")
    assignments = _section(niah, "assignments")
    train_assignment = _section(assignments, "train")
    if train_assignment.get("sha256") != sha256_file(arguments.niah_assignments):
        raise ValueError("NIAH assignments are not the file frozen by M0")
    niah_datasets = _section(niah, "datasets")
    niah_train = _section(niah_datasets, "train")
    twowiki = _section(report, "twowiki")
    twowiki_train = _section(twowiki, "train")
    if twowiki_train.get("manifest_sha256") != sha256_file(arguments.twowiki_manifest):
        raise ValueError("2Wiki train manifest is not the dataset frozen by M0")
    pools = _section(report, "candidate_pools")
    expected_paths = {
        "niah-train": arguments.niah_candidates,
        "2wiki-train": arguments.twowiki_candidates,
    }
    for name, path in expected_paths.items():
        pool = _section(pools, name)
        if pool.get("sha256") != sha256_file(path):
            raise ValueError(f"{name} candidates are not the pool frozen by M0")
    return (
        _text(niah_train, "dataset_signature"),
        _text(twowiki_train, "dataset_signature"),
    )


def _batches(
    left: Sequence[BeamTrainingExample],
    right: Sequence[BeamTrainingExample],
    *,
    batch_size: int,
    seed: int,
) -> Iterator[tuple[BeamTrainingExample, ...]]:
    if not left or not right:
        raise ValueError("both NIAH and 2Wiki training examples are required")
    generator = random.Random(seed)
    shuffled = [list(left), list(right)]
    for values in shuffled:
        generator.shuffle(values)
    chunks = [
        [tuple(values[start : start + batch_size]) for start in range(0, len(values), batch_size)]
        for values in shuffled
    ]
    steps = max(len(chunks[0]), len(chunks[1]))
    for step in range(steps):
        yield chunks[0][step % len(chunks[0])]
        yield chunks[1][step % len(chunks[1])]


def _loss_for_batch(
    network: TorchBeamNetwork,
    batch: Sequence[BeamTrainingExample],
    *,
    max_length: int,
    class_weights: Any,
) -> tuple[Any, int, int, tuple[int, ...], tuple[int, ...]]:
    encoded = network.encode(
        questions=[item.question for item in batch],
        selected_passages=[item.selected_passages for item in batch],
        candidate_passages=[item.candidate_passage for item in batch],
        max_length=max_length,
    )
    logits = network.logits(encoded, [item.hop for item in batch])
    targets = network.torch.tensor(
        [int(item.label) for item in batch], device=network.device, dtype=network.torch.long
    )
    loss = network.torch.nn.functional.cross_entropy(logits, targets, weight=class_weights)
    predicted = tuple(int(value) for value in logits.argmax(dim=-1).detach().cpu().tolist())
    expected = tuple(int(item.label) for item in batch)
    correct = sum(left == right for left, right in zip(predicted, expected, strict=True))
    return loss, correct, len(batch), predicted, expected


def _classification_metrics(
    network: TorchBeamNetwork,
    examples: Sequence[BeamTrainingExample],
    *,
    batch_size: int,
    max_length: int,
) -> dict[str, object]:
    network.eval()
    predicted_all: list[int] = []
    expected_all: list[int] = []
    with network.torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            encoded = network.encode(
                questions=[item.question for item in batch],
                selected_passages=[item.selected_passages for item in batch],
                candidate_passages=[item.candidate_passage for item in batch],
                max_length=max_length,
            )
            logits = network.logits(encoded, [item.hop for item in batch])
            predicted_all.extend(
                int(value) for value in logits.argmax(dim=-1).detach().cpu().tolist()
            )
            expected_all.extend(int(item.label) for item in batch)
    recalls: list[float] = []
    for label in range(3):
        indices = [index for index, expected in enumerate(expected_all) if expected == label]
        if not indices:
            raise ValueError(f"sanity examples contain no class {label}")
        recalls.append(sum(predicted_all[index] == label for index in indices) / len(indices))
    accuracy = sum(
        predicted == expected
        for predicted, expected in zip(predicted_all, expected_all, strict=True)
    ) / len(expected_all)
    return {
        "accuracy": accuracy,
        "class_recall": recalls,
        "macro_recall": sum(recalls) / len(recalls),
    }


def _candidate_set(case: BeamSelectorCase) -> CandidateSet:
    # Selection correctness needs only the original IDs and ranking.  Reconstructing a contract
    # value here does not expose labels to the Selector.
    from evidence_rag.contracts.models import EvidenceCandidate

    return CandidateSet(
        query_id=case.query_id,
        candidates=tuple(
            EvidenceCandidate(
                evidence_id=item.evidence_id,
                document_id=item.document_id,
                chunk_id=item.evidence_id,
                text=item.text,
                source_uri="frozen-candidate",
                retrieval_score=item.retrieval_score,
                retrieval_rank=item.retrieval_rank,
            )
            for item in case.candidates
        ),
    )


def _sanity_cases(
    niah_cases: Sequence[BeamSelectorCase],
    twowiki_cases: Sequence[BeamSelectorCase],
    *,
    count: int,
    max_selected: int,
) -> tuple[tuple[BeamSelectorCase, ...], tuple[BeamSelectorCase, ...]]:
    """Choose fully observed sanity cases whose success condition is actually attainable."""

    if count < 1 or max_selected < 1:
        raise ValueError("sanity count and max_selected must be positive")

    def candidate_documents(case: BeamSelectorCase) -> set[str]:
        return {candidate.document_id for candidate in case.candidates}

    niah_eligible = tuple(
        case
        for case in sorted(niah_cases, key=lambda item: item.query_id)
        if set(case.required_document_ids) <= candidate_documents(case)
        and case.harmful_document_id in candidate_documents(case)
        and len(set(case.required_document_ids)) <= max_selected
    )
    twowiki_eligible = tuple(
        case
        for case in sorted(twowiki_cases, key=lambda item: item.query_id)
        if 2 <= len(set(case.required_document_ids)) <= max_selected
        and set(case.required_document_ids) <= candidate_documents(case)
    )
    if len(niah_eligible) < count or len(twowiki_eligible) < count:
        raise ValueError("not enough fully labeled and selectable Top-20 cases for the sanity run")
    return niah_eligible[:count], twowiki_eligible[:count]


def _selection_accuracy(
    network: TorchBeamNetwork,
    cases: Sequence[BeamSelectorCase],
    *,
    max_length: int,
    batch_size: int,
    beam_size: int,
    max_selected: int,
    required_threshold: float,
    reject_threshold: float,
) -> float:
    scorer = TorchBeamTextScorer(network, max_length=max_length, batch_size=batch_size)
    selector = ThreeClassBeamSelector(
        scorer,
        beam_size=beam_size,
        required_threshold=required_threshold,
        reject_threshold=reject_threshold,
    )
    correct = 0
    for case in cases:
        if len(set(case.required_document_ids)) > max_selected:
            raise ValueError(
                f"sanity case {case.query_id} requires more evidence than max_selected"
            )
        document_by_evidence = {
            candidate.evidence_id: candidate.document_id for candidate in case.candidates
        }
        selected_ids = {
            item.evidence_id
            for item in selector.select(
                Query(query_id=case.query_id, text=case.question),
                _candidate_set(case),
                max_selected,
            ).items
        }
        selected_documents = {document_by_evidence[evidence_id] for evidence_id in selected_ids}
        required = set(case.required_document_ids)
        harmful = case.harmful_document_id
        correct += int(
            required <= selected_documents
            and (harmful is None or harmful not in selected_documents)
        )
    return 0.0 if not cases else correct / len(cases)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    niah_signature, twowiki_signature = _validate_m0(arguments)
    config = tomllib.loads(arguments.config.read_text(encoding="utf-8"))
    model_config = _section(config, "model")
    training = _section(config, "training")
    selection = _section(config, "selection")
    sanity = _section(config, "sanity")
    declared_seeds = _integer_list(training, "seeds")
    if arguments.seed not in declared_seeds:
        raise ValueError(f"seed {arguments.seed} is not frozen in the config")
    if _text(training, "niah_twowiki_ratio") != "1:1":
        raise ValueError("this runner implements only the frozen 1:1 batch-source ratio")
    if _text(training, "class_weighting") != "inverse-sqrt-frequency-per-source":
        raise ValueError("this runner implements only the frozen class weighting policy")

    random.seed(arguments.seed)
    os.environ["PYTHONHASHSEED"] = str(arguments.seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    niah_cases = load_niah_cases(
        manifest_path=arguments.niah_manifest,
        candidate_path=arguments.niah_candidates,
        assignment_path=arguments.niah_assignments,
        expected_dataset_signature=niah_signature,
    )
    twowiki_cases = load_twowiki_cases(
        manifest_path=arguments.twowiki_manifest,
        candidate_path=arguments.twowiki_candidates,
        expected_dataset_signature=twowiki_signature,
    )
    if arguments.sanity:
        count = _integer(sanity, "questions_per_dataset")
        niah_cases, twowiki_cases = _sanity_cases(
            niah_cases,
            twowiki_cases,
            count=count,
            max_selected=_integer(selection, "max_selected"),
        )
        epochs = _integer(sanity, "epochs")
    else:
        epochs = _integer(training, "epochs")

    hard_negatives = _integer(training, "hard_negatives_per_query")
    niah_examples = training_examples(
        niah_cases, hard_negatives_per_query=hard_negatives, seed=arguments.seed
    )
    twowiki_examples = training_examples(
        twowiki_cases, hard_negatives_per_query=hard_negatives, seed=arguments.seed
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    write_example_manifest(
        arguments.output / "training_examples_manifest.json",
        (*niah_examples, *twowiki_examples),
    )

    network = TorchBeamNetwork(
        model_id=_text(model_config, "model_id"),
        revision=_text(model_config, "revision"),
        device=arguments.device,
        cache_dir=arguments.cache_dir,
        seed=arguments.seed,
    )
    class_weight_values = _balanced_class_weights((niah_examples, twowiki_examples))
    class_weights = network.torch.tensor(
        class_weight_values, device=network.device, dtype=network.torch.float
    )
    optimizer = network.torch.optim.AdamW(
        list(network.parameters()), lr=_number(training, "learning_rate")
    )
    batch_size = _integer(training, "batch_size")
    accumulation = _integer(training, "gradient_accumulation_steps")
    max_length = _integer(model_config, "max_length")
    sanity_minimum = _number(sanity, "minimum_training_accuracy")
    history: list[dict[str, object]] = []
    for epoch in range(epochs):
        network.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        correct = 0
        total = 0
        predicted_all: list[int] = []
        expected_all: list[int] = []
        batches = tuple(
            _batches(
                niah_examples,
                twowiki_examples,
                batch_size=batch_size,
                seed=arguments.seed + epoch,
            )
        )
        for index, batch in enumerate(batches, 1):
            loss, batch_correct, batch_total, predicted, expected = _loss_for_batch(
                network,
                batch,
                max_length=max_length,
                class_weights=class_weights,
            )
            loss_value = float(loss.detach().cpu().item())
            if not math.isfinite(loss_value):
                raise RuntimeError(f"non-finite loss at epoch {epoch + 1}, batch {index}")
            remainder = len(batches) % accumulation
            final_group = remainder > 0 and index > len(batches) - remainder
            group_size = remainder if final_group else accumulation
            (loss / group_size).backward()
            if index % accumulation == 0 or index == len(batches):
                gradient_norm = network.torch.nn.utils.clip_grad_norm_(
                    list(network.parameters()), 1.0
                )
                if not math.isfinite(float(gradient_norm.detach().cpu().item())):
                    raise RuntimeError(f"non-finite gradient at epoch {epoch + 1}, batch {index}")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += loss_value * batch_total
            correct += batch_correct
            total += batch_total
            predicted_all.extend(predicted)
            expected_all.extend(expected)
        recalls = []
        for label in range(3):
            label_total = sum(value == label for value in expected_all)
            label_correct = sum(
                predicted == expected == label
                for predicted, expected in zip(predicted_all, expected_all, strict=True)
            )
            recalls.append(None if label_total == 0 else label_correct / label_total)
        record: dict[str, object] = {
            "epoch": epoch + 1,
            "loss": total_loss / total,
            "online_accuracy": correct / total,
            "online_class_recall": recalls,
        }
        if arguments.sanity:
            evaluation = _classification_metrics(
                network,
                (*niah_examples, *twowiki_examples),
                batch_size=batch_size,
                max_length=max_length,
            )
            record["post_epoch_classification"] = evaluation
            if (
                _number(evaluation, "accuracy") >= sanity_minimum
                and _minimum_class_recall(evaluation) >= sanity_minimum
            ):
                selection_check = _selection_accuracy(
                    network,
                    (*niah_cases, *twowiki_cases),
                    max_length=max_length,
                    batch_size=batch_size,
                    beam_size=_integer(selection, "beam_size"),
                    max_selected=_integer(selection, "max_selected"),
                    required_threshold=_number(sanity, "required_threshold"),
                    reject_threshold=_number(sanity, "reject_threshold"),
                )
                record["post_epoch_selection_accuracy"] = selection_check
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if (
            arguments.sanity
            and "post_epoch_selection_accuracy" in record
            and _number(record, "post_epoch_selection_accuracy") >= sanity_minimum
        ):
            break

    checkpoint = arguments.output / "model.pt"
    network.save(checkpoint)
    final_classification = (
        _classification_metrics(
            network,
            (*niah_examples, *twowiki_examples),
            batch_size=batch_size,
            max_length=max_length,
        )
        if arguments.sanity
        else None
    )
    selection_accuracy = (
        _selection_accuracy(
            network,
            (*niah_cases, *twowiki_cases),
            max_length=max_length,
            batch_size=batch_size,
            beam_size=_integer(selection, "beam_size"),
            max_selected=_integer(selection, "max_selected"),
            required_threshold=_number(sanity, "required_threshold"),
            reject_threshold=_number(sanity, "reject_threshold"),
        )
        if arguments.sanity
        else None
    )
    minimum = sanity_minimum
    passed = not arguments.sanity or (
        final_classification is not None
        and _number(final_classification, "accuracy") >= minimum
        and _minimum_class_recall(final_classification) >= minimum
        and selection_accuracy is not None
        and selection_accuracy >= minimum
    )
    report = {
        "schema_version": "1.0",
        "stage": "M1" if arguments.sanity else "training",
        "status": "PASS" if passed else "FAIL",
        "seed": arguments.seed,
        "questions": {"niah": len(niah_cases), "2wiki": len(twowiki_cases)},
        "examples": {"niah": len(niah_examples), "2wiki": len(twowiki_examples)},
        "epochs": len(history),
        "final_classification": final_classification,
        "selection_accuracy": selection_accuracy,
        "minimum_accuracy": minimum if arguments.sanity else None,
        "deterministic_algorithms": True,
        "class_weights": list(class_weight_values),
        "history": history,
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "inputs": {
            "config_sha256": sha256_file(arguments.config),
            "niah_candidates_sha256": sha256_file(arguments.niah_candidates),
            "niah_assignments_sha256": sha256_file(arguments.niah_assignments),
            "twowiki_candidates_sha256": sha256_file(arguments.twowiki_candidates),
        },
    }
    report_path = arguments.output / ("M1_REPORT.json" if arguments.sanity else "TRAIN_REPORT.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "status": report["status"]}), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
