"""CLI: Gate 0B sweep over several zero-training relation checkpoints (design §3.1, §3.2).

All candidate models run in ONE sweep. Testing a single arm cannot separate "this checkpoint is
insufficient" from "no zero-shot model is sufficient", and that distinction is exactly what
decides whether the training path starts.
"""

import argparse
import dataclasses
import importlib
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.relations.gate0b import external_report, task_report
from evidence_rag.relations.models import RelationLabel
from evidence_rag.relations.predictor import NLIRelationPredictor, ScoreFn

# NLI heads order their classes differently and the mapping is per checkpoint. A wrong order
# silently swaps REFUTES and SUPPORTS while every number stays plausible, so unknown ids are
# rejected rather than guessed.
#
# Each entry below was verified against the real checkpoint on 2026-07-30 with
#   AutoConfig.from_pretrained(model_id).id2label
# and the observed output is recorded beside it. Re-verify and update the recorded output before
# adding any new checkpoint — a guess here is not detectable from the numbers.
LABEL_ORDER: dict[str, tuple[str, ...]] = {
    # id2label = {0: 'SUPPORTS', 1: 'REFUTES', 2: 'NOT ENOUGH INFO'}
    "tals/albert-xlarge-vitaminc-mnli": ("SUPPORTS", "REFUTES", "UNKNOWN"),
    # id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli": (
        "SUPPORTS",
        "UNKNOWN",
        "REFUTES",
    ),
}

BATCH_SIZE = 32


def load_score_fn(model_id: str) -> ScoreFn:
    """Load a HF sequence-classification checkpoint as a three-class scorer.

    Seam for testing. Fails loudly on an unverified model id — see LABEL_ORDER.
    """
    if model_id not in LABEL_ORDER:
        raise ValueError(
            f"no verified label order for {model_id!r}; inspect config.id2label and add it "
            "to LABEL_ORDER before running the gate"
        )
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")

    order = LABEL_ORDER[model_id]
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()

    def score(pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        results: list[dict[str, float]] = []
        for start in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[start : start + BATCH_SIZE]
            encoded = tokenizer(
                [premise for premise, _ in batch],
                [hypothesis for _, hypothesis in batch],
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = model(**encoded).logits
            for row in torch.softmax(logits, dim=-1).tolist():
                results.append({name: float(row[index]) for index, name in enumerate(order)})
        return results

    return score


def _read_pairs(path: Path) -> list[dict[str, str]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate 0B sweep")
    parser.add_argument("--task-pairs", required=True, type=Path, help="0B-2 probe jsonl")
    parser.add_argument("--external-pairs", type=Path, help="0B-1 VitaminC official test jsonl")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models", nargs="+", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    task_rows = _read_pairs(arguments.task_pairs)
    external_rows = _read_pairs(arguments.external_pairs) if arguments.external_pairs else []

    models: dict[str, object] = {}
    for model_id in arguments.models:
        predictor = NLIRelationPredictor(
            score_fn=load_score_fn(model_id), model_version=model_id
        )
        task_predictions = predictor.predict(
            [(row["premise"], row["hypothesis"]) for row in task_rows]
        )
        entry: dict[str, object] = {
            "task": dataclasses.asdict(
                task_report(
                    kinds=[row["kind"] for row in task_rows],
                    gold=[RelationLabel(row["label"]) for row in task_rows],
                    predicted=[prediction.label for prediction in task_predictions],
                )
            )
        }
        if external_rows:
            external_predictions = predictor.predict(
                [(row["premise"], row["hypothesis"]) for row in external_rows]
            )
            entry["external"] = dataclasses.asdict(
                external_report(
                    gold=[RelationLabel(row["label"]) for row in external_rows],
                    predicted=[prediction.label for prediction in external_predictions],
                )
            )
        models[model_id] = entry

    payload = {
        "models": models,
        "n_task_pairs": len(task_rows),
        "n_external_pairs": len(external_rows),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
