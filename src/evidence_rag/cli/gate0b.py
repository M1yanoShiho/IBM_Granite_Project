"""CLI: Gate 0B sweep over several zero-training relation checkpoints (design §3.1, §3.2).

All candidate models run in ONE sweep. Testing a single arm cannot separate "this checkpoint is
insufficient" from "no zero-shot model is sufficient", and that distinction is exactly what
decides whether the training path starts.

Under A2 (g2-proto-3) a three-class head is scored as three classes and nothing is collapsed on
this path. Both tiers therefore compose again: `--external-pairs` (0B-1) reads VitaminC's
three-class official gold and evaluates its five frozen thresholds at their original
calibration, while `--task-pairs` (0B-2) derives its two classes inside `task_report` by asking
"is this SUPPORTS?". Pairs files from either side of A1 parse and score identically — the 0B-2
metrics never read the gold column (see `relations/task_probe.py`).

A natively-binary checkpoint (MiniCheck-FT5) can be swept, but it produces NO 0B-1 reading: with
no third class it cannot be certified against thresholds defined on the three-class split
(M0 §10.6 cost 3). Run it with `--external-pairs` omitted rather than reading its 0B-1 row as a
result.
"""

import argparse
import dataclasses
import importlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from evidence_rag.relations.gate0b import external_report, task_report
from evidence_rag.relations.minicheck import (
    VERIFIED_BINARY_PROTOCOLS,
    build_input,
    protocol_for,
    resolve_label_token_ids,
    to_scores,
)
from evidence_rag.relations.models import RelationLabel, RelationPrediction
from evidence_rag.relations.predictor import (
    NLIRelationPredictor,
    ScoreFn,
    fingerprinted_version,
    weight_fingerprint,
)

# NLI heads order their classes differently and the mapping is per checkpoint. A wrong order
# silently swaps REFUTES and SUPPORTS while every number stays plausible, so unknown ids are
# rejected rather than guessed.
#
# Each entry below was verified against the real checkpoint on 2026-07-30 with
#   AutoConfig.from_pretrained(model_id).id2label
# and the observed output is recorded beside it. Re-verify and update the recorded output before
# adding any new checkpoint — a guess here is not detectable from the numbers.
#
# Neither A1 nor A2 retired this table. A2 (g2-proto-3) restored the three-class output space, so
# these names now reach the predictor and the 0B-1 report directly instead of being collapsed
# first — which puts a wrong order back where it is visible, in the predicted label. Under A1 it
# survived only as a wrong `confidence` on an otherwise-correct edge. A natively-binary checkpoint
# (MiniCheck-FT5, M0 §9.8) needs no entry here — it needs its own two-class scorer, added as a
# separate path rather than by inventing a three-name order for it.
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


def _load_tokenizer(transformers: Any, model_id: str) -> tuple[Any, str]:
    """Return (tokenizer, variant).

    Some sentencepiece checkpoints ship no tokenizer.json, so transformers converts the slow
    tokenizer on the fly, and that conversion currently fails for ALBERT
    (convert_slow_tokenizer hits vocab_file=None). The slow tokenizer reads the same
    sentencepiece model and is a safe fallback, but WHICH one was used is returned and
    recorded rather than swallowed: fast and slow tokenizers can differ on edge cases, and
    this output feeds a go/no-go decision.
    """
    try:
        return transformers.AutoTokenizer.from_pretrained(model_id), "fast"
    except Exception as fast_error:
        try:
            return transformers.AutoTokenizer.from_pretrained(model_id, use_fast=False), "slow"
        except Exception as slow_error:
            # AutoTokenizer silently falls back to the FAST class when the slow one cannot be
            # imported, so use_fast=False is a no-op then and both attempts fail identically.
            # That signature is worth naming: it means a missing backend, not a broken
            # checkpoint, and the two look nothing alike to fix.
            same_failure = repr(fast_error) == repr(slow_error)
            hint = (
                " both attempts failed identically, which means use_fast=False did not select a"
                " different class — the slow tokenizer is unavailable, usually because"
                " 'sentencepiece' is not installed"
                if same_failure
                else ""
            )
            raise RuntimeError(
                f"could not load a tokenizer for {model_id!r};"
                f" fast failed with {fast_error!r} and slow with {slow_error!r}.{hint}"
            ) from slow_error


def _weight_buffers(model: Any) -> Iterator[tuple[str, bytes]]:
    """Adapt a torch model's parameters to the torch-free `weight_fingerprint` input.

    Name, shape and dtype ride in the spec string and the values ride in the bytes, so two
    fine-tuning seeds — identical in every field except the numbers — cannot collide.
    """
    for name, tensor in model.state_dict().items():
        spec = f"{name}|{tuple(tensor.shape)}|{tensor.dtype}"
        yield spec, tensor.detach().cpu().contiguous().numpy().tobytes()


def load_score_fn(model_id: str) -> tuple[ScoreFn, str, str]:
    """Load a checkpoint as a BINARY scorer (A1 §9.1), by whichever route it was verified on.

    Two routes exist because the arms are not the same shape of model, and neither can be made
    to stand in for the other:

      * a three-class sequence-classification head, registered in `LABEL_ORDER` and reduced by
        `_collapse_to_binary` (albert, DeBERTa);
      * a natively-binary checkpoint, registered in `relations.minicheck` with its verified
        prompt format and label token ids (MiniCheck-FT5, M0 §3.1 / §9.8).

    Returns (scorer, tokenizer_variant, model_version). `model_version` is derived from the
    checkpoint weights, never from the id alone: the id is a name two different checkpoints
    can share, and it lands on every edge and every dump row.

    Seam for testing. Fails loudly on a model id verified in neither registry.
    """
    if model_id in LABEL_ORDER:
        return _load_three_class_score_fn(model_id)
    if model_id in VERIFIED_BINARY_PROTOCOLS:
        return _load_binary_score_fn(model_id)
    raise ValueError(
        f"{model_id!r} is verified in neither checkpoint registry: no verified label order "
        "(three-class head — inspect config.id2label and add it to LABEL_ORDER) and no "
        "verified binary protocol (natively-binary head — record its prompt format and label "
        "token ids in relations.minicheck.VERIFIED_BINARY_PROTOCOLS). Add it to the ONE that "
        "matches its architecture; a three-name order for a model with no classes to order is "
        "a fabrication, not a workaround."
    )


def _load_three_class_score_fn(model_id: str) -> tuple[ScoreFn, str, str]:
    """Load a HF sequence-classification checkpoint as a BINARY scorer (A1 §9.1).

    A1 collapsed here, at the checkpoint adapter. A2 (§10.2) moved the collapse to the consumers,
    so all three class probabilities leave this function intact. That is the whole mechanism by
    which 0B-1 became runnable again: its five frozen thresholds are defined on the three-class
    split, and collapsing before the predictor is exactly what made them structurally
    unevaluable under A1 (§9.11). The consumers that need two classes — `task_report` and
    `RelationGraph` — both derive them by asking "is this SUPPORTS?", so no explicit collapse
    step exists anywhere any more.
    """
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")

    order = LABEL_ORDER[model_id]
    tokenizer, tokenizer_variant = _load_tokenizer(transformers, model_id)
    # use_safetensors=True, not left to resolution: this project pins torch < 2.6, and
    # transformers refuses to torch.load a `.bin` under that pin (CVE-2025-32434). A repo
    # shipping BOTH formats resolved to the `.bin` on the cluster even with safetensors
    # already cached, so the load died on a message about a CVE rather than about this
    # checkpoint, and it died after the download rather than before. Demanding safetensors
    # makes the safe path the only path and turns "this repo ships no safetensors" into
    # what the error says. Do NOT "fix" a failure here by relaxing this — the fix is either
    # a checkpoint with safetensors or a protocol decision to move off the pinned torch.
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_id, use_safetensors=True
    )
    model.eval()

    # Before .to(device): the parameters are still on CPU, so this costs no PCIe transfer.
    model_version = fingerprinted_version(model_id, weight_fingerprint(_weight_buffers(model)))

    # Without this the job holds a GPU and runs on CPU anyway. ALBERT-xlarge is far heavier than
    # its 59M parameter count suggests — 24 layers share one weight set, so the compute is that
    # of a 24-layer hidden-2048 model — and on CPU it simply looks like a hang.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[gate0b] {model_version} on {device} (tokenizer: {tokenizer_variant})", flush=True)

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
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits
            for row in torch.softmax(logits.float().cpu(), dim=-1).tolist():
                results.append(
                    {name: float(value) for name, value in zip(order, row, strict=True)}
                )
            if start % (BATCH_SIZE * 20) == 0:
                print(f"[gate0b] {model_id} {start}/{len(pairs)}", flush=True)
        return results

    return score, tokenizer_variant, model_version


def _load_binary_score_fn(model_id: str) -> tuple[ScoreFn, str, str]:
    """Load a natively-binary checkpoint as a scorer, with no collapse in the path.

    Three things differ from the three-class route, and each of them is silent if wrong:

      1. `AutoModelForSeq2SeqLM`, not `AutoModelForSequenceClassification` — this checkpoint is a
         `T5ForConditionalGeneration` and the classification loader cannot open it at all. That
         one is loud; the other two are not.
      2. ONE sequence built by `build_input`, not a (premise, hypothesis) pair. The tokenizer
         would accept a pair argument here and encode something the checkpoint was never tuned
         on.
      3. A single decoder step, reading the two label columns resolved from the tokenizer. The
         model emits a full vocabulary distribution and any two columns of it would softmax to a
         plausible-looking pair.

    `max_length` comes from the protocol (2048) rather than the tokenizer default. The probe's
    premises are single ~100-word passages, so nothing truncates either way — but the value is
    part of what was verified, and it will start to matter the first time this scorer sees a
    real pool.
    """
    protocol = protocol_for(model_id)
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")

    tokenizer, tokenizer_variant = _load_tokenizer(transformers, model_id)
    # Before the weights load: a tokenizer that disagrees with the recorded ids means the wrong
    # checkpoint, and there is no reason to spend a download finding that out afterwards.
    negative_id, positive_id = resolve_label_token_ids(protocol, tokenizer.encode)
    # use_safetensors=True, not left to resolution: this project pins torch < 2.6, and
    # transformers refuses to torch.load a `.bin` under that pin (CVE-2025-32434). A repo
    # shipping BOTH formats resolved to the `.bin` on the cluster even with safetensors
    # already cached, so the load died on a message about a CVE rather than about this
    # checkpoint, and it died after the download rather than before. Demanding safetensors
    # makes the safe path the only path and turns "this repo ships no safetensors" into
    # what the error says. Do NOT "fix" a failure here by relaxing this — the fix is either
    # a checkpoint with safetensors or a protocol decision to move off the pinned torch.
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_id, use_safetensors=True)
    model.eval()

    model_version = fingerprinted_version(model_id, weight_fingerprint(_weight_buffers(model)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(
        f"[gate0b] {model_version} on {device} (tokenizer: {tokenizer_variant}, "
        f"binary head, label ids {negative_id}/{positive_id})",
        flush=True,
    )

    def score(pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        results: list[dict[str, float]] = []
        for start in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[start : start + BATCH_SIZE]
            encoded = tokenizer(
                [build_input(protocol, premise, hypothesis) for premise, hypothesis in batch],
                max_length=protocol.max_input_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            # One decode step from the start token: the checkpoint answers with its first
            # generated token, so the whole prediction lives at position 0.
            decoder_input_ids = torch.zeros(
                (encoded["input_ids"].size(0), 1), dtype=torch.long, device=device
            )
            with torch.no_grad():
                logits = model(**encoded, decoder_input_ids=decoder_input_ids).logits
            label_logits = logits.squeeze(1)[:, [negative_id, positive_id]].float().cpu()
            for row in torch.softmax(label_logits, dim=-1).tolist():
                results.append(to_scores(float(row[1])))
            if start % (BATCH_SIZE * 20) == 0:
                print(f"[gate0b] {model_id} {start}/{len(pairs)}", flush=True)
        return results

    return score, tokenizer_variant, model_version


def _read_pairs(path: Path) -> list[dict[str, str]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


class _ScoreRecorder:
    """Wraps a scorer and keeps the raw class probabilities of its most recent call.

    A RelationPrediction carries only the winning class's probability, and re-running the scorer
    to recover the other two would double GPU time on a 55k-pair tier, so they are captured in
    flight. The predictor calls the scorer once per tier with the whole list.
    """

    def __init__(self, score_fn: ScoreFn) -> None:
        self._score_fn = score_fn
        self.scores: list[Mapping[str, float]] = []

    def __call__(self, pairs: Sequence[tuple[str, str]]) -> Sequence[Mapping[str, float]]:
        self.scores = list(self._score_fn(pairs))
        return self.scores


def _dump_rows(
    *,
    model_id: str,
    tier: str,
    rows: Sequence[Mapping[str, str]],
    predictions: Sequence[RelationPrediction],
    scores: Sequence[Mapping[str, float]],
) -> list[dict[str, Any]]:
    """One record per scored pair.

    `kind` is the breakdown the dump exists for. External rows have no kind and say so with null
    rather than dropping the key, so every line of the file shares one schema.

    `probabilities` is written from the score keys the scorer actually produced — three columns
    for a three-class head after A2, two for a natively-binary checkpoint — rather than from a
    fixed label tuple. That keeps the two arm shapes distinguishable in the dump and makes it
    impossible to fabricate a REFUTES or UNKNOWN column for a model that has neither. The pre-A1
    dumps on which R012/R012b are recomputed (§9.10a) are three-class files and round-trip
    unchanged.
    """
    dumped: list[dict[str, Any]] = []
    for row, prediction, scored in zip(rows, predictions, scores, strict=True):
        record: dict[str, Any] = {
            "model_id": model_id,
            "tier": tier,
            "kind": row.get("kind"),
            "gold": row["label"],
            "predicted": prediction.label.value,
            "probabilities": {name: float(value) for name, value in scored.items()},
            "premise_hash": prediction.premise_hash,
            "hypothesis_hash": prediction.hypothesis_hash,
        }
        if "query_id" in row:
            record["query_id"] = row["query_id"]
        dumped.append(record)
    return dumped


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate 0B sweep")
    parser.add_argument("--task-pairs", required=True, type=Path, help="0B-2 probe jsonl")
    parser.add_argument("--external-pairs", type=Path, help="0B-1 VitaminC official test jsonl")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--dump",
        type=Path,
        help="optional jsonl of per-pair predictions, one line per (model, pair). Diagnostic "
        "only: the --output aggregate is byte-identical with and without it.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    task_rows = _read_pairs(arguments.task_pairs)
    external_rows = _read_pairs(arguments.external_pairs) if arguments.external_pairs else []

    dump_path: Path | None = arguments.dump
    dumped: list[dict[str, Any]] = []

    models: dict[str, object] = {}
    for model_id in arguments.models:
        score_fn, tokenizer_variant, model_version = load_score_fn(model_id)
        recorder = _ScoreRecorder(score_fn) if dump_path is not None else None
        predictor = NLIRelationPredictor(
            score_fn=score_fn if recorder is None else recorder, model_version=model_version
        )
        task_predictions = predictor.predict(
            [(row["premise"], row["hypothesis"]) for row in task_rows]
        )
        if recorder is not None:
            dumped.extend(
                _dump_rows(
                    model_id=model_id,
                    tier="task",
                    rows=task_rows,
                    predictions=task_predictions,
                    scores=recorder.scores,
                )
            )
        entry: dict[str, object] = {
            "tokenizer_variant": tokenizer_variant,
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
            if recorder is not None:
                dumped.extend(
                    _dump_rows(
                        model_id=model_id,
                        tier="external",
                        rows=external_rows,
                        predictions=external_predictions,
                        scores=recorder.scores,
                    )
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
    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in dumped),
            encoding="utf-8",
        )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
