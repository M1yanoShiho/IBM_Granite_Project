"""Audit lengths and train the G220 Granite draft LoRA arms.

GC and GM use the same G200 queries, semantic targets, eight example slots per
query, optimiser, and number of updates. GC maps every slot to support-only;
GM maps the slots to the eight frozen context variants. Loss is applied only to
assistant target tokens. This script never reads development, sealed, or
system-held-out data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import re
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from full_flow_g200 import VARIANT_NAMES

Arm = Literal["gc", "gm"]
RunKind = Literal["smoke", "formal"]
ALLOWED_SEEDS = (13, 42, 73)
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATION = 8
EPOCHS = 1
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
LENGTH_THRESHOLDS = (1536, 1792, 2048, 2304, 2560, 3072, 4096)


@dataclass(frozen=True, slots=True)
class TrainingExample:
    query_id: str
    slot: str
    source_variant: str
    prompt: str
    target: str


def _jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _require_empty(output_dir: Path, label: str) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"{label} output directory must be absent or empty")


def _load_data_manifest(
    *, data_manifest_path: Path, train_cases_path: Path, validation_cases_path: Path
) -> Mapping[str, Any]:
    value = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("status") != "COMPLETE":
        raise ValueError("G220 requires a COMPLETE G200 data manifest")
    if value.get("schema_version") != "full-flow-g200-data-manifest-v1":
        raise ValueError("G220 received an unknown G200 data manifest schema")
    if str(value.get("train_cases_sha256", "")) != _sha256(train_cases_path):
        raise ValueError("G200 train cases hash differs from its manifest")
    if str(value.get("validation_cases_sha256", "")) != _sha256(validation_cases_path):
        raise ValueError("G200 validation cases hash differs from its manifest")
    if int(value.get("variants_per_query", 0)) != len(VARIANT_NAMES):
        raise ValueError("G200 variant count differs from the frozen G220 design")
    if tuple(value.get("variant_order", ())) != VARIANT_NAMES:
        raise ValueError("G200 variant order differs from the frozen G220 design")
    if int(value.get("gc_examples", -1)) != int(value.get("gm_examples", -2)):
        raise ValueError("G200 GC/GM example counts differ")
    return value


def _validated_cases(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    output: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        query_id = str(row.get("query_id", ""))
        variants = row.get("variants")
        if (
            not query_id
            or query_id in seen
            or row.get("schema_version") != "full-flow-g200-case-v1"
            or not isinstance(variants, Mapping)
            or set(variants) != set(VARIANT_NAMES)
        ):
            raise ValueError(f"invalid or duplicate G200 case: {query_id!r}")
        targets = set()
        for name in VARIANT_NAMES:
            variant = variants.get(name)
            if not isinstance(variant, Mapping):
                raise ValueError(f"query {query_id} lacks variant {name}")
            prompt = str(variant.get("prompt", ""))
            target = str(variant.get("target", ""))
            if not prompt or not target:
                raise ValueError(f"query {query_id} has an empty {name} example")
            targets.add(_strip_citation(target))
        if len(targets) != 1:
            raise ValueError(f"query {query_id} variants differ in semantic target")
        seen.add(query_id)
        output.append(row)
    if not output:
        raise ValueError("G220 received no G200 cases")
    return output


def _strip_citation(target: str) -> str:
    return re.sub(r"\s+\[\d+\]([.!?])$", r"\1", target.strip())


def training_examples(
    cases: Sequence[Mapping[str, Any]], arm: Arm
) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for row in _validated_cases(cases):
        query_id = str(row["query_id"])
        variants = row["variants"]
        assert isinstance(variants, Mapping)
        for slot in VARIANT_NAMES:
            source = "support_only" if arm == "gc" else slot
            variant = variants[source]
            assert isinstance(variant, Mapping)
            examples.append(
                TrainingExample(
                    query_id=query_id,
                    slot=slot,
                    source_variant=source,
                    prompt=str(variant["prompt"]),
                    target=str(variant["target"]),
                )
            )
    return examples


def _chat_prompt_ids(tokenizer: Any, prompt: str) -> list[int]:
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return list(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])


def example_length(tokenizer: Any, prompt: str, target: str) -> int:
    prompt_ids = _chat_prompt_ids(tokenizer, prompt)
    target_ids = list(tokenizer(target, add_special_tokens=False)["input_ids"])
    if tokenizer.eos_token_id is None:
        raise ValueError("Granite tokenizer has no EOS token")
    return len(prompt_ids) + len(target_ids) + 1


def encode_example(
    tokenizer: Any, prompt: str, target: str, *, max_length: int
) -> tuple[list[int], list[int]]:
    prompt_ids = _chat_prompt_ids(tokenizer, prompt)
    target_ids = list(tokenizer(target, add_special_tokens=False)["input_ids"])
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Granite tokenizer has no EOS token")
    input_ids = [*prompt_ids, *target_ids, int(eos_id)]
    if len(input_ids) > max_length:
        raise ValueError(f"G220 example length {len(input_ids)} exceeds frozen {max_length}")
    labels = [-100] * len(prompt_ids) + target_ids + [int(eos_id)]
    if len(input_ids) != len(labels) or all(value == -100 for value in labels):
        raise AssertionError("invalid G220 assistant-only labels")
    return input_ids, labels


def _percentile(values: Sequence[int], probability: float) -> int:
    if not values:
        raise ValueError("cannot take percentile of empty values")
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def audit_lengths(
    *,
    model_snapshot: Path,
    data_manifest_path: Path,
    train_cases_path: Path,
    validation_cases_path: Path,
    output_json: Path,
) -> dict[str, object]:
    _load_data_manifest(
        data_manifest_path=data_manifest_path,
        train_cases_path=train_cases_path,
        validation_cases_path=validation_cases_path,
    )
    transformers = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_snapshot.resolve()), local_files_only=True
    )
    train = _validated_cases(_jsonl(train_cases_path))
    validation = _validated_cases(_jsonl(validation_cases_path))
    report: dict[str, object] = {
        "schema_version": "full-flow-g220-length-audit-v1",
        "status": "COMPLETE",
        "model_snapshot": str(model_snapshot.resolve()),
        "model_snapshot_config_sha256": _sha256(model_snapshot / "config.json"),
        "train_cases_sha256": _sha256(train_cases_path),
        "validation_cases_sha256": _sha256(validation_cases_path),
        "arms": {},
    }
    arm_reports: dict[str, object] = {}
    for arm in ("gc", "gm"):
        examples = [*training_examples(train, arm), *training_examples(validation, arm)]
        lengths = [example_length(tokenizer, item.prompt, item.target) for item in examples]
        arm_reports[arm] = {
            "queries": len(train) + len(validation),
            "examples": len(examples),
            "min": min(lengths),
            "p50": _percentile(lengths, 0.50),
            "p90": _percentile(lengths, 0.90),
            "p95": _percentile(lengths, 0.95),
            "p99": _percentile(lengths, 0.99),
            "max": max(lengths),
            "above_threshold": {
                str(threshold): sum(value > threshold for value in lengths)
                for threshold in LENGTH_THRESHOLDS
            },
        }
    report["arms"] = arm_reports
    _write_json(output_json, report)
    return report


def _evaluate_losses(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    cases: Sequence[Mapping[str, Any]],
    max_length: int,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for row in cases:
            query_id = str(row["query_id"])
            variants = row["variants"]
            assert isinstance(variants, Mapping)
            for name in VARIANT_NAMES:
                variant = variants[name]
                assert isinstance(variant, Mapping)
                input_ids, labels = encode_example(
                    tokenizer,
                    str(variant["prompt"]),
                    str(variant["target"]),
                    max_length=max_length,
                )
                ids = torch.tensor([input_ids], dtype=torch.long, device="cuda")
                target = torch.tensor([labels], dtype=torch.long, device="cuda")
                mask = torch.ones_like(ids)
                loss = model(input_ids=ids, attention_mask=mask, labels=target).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite G220 validation loss for {query_id}/{name}"
                    )
                losses[name].append(float(loss.detach().cpu()))
    model.train()
    return {name: sum(values) / len(values) for name, values in losses.items()}


def train(
    *,
    arm: Arm,
    run_kind: RunKind,
    seed: int,
    max_length: int,
    model_snapshot: Path,
    data_manifest_path: Path,
    train_cases_path: Path,
    validation_cases_path: Path,
    output_dir: Path,
    max_queries: int | None = None,
) -> dict[str, object]:
    _require_empty(output_dir, "G220 training")
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed must be one of {ALLOWED_SEEDS}")
    if max_length <= 0:
        raise ValueError("max length must be positive")
    if run_kind == "formal" and max_queries is not None:
        raise ValueError("formal G220 training cannot use --max-queries")
    if run_kind == "smoke" and (max_queries is None or max_queries <= 0):
        raise ValueError("smoke G220 training requires positive --max-queries")
    data_manifest = _load_data_manifest(
        data_manifest_path=data_manifest_path,
        train_cases_path=train_cases_path,
        validation_cases_path=validation_cases_path,
    )
    try:
        peft = importlib.import_module("peft")
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as error:  # pragma: no cover - server-only dependencies
        raise RuntimeError("G220 training requires torch, transformers and peft") from error

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cases = _validated_cases(_jsonl(train_cases_path))
    validation = _validated_cases(_jsonl(validation_cases_path))
    if max_queries is not None:
        cases = cases[:max_queries]
        validation = validation[: min(max_queries, len(validation))]
    examples = training_examples(cases, arm)
    random.Random(seed).shuffle(examples)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_snapshot.resolve()), local_files_only=True
    )
    lengths = [example_length(tokenizer, item.prompt, item.target) for item in examples]
    over_length = sum(value > max_length for value in lengths)
    if over_length:
        raise ValueError(
            f"{over_length}/{len(lengths)} G220 examples exceed frozen max_length={max_length}"
        )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_snapshot.resolve()),
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=list(TARGET_MODULES),
            bias="none",
        ),
    )
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    all_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)

    started = time.perf_counter()
    losses: list[float] = []
    optimizer_steps = 0
    peak_allocated = 0
    for example_index, example in enumerate(examples, start=1):
        input_ids, labels = encode_example(
            tokenizer, example.prompt, example.target, max_length=max_length
        )
        ids = torch.tensor([input_ids], dtype=torch.long, device="cuda")
        target_ids = torch.tensor([labels], dtype=torch.long, device="cuda")
        mask = torch.ones_like(ids)
        loss = model(input_ids=ids, attention_mask=mask, labels=target_ids).loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite G220 loss at example {example_index}")
        losses.append(float(loss.detach().cpu()))
        (loss / GRADIENT_ACCUMULATION).backward()
        should_step = (
            example_index % GRADIENT_ACCUMULATION == 0 or example_index == len(examples)
        )
        if should_step:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated()))
        if example_index % 100 == 0 or example_index == len(examples):
            recent = losses[-100:]
            print(
                f"[G220 {arm}/{seed}] {example_index}/{len(examples)} "
                f"loss={sum(recent)/len(recent):.4f} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )

    validation_losses = _evaluate_losses(
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        cases=validation,
        max_length=max_length,
    )
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not weights_path.is_file():
        raise ValueError("G220 did not write adapter_model.safetensors")

    manifest: dict[str, object] = {
        "schema_version": "full-flow-g220-training-manifest-v1",
        "status": "COMPLETE",
        "run_kind": run_kind,
        "arm": arm,
        "seed": seed,
        "source_role": "G200 COMPLETE NIAH train only",
        "decision_dev_used": False,
        "sealed_or_heldout_read": False,
        "adapter_scope": "draft generation call only",
        "key_fact_extraction_scope": "not used",
        "claim_splitter_scope": "frozen Granite base with adapter disabled",
        "true_scope": "frozen verifier; not trained",
        "model_snapshot": str(model_snapshot.resolve()),
        "model_snapshot_config_sha256": _sha256(model_snapshot / "config.json"),
        "data_manifest_sha256": _sha256(data_manifest_path),
        "train_cases_sha256": _sha256(train_cases_path),
        "validation_cases_sha256": _sha256(validation_cases_path),
        "full_data_train_queries": int(data_manifest["train_queries"]),
        "queries": len(cases),
        "training_examples": len(examples),
        "validation_queries": len(validation),
        "epochs": EPOCHS,
        "max_length": max_length,
        "observed_max_training_length": max(lengths),
        "truncated_examples": 0,
        "truncation_rate": 0.0,
        "microbatch_size": 1,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "optimizer_steps": optimizer_steps,
        "learning_rate": LEARNING_RATE,
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": list(TARGET_MODULES),
            "bias": "none",
        },
        "trainable_parameters": trainable_count,
        "all_parameters_with_adapter": all_count,
        "trainable_percent": 100.0 * trainable_count / all_count,
        "mean_training_loss": sum(losses) / len(losses),
        "final_100_training_loss": sum(losses[-100:]) / len(losses[-100:]),
        "validation_loss_by_variant": validation_losses,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": peak_allocated,
        "adapter_weights_sha256": _sha256(weights_path),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
    }
    _write_json(output_dir / "training_manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    lengths = commands.add_parser("audit-lengths")
    lengths.add_argument("--model-snapshot", required=True, type=Path)
    lengths.add_argument("--data-manifest", required=True, type=Path)
    lengths.add_argument("--train-cases", required=True, type=Path)
    lengths.add_argument("--validation-cases", required=True, type=Path)
    lengths.add_argument("--output-json", required=True, type=Path)

    fit = commands.add_parser("train")
    fit.add_argument("--arm", required=True, choices=("gc", "gm"))
    fit.add_argument("--run-kind", required=True, choices=("smoke", "formal"))
    fit.add_argument("--seed", required=True, type=int, choices=ALLOWED_SEEDS)
    fit.add_argument("--max-length", required=True, type=int)
    fit.add_argument("--model-snapshot", required=True, type=Path)
    fit.add_argument("--data-manifest", required=True, type=Path)
    fit.add_argument("--train-cases", required=True, type=Path)
    fit.add_argument("--validation-cases", required=True, type=Path)
    fit.add_argument("--output-dir", required=True, type=Path)
    fit.add_argument("--max-queries", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit-lengths":
        report = audit_lengths(
            model_snapshot=args.model_snapshot,
            data_manifest_path=args.data_manifest,
            train_cases_path=args.train_cases,
            validation_cases_path=args.validation_cases,
            output_json=args.output_json,
        )
    else:
        report = train(
            arm=args.arm,
            run_kind=args.run_kind,
            seed=args.seed,
            max_length=args.max_length,
            model_snapshot=args.model_snapshot,
            data_manifest_path=args.data_manifest,
            train_cases_path=args.train_cases,
            validation_cases_path=args.validation_cases,
            output_dir=args.output_dir.resolve(),
            max_queries=args.max_queries,
        )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
