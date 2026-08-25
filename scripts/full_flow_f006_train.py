"""Train one frozen F006 LoRA arm for key-fact JSON extraction.

Both arms use the same query order, targets, optimiser, number of examples and
steps.  ``clean`` duplicates each clean prompt; ``mixed`` uses one clean and one
mixed prompt per query.  Loss is applied only to assistant target tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

Arm = Literal["clean", "mixed"]
MAX_LENGTH = 1792
SEED = 13
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATION = 8
EPOCHS = 1
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        yield value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_examples(
    cases: Sequence[Mapping[str, Any]], arm: Arm
) -> list[tuple[str, str, str]]:
    """Return (query ID, prompt, target), with equal counts for both arms."""

    examples: list[tuple[str, str, str]] = []
    for row in cases:
        query_id = str(row["query_id"])
        clean = (query_id, str(row["clean_prompt"]), str(row["clean_target"]))
        if arm == "clean":
            examples.extend((clean, clean))
        else:
            examples.extend(
                (
                    clean,
                    (query_id, str(row["mixed_prompt"]), str(row["mixed_target"])),
                )
            )
    return examples


def encode_example(
    tokenizer: Any,
    prompt: str,
    target: str,
    *,
    max_length: int = MAX_LENGTH,
) -> tuple[list[int], list[int]]:
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = list(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    target_ids = list(tokenizer(target, add_special_tokens=False)["input_ids"])
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Granite tokenizer has no EOS token")
    input_ids = [*prompt_ids, *target_ids, int(eos_id)]
    if len(input_ids) > max_length:
        raise ValueError(f"F006 example length {len(input_ids)} exceeds frozen {max_length}")
    labels = [-100] * len(prompt_ids) + target_ids + [int(eos_id)]
    if len(input_ids) != len(labels) or all(value == -100 for value in labels):
        raise AssertionError("invalid F006 assistant-only labels")
    return input_ids, labels


def _evaluate_loss(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    cases: Sequence[Mapping[str, Any]],
    field: Literal["clean", "mixed"],
) -> float:
    model.eval()
    losses: list[float] = []
    prompt_key = f"{field}_prompt"
    target_key = f"{field}_target"
    with torch.no_grad():
        for row in cases:
            input_ids, labels = encode_example(
                tokenizer, str(row[prompt_key]), str(row[target_key])
            )
            ids = torch.tensor([input_ids], dtype=torch.long, device="cuda")
            target = torch.tensor([labels], dtype=torch.long, device="cuda")
            mask = torch.ones_like(ids)
            loss = model(input_ids=ids, attention_mask=mask, labels=target).loss
            losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses)


def train(
    *,
    arm: Arm,
    model_snapshot: Path,
    train_cases_path: Path,
    validation_cases_path: Path,
    output_dir: Path,
    max_queries: int | None = None,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("F006 training output must be absent or empty")
    try:
        peft = importlib.import_module("peft")
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as error:  # pragma: no cover - server-only dependencies
        raise RuntimeError("F006 training requires torch, transformers and peft") from error

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cases = list(_jsonl(train_cases_path))
    validation = list(_jsonl(validation_cases_path))
    if max_queries is not None:
        if max_queries <= 0:
            raise ValueError("--max-queries must be positive")
        cases = cases[:max_queries]
        validation = validation[: min(max_queries, len(validation))]
    examples = training_examples(cases, arm)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_snapshot.resolve()), local_files_only=True
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
    for example_index, (_query_id, prompt, target) in enumerate(examples, start=1):
        input_ids, labels = encode_example(tokenizer, prompt, target)
        ids = torch.tensor([input_ids], dtype=torch.long, device="cuda")
        target_ids = torch.tensor([labels], dtype=torch.long, device="cuda")
        mask = torch.ones_like(ids)
        loss = model(input_ids=ids, attention_mask=mask, labels=target_ids).loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite F006 loss at example {example_index}")
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
                f"[F006 {arm}] {example_index}/{len(examples)} "
                f"loss={sum(recent)/len(recent):.4f} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )

    clean_validation_loss = _evaluate_loss(
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        cases=validation,
        field="clean",
    )
    mixed_validation_loss = _evaluate_loss(
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        cases=validation,
        field="mixed",
    )
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)

    manifest: dict[str, object] = {
        "schema_version": "full-flow-f006-training-manifest-v1",
        "status": "COMPLETE",
        "arm": arm,
        "seed": SEED,
        "source_role": "NIAH train only",
        "decision_dev_used": False,
        "sealed_or_heldout_read": False,
        "model_snapshot": str(model_snapshot.resolve()),
        "model_snapshot_config_sha256": _sha256(model_snapshot / "config.json"),
        "train_cases_sha256": _sha256(train_cases_path),
        "validation_cases_sha256": _sha256(validation_cases_path),
        "queries": len(cases),
        "training_examples": len(examples),
        "validation_queries": len(validation),
        "epochs": EPOCHS,
        "max_length": MAX_LENGTH,
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
        "clean_validation_loss": clean_validation_loss,
        "mixed_validation_loss": mixed_validation_loss,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": peak_allocated,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("clean", "mixed"))
    parser.add_argument("--model-snapshot", required=True, type=Path)
    parser.add_argument("--train-cases", required=True, type=Path)
    parser.add_argument("--validation-cases", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-queries", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    train(
        arm=args.arm,
        model_snapshot=args.model_snapshot,
        train_cases_path=args.train_cases,
        validation_cases_path=args.validation_cases,
        output_dir=args.output_dir.resolve(),
        max_queries=args.max_queries,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
