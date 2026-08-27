"""G300 controlled-entry Granite draft LoRA training utilities.

G300 consumes only the G223 controlled-continuation candidate.  It supports a
limited smoke run and the later GR-F/GR-C recipes, while preserving the frozen
runtime boundary: the adapter is for the draft generation call only, the claim
splitter stays on the frozen Granite base, TRUE is frozen, and no dev, sealed,
held-out, or Selector utility-label data is read here.
"""

from __future__ import annotations

import argparse
import gc
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

Recipe = Literal["gr-f", "gr-c"]
RunKind = Literal["smoke", "formal"]
SmokeSelection = Literal["longest"]

SCHEMA_G223_MANIFEST = "full-flow-g223-residual-sample-failure-candidate-manifest-v1"
SCHEMA_CASE = "full-flow-g200-case-v2"
SCHEMA_G300_LENGTH_AUDIT = "full-flow-g300-length-audit-v1"
SCHEMA_G300_TRAINING = "full-flow-g300-draft-lora-training-manifest-v1"

MAX_LENGTH = 2304
ALLOWED_SEEDS = (13, 42, 73)
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
GR_F_LEARNING_RATE = 1e-4
GR_C_LEARNING_RATE = 5e-5
GRADIENT_ACCUMULATION_GROUPS = 8
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
CITATION_RE = re.compile(r"\[\d+\]")
SCRIPT_PATH = Path(__file__).resolve()

CAT_PROMPT = 0
CAT_ANSWER = 1
CAT_CITATION = 2
CAT_EOS = 3


@dataclass(frozen=True, slots=True)
class TrainingVariant:
    name: str
    prompt: str
    target: str


@dataclass(frozen=True, slots=True)
class TrainingGroup:
    case_id: str
    dataset: str
    role: str
    target_kind: str
    answerable: bool
    variants: tuple[TrainingVariant, ...]


@dataclass(frozen=True, slots=True)
class EncodedExample:
    input_ids: list[int]
    labels: list[int]
    weights: list[float]
    categories: list[int]


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} is not a JSON object")
    return value


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


def _load_g223_manifest(
    *,
    data_manifest_path: Path,
    train_cases_path: Path,
    validation_cases_path: Path,
    ordered_ids_path: Path,
) -> Mapping[str, Any]:
    manifest = _json(data_manifest_path)
    if manifest.get("schema_version") != SCHEMA_G223_MANIFEST:
        raise ValueError("G300 expects the G223 controlled-continuation manifest")
    if manifest.get("status") != "CONTROLLED_CONTINUATION_READY":
        raise ValueError("G300 requires G223 status CONTROLLED_CONTINUATION_READY")
    if manifest.get("controlled_continuation_ready") is not True:
        raise ValueError("G300 requires controlled_continuation_ready=true")
    if manifest.get("g300_unlocked") is not True:
        raise ValueError("G300 requires g300_unlocked=true")
    if manifest.get("g300_entry_mode") != "controlled_continuation_limited_internal_screen":
        raise ValueError("G300 requires the limited controlled entry mode")
    if manifest.get("clean_freeze_ready") is not False:
        raise ValueError("G300 controlled entry must not be a clean freeze")
    if manifest.get("training_started") is not False:
        raise ValueError("G223 input must record training_started=false")
    if manifest.get("utility_labels_started") is not False:
        raise ValueError("G223 input must record utility_labels_started=false")
    if manifest.get("sealed_or_heldout_read") is not False or manifest.get("dev_read") is not False:
        raise ValueError("G223 input must not have read dev/sealed/held-out data")
    if str(manifest.get("train_cases_sha256", "")) != _sha256(train_cases_path):
        raise ValueError("train cases hash differs from the G223 manifest")
    if str(manifest.get("validation_cases_sha256", "")) != _sha256(validation_cases_path):
        raise ValueError("validation cases hash differs from the G223 manifest")
    if str(manifest.get("ordered_ids_sha256", "")) != _sha256(ordered_ids_path):
        raise ValueError("ordered IDs hash differs from the G223 manifest")
    return manifest


def _validated_cases(rows: Sequence[Mapping[str, Any]], *, split_name: str) -> list[Mapping[str, Any]]:
    output: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        case_id = str(row.get("case_id", ""))
        variants = row.get("variants")
        if not case_id or case_id in seen:
            raise ValueError(f"duplicate or blank {split_name} case_id: {case_id!r}")
        if row.get("schema_version") != SCHEMA_CASE:
            raise ValueError(f"{case_id} has an unsupported case schema")
        if not isinstance(variants, Mapping) or not variants:
            raise ValueError(f"{case_id} lacks variants")
        for variant_name, variant in variants.items():
            if not isinstance(variant, Mapping):
                raise ValueError(f"{case_id}/{variant_name} is not a variant object")
            if not str(variant.get("prompt", "")).strip():
                raise ValueError(f"{case_id}/{variant_name} has an empty prompt")
            if not str(variant.get("target", "")).strip():
                raise ValueError(f"{case_id}/{variant_name} has an empty target")
        seen.add(case_id)
        output.append(row)
    if not output:
        raise ValueError(f"G300 received no {split_name} cases")
    return output


def training_groups(cases: Sequence[Mapping[str, Any]]) -> list[TrainingGroup]:
    groups: list[TrainingGroup] = []
    for row in _validated_cases(cases, split_name="training"):
        variants = row["variants"]
        assert isinstance(variants, Mapping)
        groups.append(
            TrainingGroup(
                case_id=str(row["case_id"]),
                dataset=str(row.get("dataset", "")),
                role=str(row.get("role", "")),
                target_kind=str(row.get("target_kind", "")),
                answerable=bool(row.get("answerable")),
                variants=tuple(
                    TrainingVariant(
                        name=str(name),
                        prompt=str(variant["prompt"]),
                        target=str(variant["target"]),
                    )
                    for name, variant in variants.items()
                    if isinstance(variant, Mapping)
                ),
            )
        )
    return groups


def _chat_prompt_ids(tokenizer: Any, prompt: str) -> list[int]:
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return list(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])


def _citation_spans(target: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in CITATION_RE.finditer(target)]


def _overlaps(start: int, end: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(start < span_end and end > span_start for span_start, span_end in spans)


def _target_ids_weights_categories(tokenizer: Any, target: str) -> tuple[list[int], list[float], list[int]]:
    try:
        encoded = tokenizer(
            target,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except TypeError as error:
        raise ValueError("G300 citation weighting requires tokenizer offset mappings") from error
    if "offset_mapping" not in encoded:
        raise ValueError("G300 citation weighting requires tokenizer offset mappings")
    input_ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    if len(input_ids) != len(offsets):
        raise ValueError("tokenizer returned mismatched input_ids and offset_mapping")
    spans = _citation_spans(target)
    weights: list[float] = []
    categories: list[int] = []
    for start, end in offsets:
        start_i = int(start)
        end_i = int(end)
        if _overlaps(start_i, end_i, spans):
            weights.append(4.0)
            categories.append(CAT_CITATION)
        else:
            weights.append(1.0)
            categories.append(CAT_ANSWER)
    return input_ids, weights, categories


def encode_example(
    tokenizer: Any,
    prompt: str,
    target: str,
    *,
    max_length: int = MAX_LENGTH,
) -> EncodedExample:
    prompt_ids = _chat_prompt_ids(tokenizer, prompt)
    target_ids, target_weights, target_categories = _target_ids_weights_categories(
        tokenizer, target
    )
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Granite tokenizer has no EOS token")
    input_ids = [*prompt_ids, *target_ids, int(eos_id)]
    if len(input_ids) > max_length:
        raise ValueError(f"G300 example length {len(input_ids)} exceeds frozen {max_length}")
    labels = [-100] * len(prompt_ids) + target_ids + [int(eos_id)]
    weights = [0.0] * len(prompt_ids) + target_weights + [1.0]
    categories = [CAT_PROMPT] * len(prompt_ids) + target_categories + [CAT_EOS]
    if not (len(input_ids) == len(labels) == len(weights) == len(categories)):
        raise AssertionError("invalid G300 encoded example lengths")
    if all(label == -100 for label in labels):
        raise AssertionError("invalid G300 assistant-only labels")
    return EncodedExample(
        input_ids=input_ids,
        labels=labels,
        weights=weights,
        categories=categories,
    )


def example_length(tokenizer: Any, prompt: str, target: str) -> int:
    return len(encode_example(tokenizer, prompt, target, max_length=10**9).input_ids)


def _percentile(values: Sequence[int], probability: float) -> int:
    if not values:
        raise ValueError("cannot take percentile of empty values")
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def _case_count(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        dataset = str(row.get("dataset", ""))
        role = str(row.get("role", ""))
        target_kind = str(row.get("target_kind", ""))
        answerable = "answerable" if row.get("answerable") else "unsupported"
        counts[f"{dataset}:{role}:{target_kind}:{answerable}"] += 1
    return dict(sorted(counts.items()))


def _length_summary(
    *,
    tokenizer: Any,
    groups: Sequence[TrainingGroup],
    max_length: int,
) -> dict[str, object]:
    lengths: list[int] = []
    examples = 0
    token_counts = {
        "answer_tokens": 0,
        "citation_tokens": 0,
        "eos_tokens": 0,
        "prompt_tokens_masked": 0,
    }
    for group in groups:
        for variant in group.variants:
            encoded = encode_example(
                tokenizer,
                variant.prompt,
                variant.target,
                max_length=max_length,
            )
            lengths.append(len(encoded.input_ids))
            examples += 1
            token_counts["answer_tokens"] += sum(
                category == CAT_ANSWER for category in encoded.categories
            )
            token_counts["citation_tokens"] += sum(
                category == CAT_CITATION for category in encoded.categories
            )
            token_counts["eos_tokens"] += sum(category == CAT_EOS for category in encoded.categories)
            token_counts["prompt_tokens_masked"] += sum(
                category == CAT_PROMPT for category in encoded.categories
            )
    return {
        "groups": len(groups),
        "examples": examples,
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
        "over_frozen_max_length": sum(value > max_length for value in lengths),
        "token_counts": token_counts,
    }


def audit_lengths(
    *,
    model_snapshot: Path,
    data_manifest_path: Path,
    train_cases_path: Path,
    validation_cases_path: Path,
    ordered_ids_path: Path,
    output_json: Path,
    max_length: int = MAX_LENGTH,
) -> dict[str, object]:
    manifest = _load_g223_manifest(
        data_manifest_path=data_manifest_path,
        train_cases_path=train_cases_path,
        validation_cases_path=validation_cases_path,
        ordered_ids_path=ordered_ids_path,
    )
    transformers = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_snapshot.resolve()),
        local_files_only=True,
    )
    train_rows = _validated_cases(_jsonl(train_cases_path), split_name="train")
    validation_rows = _validated_cases(_jsonl(validation_cases_path), split_name="validation")
    train_groups = training_groups(train_rows)
    validation_groups = training_groups(validation_rows)
    report: dict[str, object] = {
        "schema_version": SCHEMA_G300_LENGTH_AUDIT,
        "status": "PASS",
        "stage": "G300",
        "entry_mode": "controlled_continuation_limited_internal_screen",
        "clean_freeze_ready": False,
        "script_sha256": _sha256(SCRIPT_PATH),
        "model_snapshot": str(model_snapshot.resolve()),
        "model_snapshot_config_sha256": _sha256(model_snapshot / "config.json"),
        "g223_manifest_sha256": _sha256(data_manifest_path),
        "g223_status": str(manifest.get("status", "")),
        "g223_twowiki_modelval_groups": int(
            manifest.get("counts", {}).get("2wiki", {}).get("train-modelval_answerable_groups", 0)
            if isinstance(manifest.get("counts"), Mapping)
            else 0
        ),
        "train_cases_sha256": _sha256(train_cases_path),
        "validation_cases_sha256": _sha256(validation_cases_path),
        "ordered_ids_sha256": _sha256(ordered_ids_path),
        "max_length": max_length,
        "train": _length_summary(tokenizer=tokenizer, groups=train_groups, max_length=max_length),
        "validation": _length_summary(
            tokenizer=tokenizer, groups=validation_groups, max_length=max_length
        ),
        "train_case_counts": _case_count(train_rows),
        "validation_case_counts": _case_count(validation_rows),
        "token_weighting": {
            "prompt": 0,
            "answer_or_punctuation": 1,
            "citation_brackets_or_index": 4,
            "eos": 1,
        },
        "query_group_equalization": (
            "Each case contributes the mean loss over its variants; variants are not "
            "treated as independent questions."
        ),
        "sealed_or_heldout_read": False,
        "dev_read": False,
        "utility_labels_started": False,
        "training_started": False,
    }
    if (
        int(report["train"]["over_frozen_max_length"])  # type: ignore[index]
        or int(report["validation"]["over_frozen_max_length"])  # type: ignore[index]
    ):
        report["status"] = "FAIL"
    _write_json(output_json, report)
    return report


def select_longest_groups(
    groups: Sequence[TrainingGroup],
    tokenizer: Any,
    max_groups: int,
    *,
    max_length: int = MAX_LENGTH,
) -> list[TrainingGroup]:
    if max_groups <= 0:
        raise ValueError("max groups must be positive")
    scored: list[tuple[int, str, TrainingGroup]] = []
    for group in groups:
        maximum = max(
            example_length(tokenizer, variant.prompt, variant.target)
            for variant in group.variants
        )
        if maximum > max_length:
            raise ValueError(
                f"{group.case_id} has an example length {maximum} above max_length={max_length}"
            )
        scored.append((maximum, group.case_id, group))
    return [group for _length, _case_id, group in sorted(scored, key=lambda x: (-x[0], x[1]))][
        :max_groups
    ]


def _learning_rate(recipe: Recipe) -> float:
    return GR_C_LEARNING_RATE if recipe == "gr-c" else GR_F_LEARNING_RATE


def _validate_recipe(recipe: Recipe, init_adapter: Path | None) -> None:
    if recipe == "gr-c" and init_adapter is None:
        raise ValueError("GR-C requires --init-adapter")
    if recipe == "gr-f" and init_adapter is not None:
        raise ValueError("GR-F must start from the frozen Granite base, without --init-adapter")


def _load_trainable_model(
    *,
    peft: Any,
    transformers: Any,
    torch: Any,
    recipe: Recipe,
    model_snapshot: Path,
    init_adapter: Path | None,
) -> Any:
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_snapshot.resolve()),
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()
    if recipe == "gr-c":
        assert init_adapter is not None
        return peft.PeftModel.from_pretrained(
            base_model,
            str(init_adapter.resolve()),
            is_trainable=True,
        )
    return peft.get_peft_model(
        base_model,
        peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=list(TARGET_MODULES),
            bias="none",
        ),
    )


def _weighted_causal_lm_loss(
    *,
    torch: Any,
    logits: Any,
    labels: Any,
    weights: Any,
    categories: Any,
) -> tuple[Any, dict[str, float]]:
    functional = torch.nn.functional
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_weights = weights[..., 1:].contiguous()
    shift_categories = categories[..., 1:].contiguous()
    flat_losses = functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shift_labels)
    valid = shift_labels.ne(-100)
    weighted_denominator = shift_weights[valid].sum()
    if float(weighted_denominator.detach().cpu()) <= 0:
        raise ValueError("G300 weighted loss has no target tokens")
    total = (flat_losses * shift_weights).sum() / weighted_denominator
    stats: dict[str, float] = {}
    for name, category in (
        ("answer", CAT_ANSWER),
        ("citation", CAT_CITATION),
        ("eos", CAT_EOS),
    ):
        mask = valid & shift_categories.eq(category)
        if bool(mask.any()):
            stats[name] = float(flat_losses[mask].mean().detach().cpu())
    return total, stats


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _evaluate_groups(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    groups: Sequence[TrainingGroup],
    max_length: int,
) -> dict[str, object]:
    model.eval()
    group_losses: list[float] = []
    category_losses: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for group in groups:
            variant_losses: list[float] = []
            for variant in group.variants:
                encoded = encode_example(
                    tokenizer,
                    variant.prompt,
                    variant.target,
                    max_length=max_length,
                )
                ids = torch.tensor([encoded.input_ids], dtype=torch.long, device="cuda")
                labels = torch.tensor([encoded.labels], dtype=torch.long, device="cuda")
                weights = torch.tensor([encoded.weights], dtype=torch.float32, device="cuda")
                categories = torch.tensor([encoded.categories], dtype=torch.long, device="cuda")
                attention_mask = torch.ones_like(ids)
                outputs = model(input_ids=ids, attention_mask=attention_mask)
                loss, stats = _weighted_causal_lm_loss(
                    torch=torch,
                    logits=outputs.logits,
                    labels=labels,
                    weights=weights,
                    categories=categories,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite G300 validation loss for {group.case_id}")
                variant_losses.append(float(loss.detach().cpu()))
                for name, value in stats.items():
                    category_losses[name].append(value)
            group_losses.append(sum(variant_losses) / len(variant_losses))
    model.train()
    return {
        "groups": len(groups),
        "mean_group_weighted_loss": _mean(group_losses),
        "mean_answer_token_loss": _mean(category_losses["answer"]),
        "mean_citation_token_loss": _mean(category_losses["citation"]),
        "mean_eos_token_loss": _mean(category_losses["eos"]),
    }


def train(
    *,
    recipe: Recipe,
    run_kind: RunKind,
    seed: int,
    model_snapshot: Path,
    data_manifest_path: Path,
    train_cases_path: Path,
    validation_cases_path: Path,
    ordered_ids_path: Path,
    output_dir: Path,
    init_adapter: Path | None = None,
    max_groups: int | None = None,
    smoke_selection: SmokeSelection | None = None,
    max_length: int = MAX_LENGTH,
) -> dict[str, object]:
    _require_empty(output_dir, "G300 training")
    _validate_recipe(recipe, init_adapter)
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed must be one of {ALLOWED_SEEDS}")
    if max_length <= 0:
        raise ValueError("max length must be positive")
    if run_kind == "formal" and (max_groups is not None or smoke_selection is not None):
        raise ValueError("formal G300 training cannot use smoke group selection")
    if run_kind == "smoke" and (
        max_groups is None or max_groups <= 0 or smoke_selection != "longest"
    ):
        raise ValueError("smoke G300 training requires positive longest-group selection")
    manifest = _load_g223_manifest(
        data_manifest_path=data_manifest_path,
        train_cases_path=train_cases_path,
        validation_cases_path=validation_cases_path,
        ordered_ids_path=ordered_ids_path,
    )
    try:
        peft = importlib.import_module("peft")
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as error:  # pragma: no cover - server-only dependencies
        raise RuntimeError("G300 training requires torch, transformers and peft") from error
    if not torch.cuda.is_available():
        raise RuntimeError("G300 training requires a CUDA GPU")

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_rows = _validated_cases(_jsonl(train_cases_path), split_name="train")
    validation_rows = _validated_cases(_jsonl(validation_cases_path), split_name="validation")
    full_train_groups = training_groups(train_rows)
    full_validation_groups = training_groups(validation_rows)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_snapshot.resolve()),
        local_files_only=True,
    )
    if max_groups is not None:
        selected_train_groups = select_longest_groups(
            full_train_groups,
            tokenizer,
            max_groups,
            max_length=max_length,
        )
        selected_validation_groups = select_longest_groups(
            full_validation_groups,
            tokenizer,
            min(max_groups, len(full_validation_groups)),
            max_length=max_length,
        )
    else:
        selected_train_groups = list(full_train_groups)
        selected_validation_groups = list(full_validation_groups)

    random.Random(seed).shuffle(selected_train_groups)
    lengths = [
        example_length(tokenizer, variant.prompt, variant.target)
        for group in selected_train_groups
        for variant in group.variants
    ]
    over_length = sum(value > max_length for value in lengths)
    if over_length:
        raise ValueError(
            f"{over_length}/{len(lengths)} G300 examples exceed frozen max_length={max_length}"
        )

    model = _load_trainable_model(
        peft=peft,
        transformers=transformers,
        torch=torch,
        recipe=recipe,
        model_snapshot=model_snapshot,
        init_adapter=init_adapter,
    )
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("G300 training found no trainable adapter parameters")
    trainable_count = sum(parameter.numel() for parameter in trainable)
    all_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=_learning_rate(recipe), weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)

    started = time.perf_counter()
    group_losses: list[float] = []
    category_losses: dict[str, list[float]] = defaultdict(list)
    optimizer_steps = 0
    peak_allocated = 0
    for group_index, group in enumerate(selected_train_groups, start=1):
        variant_losses: list[float] = []
        window_start = ((group_index - 1) // GRADIENT_ACCUMULATION_GROUPS) * (
            GRADIENT_ACCUMULATION_GROUPS
        )
        window_end = min(
            window_start + GRADIENT_ACCUMULATION_GROUPS,
            len(selected_train_groups),
        )
        accumulation_groups = window_end - window_start
        scale = 1.0 / (len(group.variants) * accumulation_groups)
        for variant in group.variants:
            encoded = encode_example(
                tokenizer,
                variant.prompt,
                variant.target,
                max_length=max_length,
            )
            ids = torch.tensor([encoded.input_ids], dtype=torch.long, device="cuda")
            labels = torch.tensor([encoded.labels], dtype=torch.long, device="cuda")
            weights = torch.tensor([encoded.weights], dtype=torch.float32, device="cuda")
            categories = torch.tensor([encoded.categories], dtype=torch.long, device="cuda")
            attention_mask = torch.ones_like(ids)
            outputs = model(input_ids=ids, attention_mask=attention_mask)
            loss, stats = _weighted_causal_lm_loss(
                torch=torch,
                logits=outputs.logits,
                labels=labels,
                weights=weights,
                categories=categories,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite G300 loss at group {group_index}")
            variant_losses.append(float(loss.detach().cpu()))
            for name, value in stats.items():
                category_losses[name].append(value)
            (loss * scale).backward()
        group_losses.append(sum(variant_losses) / len(variant_losses))
        should_step = (
            group_index % GRADIENT_ACCUMULATION_GROUPS == 0
            or group_index == len(selected_train_groups)
        )
        if should_step:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated()))
        if group_index % 25 == 0 or group_index == len(selected_train_groups):
            recent = group_losses[-25:]
            print(
                f"[G300 {recipe}/{run_kind}/seed{seed}] "
                f"{group_index}/{len(selected_train_groups)} groups "
                f"loss={sum(recent)/len(recent):.4f} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )

    validation_loss = _evaluate_groups(
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        groups=selected_validation_groups,
        max_length=max_length,
    )
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    weights_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    if not weights_path.is_file() or not config_path.is_file():
        raise ValueError("G300 did not write a complete adapter")
    weights_sha256 = _sha256(weights_path)
    config_sha256 = _sha256(config_path)

    del optimizer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    reload_base = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_snapshot.resolve()),
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    reloaded = peft.PeftModel.from_pretrained(
        reload_base,
        str(adapter_dir.resolve()),
        is_trainable=False,
    )
    reload_trainable = sum(
        parameter.numel() for parameter in reloaded.parameters() if parameter.requires_grad
    )
    reload_parameter_count = sum(parameter.numel() for parameter in reloaded.parameters())
    if reload_trainable != 0:
        raise ValueError("G300 persisted adapter reload is unexpectedly trainable")
    del reloaded
    del reload_base
    gc.collect()
    torch.cuda.empty_cache()

    counts = manifest.get("counts", {})
    twowiki_modelval_groups = 0
    if isinstance(counts, Mapping):
        twowiki_counts = counts.get("2wiki", {})
        if isinstance(twowiki_counts, Mapping):
            twowiki_modelval_groups = int(
                twowiki_counts.get("train-modelval_answerable_groups", 0)
            )
    train_examples = sum(len(group.variants) for group in selected_train_groups)
    manifest_out: dict[str, object] = {
        "schema_version": SCHEMA_G300_TRAINING,
        "status": "COMPLETE",
        "stage": "G300",
        "run_kind": run_kind,
        "recipe": recipe,
        "seed": seed,
        "script_sha256": _sha256(SCRIPT_PATH),
        "source_role": "G223 controlled-continuation train-fit only",
        "decision_dev_used": False,
        "sealed_or_heldout_read": False,
        "dev_read": False,
        "utility_labels_started": False,
        "formal_training_started": run_kind == "formal",
        "adapter_scope": "draft generation call only",
        "claim_splitter_scope": "frozen Granite base with adapters disabled",
        "true_scope": "frozen verifier; not trained",
        "decode_policy": "greedy at runtime",
        "entry_mode": "controlled_continuation_limited_internal_screen",
        "clean_freeze_ready": False,
        "g223_manifest_sha256": _sha256(data_manifest_path),
        "g223_status": str(manifest.get("status", "")),
        "g223_entry_mode": str(manifest.get("g300_entry_mode", "")),
        "g223_clean_freeze_ready": bool(manifest.get("clean_freeze_ready")),
        "g223_controlled_continuation_ready": bool(
            manifest.get("controlled_continuation_ready")
        ),
        "g223_twowiki_modelval_groups": twowiki_modelval_groups,
        "g223_counts": counts,
        "limitations": {
            "not_clean_freeze": True,
            "twowiki_modelval_screen": twowiki_modelval_groups,
            "heldout_requires_separate_authorization": True,
            "no_selector_utility_labels": True,
        },
        "model_snapshot": str(model_snapshot.resolve()),
        "model_snapshot_config_sha256": _sha256(model_snapshot / "config.json"),
        "init_adapter": str(init_adapter.resolve()) if init_adapter is not None else None,
        "init_adapter_config_sha256": _sha256(init_adapter / "adapter_config.json")
        if init_adapter is not None
        else None,
        "train_cases_sha256": _sha256(train_cases_path),
        "validation_cases_sha256": _sha256(validation_cases_path),
        "ordered_ids_sha256": _sha256(ordered_ids_path),
        "full_data_train_groups": len(full_train_groups),
        "full_data_validation_groups": len(full_validation_groups),
        "smoke_selection": smoke_selection,
        "selected_train_case_ids": [group.case_id for group in selected_train_groups],
        "selected_validation_case_ids": [group.case_id for group in selected_validation_groups],
        "train_groups": len(selected_train_groups),
        "training_examples": train_examples,
        "validation_groups": len(selected_validation_groups),
        "epochs": EPOCHS,
        "max_length": max_length,
        "observed_max_training_length": max(lengths),
        "truncated_examples": 0,
        "truncation_rate": 0.0,
        "microbatch_size": 1,
        "gradient_accumulation_groups": GRADIENT_ACCUMULATION_GROUPS,
        "optimizer_steps": optimizer_steps,
        "learning_rate": _learning_rate(recipe),
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": list(TARGET_MODULES),
            "bias": "none",
        },
        "query_group_equalization": {
            "enabled": True,
            "implementation": (
                "The per-case loss is the mean of its context-variant losses; "
                "gradient accumulation counts groups, not variants."
            ),
        },
        "token_weighting": {
            "prompt": 0,
            "answer_or_punctuation": 1,
            "citation_brackets_or_index": 4,
            "eos": 1,
        },
        "trainable_parameters": trainable_count,
        "all_parameters_with_adapter": all_count,
        "trainable_percent": 100.0 * trainable_count / all_count,
        "mean_group_weighted_loss": _mean(group_losses),
        "final_25_group_weighted_loss": _mean(group_losses[-25:]),
        "mean_answer_token_loss": _mean(category_losses["answer"]),
        "mean_citation_token_loss": _mean(category_losses["citation"]),
        "mean_eos_token_loss": _mean(category_losses["eos"]),
        "validation_loss": validation_loss,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": peak_allocated,
        "adapter_config_sha256": config_sha256,
        "adapter_weights_sha256": weights_sha256,
        "reload_check": {
            "status": "PASS",
            "fresh_base_loaded": True,
            "persisted_adapter_loaded": True,
            "is_trainable": False,
            "all_parameters_with_adapter": reload_parameter_count,
        },
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
    }
    _write_json(output_dir / "training_manifest.json", manifest_out)
    return manifest_out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    lengths = commands.add_parser("audit-lengths")
    lengths.add_argument("--model-snapshot", required=True, type=Path)
    lengths.add_argument("--data-manifest", required=True, type=Path)
    lengths.add_argument("--train-cases", required=True, type=Path)
    lengths.add_argument("--validation-cases", required=True, type=Path)
    lengths.add_argument("--ordered-ids", required=True, type=Path)
    lengths.add_argument("--output-json", required=True, type=Path)
    lengths.add_argument("--max-length", type=int, default=MAX_LENGTH)

    fit = commands.add_parser("train")
    fit.add_argument("--recipe", required=True, choices=("gr-f", "gr-c"))
    fit.add_argument("--run-kind", required=True, choices=("smoke", "formal"))
    fit.add_argument("--seed", required=True, type=int, choices=ALLOWED_SEEDS)
    fit.add_argument("--model-snapshot", required=True, type=Path)
    fit.add_argument("--data-manifest", required=True, type=Path)
    fit.add_argument("--train-cases", required=True, type=Path)
    fit.add_argument("--validation-cases", required=True, type=Path)
    fit.add_argument("--ordered-ids", required=True, type=Path)
    fit.add_argument("--output-dir", required=True, type=Path)
    fit.add_argument("--init-adapter", type=Path)
    fit.add_argument("--max-groups", type=int)
    fit.add_argument("--smoke-selection", choices=("longest",))
    fit.add_argument("--max-length", type=int, default=MAX_LENGTH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit-lengths":
        report = audit_lengths(
            model_snapshot=args.model_snapshot,
            data_manifest_path=args.data_manifest,
            train_cases_path=args.train_cases,
            validation_cases_path=args.validation_cases,
            ordered_ids_path=args.ordered_ids,
            output_json=args.output_json,
            max_length=args.max_length,
        )
    else:
        report = train(
            recipe=args.recipe,
            run_kind=args.run_kind,
            seed=args.seed,
            model_snapshot=args.model_snapshot,
            data_manifest_path=args.data_manifest,
            train_cases_path=args.train_cases,
            validation_cases_path=args.validation_cases,
            ordered_ids_path=args.ordered_ids,
            output_dir=args.output_dir.resolve(),
            init_adapter=args.init_adapter,
            max_groups=args.max_groups,
            smoke_selection=args.smoke_selection,
            max_length=args.max_length,
        )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
