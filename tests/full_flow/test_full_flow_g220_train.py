from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_g220_train as g220  # noqa: E402


class _Tokenizer:
    eos_token_id = 99

    def apply_chat_template(
        self, messages: object, *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return "PROMPT:"

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}


def _case() -> dict[str, object]:
    variants = {}
    for index, name in enumerate(g220.VARIANT_NAMES, start=1):
        variants[name] = {
            "prompt": f"prompt-{name}",
            "target": f"The answer is 2009 [{index}].",
        }
    return {
        "schema_version": "full-flow-g200-case-v1",
        "query_id": "q1",
        "variants": variants,
    }


def _write_cases(path: Path) -> None:
    path.write_text(json.dumps(_case()) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_training_examples_keep_equal_slots_targets_and_counts() -> None:
    gc = g220.training_examples([_case()], "gc")
    gm = g220.training_examples([_case()], "gm")

    assert len(gc) == len(gm) == len(g220.VARIANT_NAMES)
    assert [item.slot for item in gc] == [item.slot for item in gm] == list(
        g220.VARIANT_NAMES
    )
    assert {item.source_variant for item in gc} == {"support_only"}
    assert [item.source_variant for item in gm] == list(g220.VARIANT_NAMES)
    assert {g220._strip_citation(item.target) for item in gc + gm} == {
        "The answer is 2009."
    }


def test_encode_example_masks_prompt_and_keeps_target_and_eos() -> None:
    input_ids, labels = g220.encode_example(
        _Tokenizer(), "question", "answer", max_length=20
    )

    prompt_length = len("PROMPT:")
    assert labels[:prompt_length] == [-100] * prompt_length
    assert labels[prompt_length:] == [*[ord(character) for character in "answer"], 99]
    assert input_ids[-1] == 99
    assert g220.example_length(_Tokenizer(), "question", "answer") == len(input_ids)


def test_encode_example_rejects_length_instead_of_truncating() -> None:
    with pytest.raises(ValueError, match="exceeds frozen"):
        g220.encode_example(_Tokenizer(), "question", "answer", max_length=10)


def test_data_manifest_must_be_complete_and_hash_bound(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_cases(train)
    _write_cases(validation)
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "schema_version": "full-flow-g200-data-manifest-v1",
        "status": "PRE_AUDIT",
        "train_cases_sha256": _sha256(train),
        "validation_cases_sha256": _sha256(validation),
        "variants_per_query": len(g220.VARIANT_NAMES),
        "variant_order": list(g220.VARIANT_NAMES),
        "gc_examples": len(g220.VARIANT_NAMES),
        "gm_examples": len(g220.VARIANT_NAMES),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="COMPLETE"):
        g220._load_data_manifest(
            data_manifest_path=manifest_path,
            train_cases_path=train,
            validation_cases_path=validation,
        )

    manifest["status"] = "COMPLETE"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = g220._load_data_manifest(
        data_manifest_path=manifest_path,
        train_cases_path=train,
        validation_cases_path=validation,
    )
    assert loaded["status"] == "COMPLETE"

    train.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="train cases hash"):
        g220._load_data_manifest(
            data_manifest_path=manifest_path,
            train_cases_path=train,
            validation_cases_path=validation,
        )
