from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_g300_draft_lora_train as g300  # noqa: E402


class _Tokenizer:
    eos_token_id = 99

    def apply_chat_template(
        self,
        messages: object,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return "PROMPT:"

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
    ) -> dict[str, object]:
        assert add_special_tokens is False
        ids = [ord(character) for character in text]
        result: dict[str, object] = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index, _character in enumerate(text)
            ]
        return result


def _case(case_id: str = "q1", *, target_suffix: str = " [2].") -> dict[str, object]:
    return {
        "schema_version": "full-flow-g200-case-v2",
        "case_id": case_id,
        "query_id": case_id,
        "dataset": "niah",
        "role": "train-fit",
        "target_kind": "niah_qa2d_single_claim",
        "answerable": True,
        "variants": {
            "support_only": {
                "prompt": f"prompt-{case_id}-support",
                "target": f"Answer{target_suffix}",
            },
            "topk": {
                "prompt": f"prompt-{case_id}-topk",
                "target": f"Answer{target_suffix}",
            },
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    train = tmp_path / "train_cases.jsonl"
    validation = tmp_path / "validation_cases.jsonl"
    ordered = tmp_path / "ordered_ids.json"
    manifest = tmp_path / "manifest.json"
    _write_jsonl(train, [_case("train-a")])
    _write_jsonl(validation, [_case("val-a")])
    ordered.write_text(
        json.dumps(
            {
                "schema_version": "full-flow-g223-ordered-ids-v1",
                "train_case_ids": ["train-a"],
                "validation_case_ids": ["val-a"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "schema_version": g300.SCHEMA_G223_MANIFEST,
                "status": "CONTROLLED_CONTINUATION_READY",
                "controlled_continuation_ready": True,
                "g300_unlocked": True,
                "g300_entry_mode": "controlled_continuation_limited_internal_screen",
                "clean_freeze_ready": False,
                "training_started": False,
                "utility_labels_started": False,
                "sealed_or_heldout_read": False,
                "dev_read": False,
                "train_cases_sha256": _sha256(train),
                "validation_cases_sha256": _sha256(validation),
                "ordered_ids_sha256": _sha256(ordered),
                "counts": {
                    "2wiki": {"train-modelval_answerable_groups": 95},
                    "niah": {"train-modelval_answerable_groups": 207},
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest, train, validation, ordered


def test_g223_manifest_must_be_controlled_entry_and_hash_bound(tmp_path: Path) -> None:
    manifest, train, validation, ordered = _write_bundle(tmp_path)

    loaded = g300._load_g223_manifest(
        data_manifest_path=manifest,
        train_cases_path=train,
        validation_cases_path=validation,
        ordered_ids_path=ordered,
    )
    assert loaded["g300_unlocked"] is True

    train.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="train cases hash"):
        g300._load_g223_manifest(
            data_manifest_path=manifest,
            train_cases_path=train,
            validation_cases_path=validation,
            ordered_ids_path=ordered,
        )


def test_training_groups_keep_variants_inside_case_group() -> None:
    groups = g300.training_groups([_case("q1")])

    assert len(groups) == 1
    assert groups[0].case_id == "q1"
    assert [variant.name for variant in groups[0].variants] == ["support_only", "topk"]


def test_encode_masks_prompt_and_weights_citation_tokens() -> None:
    encoded = g300.encode_example(
        _Tokenizer(),
        "question",
        "Answer [12].",
        max_length=100,
    )

    prompt_length = len("PROMPT:")
    assert encoded.labels[:prompt_length] == [-100] * prompt_length
    assert encoded.weights[:prompt_length] == [0.0] * prompt_length
    citation_slice = slice(prompt_length + len("Answer "), prompt_length + len("Answer [12]"))
    assert set(encoded.categories[citation_slice]) == {g300.CAT_CITATION}
    assert set(encoded.weights[citation_slice]) == {4.0}
    assert encoded.categories[-1] == g300.CAT_EOS
    assert encoded.weights[-1] == 1.0


def test_encode_rejects_length_instead_of_truncating() -> None:
    with pytest.raises(ValueError, match="exceeds frozen"):
        g300.encode_example(_Tokenizer(), "question", "answer [1].", max_length=10)


def test_smoke_selection_uses_longest_group() -> None:
    short = g300.training_groups([_case("short", target_suffix=" [1].")])[0]
    long = g300.training_groups([_case("long", target_suffix=f" {'x' * 100} [1].")])[0]

    selected = g300.select_longest_groups([short, long], _Tokenizer(), 1)

    assert [group.case_id for group in selected] == ["long"]


def test_recipe_validation_separates_fresh_and_continuation(tmp_path: Path) -> None:
    g300._validate_recipe("gr-f", None)
    with pytest.raises(ValueError, match="without --init-adapter"):
        g300._validate_recipe("gr-f", tmp_path)
    with pytest.raises(ValueError, match="requires --init-adapter"):
        g300._validate_recipe("gr-c", None)
