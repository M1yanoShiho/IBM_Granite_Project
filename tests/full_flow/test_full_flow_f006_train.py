from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_f006_train as train  # noqa: E402


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


def test_training_examples_match_counts_and_only_context_differs() -> None:
    cases = [
        {
            "query_id": "q1",
            "clean_prompt": "clean",
            "mixed_prompt": "mixed",
            "clean_target": "clean-json",
            "mixed_target": "mixed-json",
        }
    ]

    clean = train.training_examples(cases, "clean")
    mixed = train.training_examples(cases, "mixed")

    assert clean == [("q1", "clean", "clean-json"), ("q1", "clean", "clean-json")]
    assert mixed == [("q1", "clean", "clean-json"), ("q1", "mixed", "mixed-json")]
    assert len(clean) == len(mixed) == 2


def test_encode_example_masks_prompt_and_keeps_target_and_eos() -> None:
    input_ids, labels = train.encode_example(_Tokenizer(), "question", "{}", max_length=20)

    prompt_length = len("PROMPT:")
    assert labels[:prompt_length] == [-100] * prompt_length
    assert labels[prompt_length:] == [ord("{"), ord("}"), 99]
    assert input_ids[-1] == 99
