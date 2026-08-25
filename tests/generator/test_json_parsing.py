import pytest

from evidence_rag.generator.json_parsing import parse_json_object


def test_parses_clean_object() -> None:
    assert parse_json_object('{"covered": true}') == {"covered": True}


def test_tolerates_trailing_prose() -> None:
    # the exact real-Granite shape that crashed the G2 run: valid JSON + explanation
    raw = '{"claims": [{"source_text": "x", "text": "y"}]} Here is why I split it that way.'
    assert parse_json_object(raw) == {"claims": [{"source_text": "x", "text": "y"}]}


def test_tolerates_leading_prose() -> None:
    assert parse_json_object('Sure! {"found": false}') == {"found": False}


def test_strips_code_fence() -> None:
    raw = '```json\n{"covered": false, "gap_question": "what?"}\n```'
    assert parse_json_object(raw) == {"covered": False, "gap_question": "what?"}


def test_rejects_malformed_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        parse_json_object("not json at all")


def test_rejects_truncated_object() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        parse_json_object('{"covered": tru')


def test_rejects_non_object_json() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        parse_json_object("[1, 2, 3]")
