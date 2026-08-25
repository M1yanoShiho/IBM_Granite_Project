"""Tolerant JSON-object extraction for LLM responses.

Every LLM-driven step in the Generator (claim splitting, completeness, evidence
recheck) prompts for "JSON only", but a real Granite model routinely returns a
valid JSON object *followed by an explanation* -- which a strict ``json.loads``
rejects with "Extra data", aborting the whole verify->repair chain on the first
query (surfaced by the G2 real-data run; the test fakes returned clean JSON and
never exercised this). A grounding pipeline has to read the object and ignore the
model's trailing prose. Genuinely malformed JSON still raises, so a broken
response is not silently accepted.
"""

import json
from typing import Any

_DECODER = json.JSONDecoder()


def parse_json_object(raw: str) -> dict[str, Any]:
    """Return the first complete JSON object in ``raw``.

    Tolerates a leading ```json / ``` code fence and any text after the object;
    raises ``ValueError`` (same message the strict parsers used) if no
    well-formed object is present.
    """
    text = raw.strip()
    if text.startswith("```"):
        # drop the opening fence line (``` or ```json ...) and a trailing fence
        newline = text.find("\n")
        text = text[newline + 1 :] if newline != -1 else ""
        stripped = text.rstrip()
        if stripped.endswith("```"):
            text = stripped[:-3]
        text = text.strip()
    # start at the first JSON value opener; an array is decoded so a top-level
    # array still reports "must be a JSON object" (the strict parsers' behaviour)
    candidates = [pos for pos in (text.find("{"), text.find("[")) if pos != -1]
    if not candidates:
        raise ValueError("LLM output must be valid JSON")
    try:
        data, _end = _DECODER.raw_decode(text, min(candidates))
    except json.JSONDecodeError as exc:
        raise ValueError("LLM output must be valid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("LLM output must be a JSON object")
    return data
