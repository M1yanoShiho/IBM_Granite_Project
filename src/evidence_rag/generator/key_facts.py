"""Question-slot key-fact notes for the F003B Generator experiment.

The slot detector is deliberately finite and question-only.  It identifies the
*kind* of answer requested (date, place, person, number, or name/title) without
guessing the answer.  Granite then extracts candidate values from the currently
selected evidence before the existing draft→split→verify path runs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from evidence_rag.contracts.models import (
    Query,
    RetainedEvidenceGuidance,
    SelectedEvidenceSet,
    SelectionGuidance,
)
from evidence_rag.generator.granite import TextGenerator
from evidence_rag.generator.json_parsing import parse_json_object

QuestionSlot = Literal["DATE_OR_YEAR", "LOCATION", "PERSON", "NUMBER", "NAME_OR_TITLE"]

_DATE = re.compile(r"\bwhen\b|\bwhat (?:date|year)\b", re.IGNORECASE)
_LOCATION = re.compile(r"\bwhere\b|\bwhat (?:place|location|venue)\b", re.IGNORECASE)
_PERSON = re.compile(r"\bwho\b|\bwhose\b", re.IGNORECASE)
_NUMBER = re.compile(r"\bhow many\b|\bhow much\b|\bwhat (?:number|percentage|amount)\b", re.IGNORECASE)
_NAME = re.compile(r"\bwhich\b|\bwhat\b", re.IGNORECASE)


def detect_question_slots(question: str) -> tuple[QuestionSlot, ...]:
    """Return a small deterministic obligation set without predicting values."""

    slots: list[QuestionSlot] = []
    patterns: tuple[tuple[re.Pattern[str], QuestionSlot], ...] = (
        (_DATE, "DATE_OR_YEAR"),
        (_LOCATION, "LOCATION"),
        (_PERSON, "PERSON"),
        (_NUMBER, "NUMBER"),
        (_NAME, "NAME_OR_TITLE"),
    )
    for pattern, slot in patterns:
        if pattern.search(question) and slot not in slots:
            slots.append(slot)
    # Questions such as "describe..." do not fit a safe finite slot.  Empty is
    # intentional: the experiment falls back to the current Generator.
    return tuple(slots)


@dataclass(frozen=True, slots=True)
class KeyFactNote:
    slot: QuestionSlot
    value: str
    evidence_ids: tuple[str, ...]


NOTES_PROMPT = (
    "Extract short candidate facts needed to answer the question from the evidence.\n"
    "Requested fields: {slots}.\n"
    "Rules:\n"
    "- Copy the most complete value supported by the evidence; do not guess.\n"
    "- For DATE_OR_YEAR, preserve the full date when the evidence gives one.\n"
    "- For LOCATION, answer the place rather than the name of an object located there.\n"
    "- For NUMBER, keep the number and its unit or noun.\n"
    "- evidence must contain only the bracketed evidence numbers shown below.\n"
    "- If a requested field is unavailable, omit it.\n"
    "Return JSON only: "
    '{{"facts":[{{"slot":"DATE_OR_YEAR","value":"1 March 1973","evidence":[2]}}]}}.\n\n'
    "Evidence:\n{context}\n\nQuestion: {question}"
)


def _risk_label(item: RetainedEvidenceGuidance) -> str:
    if item.reason == "PROTECT_HARM_CONFLICT":
        return "protect-harm-conflict"
    if item.reason in {"MAX_DELETE_REACHED", "MIN_KEEP_REACHED"}:
        return "retained-by-conservative-cap"
    if item.reason.startswith("FALLBACK_"):
        return "selector-fallback"
    if item.protect_signal is None or item.harm_signal is None:
        return "unknown"
    if item.protect_signal > item.harm_signal:
        return "protect-leaning"
    if item.harm_signal > item.protect_signal:
        return "harm-leaning"
    return "balanced"


class KeyFactNoteExtractor:
    """One constrained Granite call that precedes the existing draft generator."""

    def __init__(self, llm: TextGenerator, *, guided: bool = False) -> None:
        self.llm = llm
        self.guided = guided

    def extract(
        self,
        query: Query,
        selected: SelectedEvidenceSet,
        guidance: SelectionGuidance | None = None,
    ) -> tuple[KeyFactNote, ...]:
        if query.query_id != selected.query_id:
            raise ValueError("query and selected evidence query IDs differ")
        if guidance is not None and guidance.query_id != query.query_id:
            raise ValueError("query and SelectionGuidance query IDs differ")
        if self.guided and guidance is None:
            raise ValueError("guided notes require SelectionGuidance")
        slots = detect_question_slots(query.text)
        if not slots or not selected.evidence:
            return ()

        guidance_by_id = (
            {item.evidence_id: item for item in guidance.retained}
            if guidance is not None
            else {}
        )
        selected_ids = tuple(item.evidence_id for item in selected.evidence)
        if guidance is not None and tuple(guidance_by_id) != selected_ids:
            raise ValueError("SelectionGuidance does not match selected evidence IDs")

        context_lines: list[str] = []
        for index, evidence in enumerate(selected.evidence, start=1):
            suffix = ""
            if self.guided:
                item = guidance_by_id[evidence.evidence_id]
                suffix = (
                    f" <selector:{_risk_label(item)}; reason:{item.reason}; "
                    f"uncalibrated-protect:{item.protect_signal}; "
                    f"uncalibrated-harm:{item.harm_signal}>"
                )
            context_lines.append(f"[{index}] {evidence.text}{suffix}")
        raw = self.llm.generate(
            NOTES_PROMPT.format(
                slots=", ".join(slots),
                context="\n".join(context_lines),
                question=query.text,
            )
        )
        try:
            data = parse_json_object(raw)
        except ValueError:
            return ()
        facts = data.get("facts")
        if not isinstance(facts, list):
            return ()
        allowed_slots = set(slots)
        output: list[KeyFactNote] = []
        seen_slots: set[str] = set()
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            slot = fact.get("slot")
            value = fact.get("value")
            evidence_indices = fact.get("evidence")
            if slot not in allowed_slots or slot in seen_slots:
                continue
            if not isinstance(value, str) or not value.strip():
                continue
            if not isinstance(evidence_indices, list):
                continue
            evidence_ids: list[str] = []
            for raw_index in evidence_indices:
                if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                    continue
                if 1 <= raw_index <= len(selected.evidence):
                    evidence_id = selected.evidence[raw_index - 1].evidence_id
                    if evidence_id not in evidence_ids:
                        evidence_ids.append(evidence_id)
            if not evidence_ids:
                continue
            seen_slots.add(str(slot))
            output.append(
                KeyFactNote(
                    slot=slot,
                    value=" ".join(value.split()),
                    evidence_ids=tuple(evidence_ids),
                )
            )
        return tuple(output)


def render_key_fact_notes(notes: tuple[KeyFactNote, ...], selected: SelectedEvidenceSet) -> str:
    """Render candidate notes as draft guidance using current evidence numbers."""

    index_by_id = {
        evidence.evidence_id: index
        for index, evidence in enumerate(selected.evidence, start=1)
    }
    rows = []
    for note in notes:
        citations = "".join(
            f"[{index_by_id[evidence_id]}]"
            for evidence_id in note.evidence_ids
            if evidence_id in index_by_id
        )
        rows.append(f"- {note.slot}: {note.value} {citations}".strip())
    return "\n".join(rows)


GUIDED_DRAFT_PROMPT = (
    "Answer the question using only the evidence below.\n"
    "Candidate key-fact notes are a reading aid, not an additional source. Use a note only when "
    "its cited evidence supports it. Make sure the answer supplies the requested field explicitly.\n\n"
    "Key-fact notes:\n{key_fact_notes}\n\n"
    "Write one independently verifiable factual claim per sentence. Keep every sentence "
    "self-contained. End every factual sentence with the bracketed evidence number(s) that "
    "support it. If the evidence does not contain the answer, say: I don't know.\n\n"
    "Evidence:\n{context}\n\nQuestion: {question}\nAnswer:"
)


def notes_manifest(notes: tuple[KeyFactNote, ...]) -> str:
    """Stable JSON for experiment traces; never includes evidence text."""

    return json.dumps(
        [
            {"slot": note.slot, "value": note.value, "evidence_ids": list(note.evidence_ids)}
            for note in notes
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
