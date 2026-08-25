"""Shared generation/output contract for Experiment 05."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidence_rag.contracts.models import (
    REVIEW_ANNOTATION,
    EvidenceCandidate,
    Query,
    SelectedEvidenceSet,
)
from evidence_rag.evaluation.experiment05_runtime import ALL_ARMS, PreparedQuery
from evidence_rag.evaluation.experiment05_scorer import CANONICAL_ABSTENTION
from evidence_rag.generator.draft import DRAFT_PROMPT

DIRECT_ARMS = (
    "bm25_rag",
    "hybrid_rag",
    "granite_rerank_rag",
    "provence_rag",
    "ablation_direct_generator",
)
GROUNDED_ARMS = (
    "ours_seed13",
    "ours_seed42",
    "ours_seed73",
    "ablation_bm25_retriever",
    "ablation_no_selector",
)
OUTPUT_SCHEMA_VERSION: Literal["experiment05.system_output.v1"] = (
    "experiment05.system_output.v1"
)

NonEmpty = Annotated[str, Field(min_length=1)]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SelectedEvidenceRecord(_FrozenModel):
    stage_ordinal: int = Field(ge=1)
    evidence_id: NonEmpty
    artifact_uri: NonEmpty
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    token_count: int = Field(ge=1)
    text: NonEmpty


class PresentedEvidenceRecord(_FrozenModel):
    prompt_ordinal: int = Field(ge=1)
    evidence_id: NonEmpty
    artifact_uri: NonEmpty
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    token_count: int = Field(ge=1)
    text: NonEmpty


class SystemOutput(_FrozenModel):
    schema_version: Literal["experiment05.system_output.v1"] = OUTPUT_SCHEMA_VERSION
    dataset: NonEmpty
    arm_id: NonEmpty
    query_id: NonEmpty
    answer_text: str
    cited_evidence_ids: tuple[str, ...]
    abstained: bool
    runtime_error: str | None
    retrieved_evidence_ids: tuple[NonEmpty, ...]
    selected_evidence_ids: tuple[NonEmpty, ...]
    selected_evidence_records: tuple[SelectedEvidenceRecord, ...]
    presented_evidence_ids: tuple[NonEmpty, ...]
    presented_evidence_records: tuple[PresentedEvidenceRecord, ...]
    model_fingerprints: dict[NonEmpty, NonEmpty]
    config_fingerprint: NonEmpty
    prompt_fingerprint: NonEmpty
    prompt_byte_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_token_count: int = Field(ge=0, le=2304)
    internal_trace_uri: str | None = None

    @model_validator(mode="after")
    def evidence_and_answer_contract(self) -> SystemOutput:
        if self.arm_id not in ALL_ARMS:
            raise ValueError("unknown Experiment 05 arm")
        selected = tuple(item.evidence_id for item in self.selected_evidence_records)
        presented = tuple(item.evidence_id for item in self.presented_evidence_records)
        if selected != self.selected_evidence_ids:
            raise ValueError("selected IDs and records differ")
        if presented != self.presented_evidence_ids:
            raise ValueError("presented IDs and records differ")
        if tuple(item.stage_ordinal for item in self.selected_evidence_records) != tuple(
            range(1, len(selected) + 1)
        ):
            raise ValueError("selected stage ordinals must be consecutive")
        if tuple(item.prompt_ordinal for item in self.presented_evidence_records) != tuple(
            range(1, len(presented) + 1)
        ):
            raise ValueError("presented prompt ordinals must be consecutive")
        if presented != selected[: len(presented)]:
            raise ValueError("presented evidence must be an ordered selected prefix")
        if not set(selected) <= set(self.retrieved_evidence_ids):
            raise ValueError("selected evidence must be retrieved")
        for item in self.selected_evidence_records:
            digest = hashlib.sha256(item.text.encode("utf-8")).hexdigest()
            if digest != item.text_sha256:
                raise ValueError("evidence text hash differs")
        for presented_item in self.presented_evidence_records:
            digest = hashlib.sha256(presented_item.text.encode("utf-8")).hexdigest()
            if digest != presented_item.text_sha256:
                raise ValueError("evidence text hash differs")
        if self.abstained != (self.answer_text == CANONICAL_ABSTENTION):
            raise ValueError("abstained must exactly identify the canonical response")
        return self


def prompt_template(dataset: str) -> str:
    if dataset in {"kilt-nq", "kilt-tqa"}:
        instruction = (
            "Give a direct answer with only the explanation needed. Cite every factual "
            "sentence using the evidence numbers."
        )
    elif dataset == "alce-asqa":
        instruction = (
            "Give a concise long-form answer covering the main facts supported by the "
            "evidence. Cite every factual sentence using the evidence numbers."
        )
    else:
        raise ValueError(f"unsupported Experiment 05 dataset: {dataset}")
    return f"{instruction}\n\n{DRAFT_PROMPT}"


def render_prompt(template: str, query: Query, selected: SelectedEvidenceSet) -> str:
    context = "\n".join(
        f"[{index}] ({item.evidence_id}) {item.text}"
        for index, item in enumerate(selected.evidence, start=1)
    )
    return template.format(
        context=context,
        question=query.text,
        focus="",
        required_facts="none",
        constraints="none",
    )


def selected_evidence_set(prepared: PreparedQuery, arm_id: str) -> SelectedEvidenceSet:
    arm = prepared.arms[arm_id]
    return SelectedEvidenceSet(
        query_id=prepared.query_id,
        evidence=tuple(
            EvidenceCandidate(
                evidence_id=item.evidence_id,
                document_id=item.evidence_id,
                chunk_id="experiment05-prepared",
                text=item.text,
                source_uri=f"sealed://{prepared.dataset}/{prepared.query_id}/{item.evidence_id}",
                retrieval_score=item.retrieval_score,
                retrieval_rank=item.retrieval_rank,
            )
            for item in arm.selected
        ),
    )


def _text_token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return len(input_ids)


def _seal_text(root: Path, text: str) -> tuple[str, str]:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path = root / digest[:2] / f"{digest}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError("content-addressed evidence artifact differs")
    else:
        path.write_text(text, encoding="utf-8")
        path.chmod(0o444)
    return f"sealed://experiment05/evidence/{digest}", digest


def prepare_prompt_evidence(
    *,
    prepared: PreparedQuery,
    arm_id: str,
    query: Query,
    template: str,
    llm: Any,
    artifact_root: Path,
    max_input_tokens: int = 2304,
) -> tuple[
    SelectedEvidenceSet,
    tuple[SelectedEvidenceRecord, ...],
    tuple[PresentedEvidenceRecord, ...],
    str,
    int,
]:
    selected = selected_evidence_set(prepared, arm_id)
    selected_records: list[SelectedEvidenceRecord] = []
    for ordinal, item in enumerate(selected.evidence, start=1):
        uri, digest = _seal_text(artifact_root, item.text)
        selected_records.append(
            SelectedEvidenceRecord(
                stage_ordinal=ordinal,
                evidence_id=item.evidence_id,
                artifact_uri=uri,
                text_sha256=digest,
                token_count=_text_token_count(llm._tokenizer, item.text),
                text=item.text,
            )
        )

    presented = selected
    prompt = render_prompt(template, query, presented)
    while presented.evidence and llm.input_token_count(prompt) > max_input_tokens:
        presented = presented.model_copy(update={"evidence": presented.evidence[:-1]})
        prompt = render_prompt(template, query, presented)
    prompt_tokens = llm.input_token_count(prompt)
    if prompt_tokens > max_input_tokens:
        raise ValueError("question and fixed prompt alone exceed the token budget")
    presented_records = tuple(
        PresentedEvidenceRecord(
            prompt_ordinal=ordinal,
            evidence_id=record.evidence_id,
            artifact_uri=record.artifact_uri,
            text_sha256=record.text_sha256,
            token_count=record.token_count,
            text=record.text,
        )
        for ordinal, record in enumerate(
            selected_records[: len(presented.evidence)], start=1
        )
    )
    return presented, tuple(selected_records), presented_records, prompt, prompt_tokens


_CITATION = re.compile(r"\[(\d+)\]")


def cited_ids(answer: str, presented_ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            presented_ids[int(raw) - 1]
            for raw in _CITATION.findall(answer)
            if 1 <= int(raw) <= len(presented_ids)
        )
    )


def attach_verified_citation(sentence: str, ordinal: int) -> str:
    normalized = " ".join(sentence.split())
    suffix = ""
    if normalized.endswith(REVIEW_ANNOTATION):
        normalized = normalized[: -len(REVIEW_ANNOTATION)].rstrip()
        suffix = f" {REVIEW_ANNOTATION}"
    if normalized and normalized[-1] in ".!?":
        return f"{normalized[:-1].rstrip()} [{ordinal}]{normalized[-1]}{suffix}"
    return f"{normalized} [{ordinal}]{suffix}"


def answer_from_routings(routings: Sequence[Any], presented_ids: Sequence[str]) -> str:
    ordinal = {evidence_id: index for index, evidence_id in enumerate(presented_ids, start=1)}
    sentences: list[str] = []
    for routing in routings:
        sentence = str(getattr(routing, "sentence", "")).strip()
        citation = getattr(routing, "citation", None)
        if not sentence:
            continue
        if getattr(routing, "outcome", None) == "verified" and citation in ordinal:
            sentence = attach_verified_citation(sentence, ordinal[str(citation)])
        sentences.append(sentence)
    return " ".join(sentences).strip()


def build_system_output(
    *,
    prepared: PreparedQuery,
    arm_id: str,
    answer: str,
    runtime_error: str | None,
    selected_records: Sequence[SelectedEvidenceRecord],
    presented_records: Sequence[PresentedEvidenceRecord],
    prompt: str,
    prompt_tokens: int,
    model_fingerprints: Mapping[str, str],
    config_fingerprint: str,
    prompt_fingerprint: str,
    internal_trace_uri: str | None = None,
) -> SystemOutput:
    final_answer = answer.strip() if answer.strip() and runtime_error is None else CANONICAL_ABSTENTION
    presented_ids = tuple(item.evidence_id for item in presented_records)
    return SystemOutput(
        dataset=prepared.dataset,
        arm_id=arm_id,
        query_id=prepared.query_id,
        answer_text=final_answer,
        cited_evidence_ids=cited_ids(final_answer, presented_ids),
        abstained=final_answer == CANONICAL_ABSTENTION,
        runtime_error=runtime_error,
        retrieved_evidence_ids=prepared.arms[arm_id].retrieved_evidence_ids,
        selected_evidence_ids=tuple(item.evidence_id for item in selected_records),
        selected_evidence_records=tuple(selected_records),
        presented_evidence_ids=presented_ids,
        presented_evidence_records=tuple(presented_records),
        model_fingerprints=dict(model_fingerprints),
        config_fingerprint=config_fingerprint,
        prompt_fingerprint=prompt_fingerprint,
        prompt_byte_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        prompt_token_count=prompt_tokens,
        internal_trace_uri=internal_trace_uri,
    )
