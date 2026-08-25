"""Formal seven-arm runtime and post-generation contracts for Experiment 04 Goal 3."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidence_rag.contracts.models import (
    CandidateSet,
    Document,
    EvidenceCandidate,
    Query,
    SelectedEvidenceSet,
    strip_annotations,
)
from evidence_rag.evaluation.citation_metrics import CitationExample, score_citation_examples
from evidence_rag.evaluation.sealed_runtime import (
    file_sha256,
    ordered_id_sha256,
    validate_runtime_record,
)
from evidence_rag.evaluation.system_scorer import score_bundle
from evidence_rag.infrastructure.corpus import CorpusBuilder, PrechunkedChunker
from evidence_rag.retriever.fusion import reciprocal_rank_fusion
from evidence_rag.retriever.granite import GraniteDenseRetriever, TextEmbedder
from evidence_rag.retriever.rerank import PairReranker
from evidence_rag.retriever.strong_bm25 import StrongBM25Retriever
from evidence_rag.selector.nli_runtime import NliRiskControlledSelector
from evidence_rag.selector.provence import PassagePruner

PRIMARY_ARMS = (
    "dense_rag",
    "hybrid_rag",
    "granite_rerank_rag",
    "provence_rag",
    "ours_seed13",
    "ours_seed42",
    "ours_seed73",
)
BASELINE_ARMS = PRIMARY_ARMS[:4]
OURS_ARMS = PRIMARY_ARMS[4:]
PREPARED_SCHEMA_VERSION: Literal["experiment04.prepared.v1"] = "experiment04.prepared.v1"
SYSTEM_OUTPUT_SCHEMA_VERSION: Literal["experiment04.system_output.v1"] = (
    "experiment04.system_output.v1"
)
GENERATION_MANIFEST_SCHEMA_VERSION = "experiment04.generation_manifest.v1"

NonEmpty = Annotated[str, Field(min_length=1)]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PreparedEvidence(FrozenModel):
    source_id: NonEmpty
    text: NonEmpty
    unit_ids: tuple[NonEmpty, ...]
    retrieval_score: float
    retrieval_rank: Annotated[int, Field(ge=1)]


class PreparedArm(FrozenModel):
    retrieved_source_ids: tuple[NonEmpty, ...]
    retrieved_unit_ids: tuple[NonEmpty, ...]
    selected: tuple[PreparedEvidence, ...]
    selection_status: Literal["normal", "fail_open_all"] = "normal"

    @model_validator(mode="after")
    def selected_is_local_to_retrieval(self) -> PreparedArm:
        retrieved_sources = set(self.retrieved_source_ids)
        retrieved_units = set(self.retrieved_unit_ids)
        selected_sources = [item.source_id for item in self.selected]
        selected_units = [unit_id for item in self.selected for unit_id in item.unit_ids]
        if len(selected_sources) != len(set(selected_sources)):
            raise ValueError("prepared selection contains duplicate source IDs")
        if not set(selected_sources) <= retrieved_sources:
            raise ValueError("prepared selected sources must be retrieved")
        if not set(selected_units) <= retrieved_units:
            raise ValueError("prepared selected units must be retrieved")
        return self


class PreparedQuery(FrozenModel):
    schema_version: Literal["experiment04.prepared.v1"]
    dataset: NonEmpty
    query_id: NonEmpty
    arms: dict[NonEmpty, PreparedArm]

    @model_validator(mode="after")
    def exact_primary_arm_set(self) -> PreparedQuery:
        if set(self.arms) != set(PRIMARY_ARMS):
            raise ValueError("prepared query does not contain exactly seven primary arms")
        ours = [self.arms[arm].model_dump_json() for arm in OURS_ARMS]
        if len(set(ours)) != 1:
            raise ValueError("the three Ours seeds must share one frozen upstream preparation")
        return self


class CitationSentence(FrozenModel):
    sentence: NonEmpty
    source_ids: tuple[NonEmpty, ...]


class SystemOutput(FrozenModel):
    schema_version: Literal["experiment04.system_output.v1"]
    dataset: NonEmpty
    arm_id: NonEmpty
    query_id: NonEmpty
    retrieved_unit_ids: tuple[NonEmpty, ...]
    selected_unit_ids: tuple[NonEmpty, ...]
    selected_source_ids: tuple[NonEmpty, ...]
    answer: str
    citation_indices: tuple[int, ...]
    citation_sentences: tuple[CitationSentence, ...]
    failure_stage: Literal["retrieval", "selection", "generation", "system"] | None
    error_code: str | None

    @model_validator(mode="after")
    def validate_output_contract(self) -> SystemOutput:
        if self.arm_id not in PRIMARY_ARMS:
            raise ValueError("system output arm is not a Goal 3 primary arm")
        if not set(self.selected_unit_ids) <= set(self.retrieved_unit_ids):
            raise ValueError("selected_unit_ids must be a subset of retrieved_unit_ids")
        if self.failure_stage is None and self.error_code is not None:
            raise ValueError("error_code requires a failure_stage")
        if self.failure_stage is not None and not self.error_code:
            raise ValueError("failure_stage requires a machine-readable error_code")
        if any(index < 1 for index in self.citation_indices):
            raise ValueError("citation indices must be positive")
        return self


def canonical_json_bytes(value: Mapping[str, Any] | BaseModel) -> bytes:
    materialized = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(
        materialized,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def append_canonical_jsonl(path: Path, value: Mapping[str, Any] | BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(value))
        handle.write(b"\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            rows.append(value)
    return rows


def maximal_whole_evidence_prefix(
    selected: SelectedEvidenceSet,
    *,
    render_prompt: Callable[[SelectedEvidenceSet], str],
    input_token_count: Callable[[str], int],
    max_input_tokens: int,
) -> SelectedEvidenceSet:
    """Fit the longest rank prefix without truncating or skipping an evidence unit.

    Experiment 04 freezes a hard chat-input ceiling and forbids tokenizer truncation.  The
    only admissible mechanical overflow repair is therefore a deterministic prefix of complete
    evidence units.  Empty evidence remains a valid result when even the first complete unit
    cannot fit; the Generator already treats that case as an unanswered query without a runtime
    exception.
    """

    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive")
    for end in range(len(selected.evidence), 0, -1):
        candidate = selected.model_copy(update={"evidence": selected.evidence[:end]})
        if input_token_count(render_prompt(candidate)) <= max_input_tokens:
            return candidate
    return selected.model_copy(update={"evidence": ()})


def ordered_prefix_count(
    path: Path,
    expected_query_ids: Sequence[str],
    validator: Callable[[Mapping[str, Any]], Any],
) -> int:
    """Validate a resumable JSONL file as an exact ordered prefix."""

    if not path.exists():
        return 0
    rows = read_jsonl(path)
    if len(rows) > len(expected_query_ids):
        raise ValueError("resume file is longer than the frozen query sequence")
    observed: list[str] = []
    for row in rows:
        validator(row)
        observed.append(str(row.get("query_id", "")))
    if observed != list(expected_query_ids[: len(observed)]):
        raise ValueError("resume file is not an exact ordered prefix")
    if len(observed) != len(set(observed)):
        raise ValueError("resume file contains duplicate query IDs")
    return len(observed)


def _source_text(candidate: Mapping[str, Any]) -> str:
    title = str(candidate["title"]).strip()
    body = str(candidate["text"]).strip()
    return f"{title}\n{body}" if title else body


def _truncate(candidates: CandidateSet, limit: int) -> CandidateSet:
    return CandidateSet(query_id=candidates.query_id, candidates=candidates.candidates[:limit])


def _rerank(
    query: Query,
    candidates: CandidateSet,
    reranker: PairReranker,
    *,
    top_k: int,
) -> CandidateSet:
    scores = tuple(float(value) for value in reranker.score(
        query.text,
        tuple(item.text for item in candidates.candidates),
    ))
    if len(scores) != len(candidates.candidates):
        raise ValueError("reranker returned a different number of scores than passages")
    ranked = sorted(
        zip(scores, candidates.candidates, strict=True),
        key=lambda item: (-item[0], item[1].retrieval_rank, item[1].evidence_id),
    )
    return CandidateSet(
        query_id=query.query_id,
        candidates=tuple(
            candidate.model_copy(
                update={"retrieval_score": score, "retrieval_rank": rank}
            )
            for rank, (score, candidate) in enumerate(ranked[:top_k], start=1)
        ),
    )


_ALIGN_TOKEN = re.compile(r"[\w]+", flags=re.UNICODE)


def _alignment_tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _ALIGN_TOKEN.finditer(text))


def _contains_sequence(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(tuple(haystack[index : index + width]) == tuple(needle) for index in range(len(haystack) - width + 1))


def retained_unit_ids_after_pruning(
    candidate: Mapping[str, Any],
    pruned_text: str,
) -> tuple[str, ...]:
    """Align official Provence output back to frozen scoreable units.

    Hotpot units are sentences, so a retained unit must reappear as a complete
    token sequence. MuSiQue/RGB expose one paragraph/document unit; for those a
    non-title body fragment means the unit remains available to the Generator.
    """

    units = list(candidate["units"])
    pruned_tokens = list(_alignment_tokens(pruned_text))
    title_tokens = list(_alignment_tokens(str(candidate["title"])))
    if title_tokens and pruned_tokens[: len(title_tokens)] == title_tokens:
        pruned_tokens = pruned_tokens[len(title_tokens) :]
    if not pruned_tokens:
        return ()
    if len(units) == 1:
        body_tokens = set(_alignment_tokens(str(units[0]["text"])))
        return (str(units[0]["unit_id"]),) if body_tokens & set(pruned_tokens) else ()
    retained: list[str] = []
    for unit in units:
        unit_tokens = _alignment_tokens(str(unit["text"]))
        if _contains_sequence(pruned_tokens, unit_tokens):
            retained.append(str(unit["unit_id"]))
    return tuple(retained)


def _source_index(record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(candidate["source_id"]): candidate for candidate in record["candidates"]}


def _unit_ids(
    sources: Mapping[str, Mapping[str, Any]],
    source_ids: Iterable[str],
) -> tuple[str, ...]:
    return tuple(
        str(unit["unit_id"])
        for source_id in source_ids
        for unit in sources[source_id]["units"]
    )


def _prepared_evidence(
    candidate: EvidenceCandidate,
    *,
    text: str | None = None,
    unit_ids: Sequence[str],
) -> PreparedEvidence:
    return PreparedEvidence(
        source_id=candidate.document_id,
        text=text if text is not None else candidate.text,
        unit_ids=tuple(unit_ids),
        retrieval_score=float(candidate.retrieval_score),
        retrieval_rank=candidate.retrieval_rank,
    )


def _prepared_arm(
    retrieved: CandidateSet,
    selected: Sequence[PreparedEvidence],
    sources: Mapping[str, Mapping[str, Any]],
    *,
    selection_status: Literal["normal", "fail_open_all"] = "normal",
) -> PreparedArm:
    source_ids = tuple(candidate.document_id for candidate in retrieved.candidates)
    return PreparedArm(
        retrieved_source_ids=source_ids,
        retrieved_unit_ids=_unit_ids(sources, source_ids),
        selected=tuple(selected),
        selection_status=selection_status,
    )


def prepare_query(
    record: Mapping[str, Any],
    *,
    embedder: TextEmbedder,
    reranker: PairReranker,
    provence: PassagePruner,
    selector: NliRiskControlledSelector,
) -> PreparedQuery:
    """Run the frozen retrieval/selection stage on one query-local candidate pool."""

    validate_runtime_record(record)
    dataset = str(record["dataset"])
    query = Query(query_id=str(record["query_id"]), text=str(record["question"]))
    sources = _source_index(record)
    documents = tuple(
        Document(
            document_id=source_id,
            text=_source_text(candidate),
            source_uri=f"sealed://{dataset}/{query.query_id}/{source_id}",
        )
        for source_id, candidate in sources.items()
    )
    corpus = CorpusBuilder(PrechunkedChunker()).build(
        documents,
        dataset_signature=f"experiment04:{dataset}:{query.query_id}",
    )
    pool_size = min(40, len(documents))
    dense_retriever = GraniteDenseRetriever.from_corpus(corpus, embedder=embedder)
    sparse_retriever = StrongBM25Retriever.from_corpus(corpus, k1=0.9, b=0.4)
    dense_pool = dense_retriever.retrieve(query, pool_size)
    sparse_pool = sparse_retriever.retrieve(query, pool_size)
    dense10 = _truncate(dense_pool, min(10, pool_size))
    hybrid10 = reciprocal_rank_fusion(
        (_truncate(sparse_pool, min(10, pool_size)), dense10),
        query_id=query.query_id,
        top_k=min(10, pool_size),
        k=60,
    )
    hybrid40 = reciprocal_rank_fusion(
        (sparse_pool, dense_pool),
        query_id=query.query_id,
        top_k=pool_size,
        k=60,
    )
    reranked10 = _rerank(query, hybrid40, reranker, top_k=min(10, pool_size))

    def keep_all(candidates: CandidateSet) -> tuple[PreparedEvidence, ...]:
        return tuple(
            _prepared_evidence(
                candidate,
                unit_ids=_unit_ids(sources, (candidate.document_id,)),
            )
            for candidate in candidates.candidates
        )

    provence_selected: list[PreparedEvidence] = []
    for candidate in hybrid10.candidates:
        source = sources[candidate.document_id]
        pruned = provence.prune(
            question=query.text,
            title=str(source["title"]),
            text=str(source["text"]),
        )
        if not pruned:
            continue
        provence_selected.append(
            _prepared_evidence(
                candidate,
                text=pruned,
                unit_ids=retained_unit_ids_after_pruning(source, pruned),
            )
        )

    selection_status: Literal["normal", "fail_open_all"] = "normal"
    try:
        selected_result, _trace = selector.select_with_trace(query, hybrid10, 10)
        selected_ids = {item.evidence_id for item in selected_result.items}
        nli_candidates = tuple(
            candidate for candidate in hybrid10.candidates if candidate.evidence_id in selected_ids
        )
    except Exception:  # noqa: BLE001 - the frozen policy explicitly fails open
        selection_status = "fail_open_all"
        nli_candidates = hybrid10.candidates
    ours_selected = tuple(
        _prepared_evidence(
            candidate,
            unit_ids=_unit_ids(sources, (candidate.document_id,)),
        )
        for candidate in nli_candidates
    )

    arms = {
        "dense_rag": _prepared_arm(dense10, keep_all(dense10), sources),
        "hybrid_rag": _prepared_arm(hybrid10, keep_all(hybrid10), sources),
        "granite_rerank_rag": _prepared_arm(
            reranked10,
            keep_all(reranked10),
            sources,
        ),
        "provence_rag": _prepared_arm(hybrid10, provence_selected, sources),
        **{
            arm: _prepared_arm(
                hybrid10,
                ours_selected,
                sources,
                selection_status=selection_status,
            )
            for arm in OURS_ARMS
        },
    }
    return PreparedQuery(
        schema_version=PREPARED_SCHEMA_VERSION,
        dataset=dataset,
        query_id=query.query_id,
        arms=arms,
    )


def selected_evidence_set(prepared: PreparedQuery, arm_id: str) -> SelectedEvidenceSet:
    arm = prepared.arms[arm_id]
    return SelectedEvidenceSet(
        query_id=prepared.query_id,
        evidence=tuple(
            EvidenceCandidate(
                evidence_id=item.source_id,
                document_id=item.source_id,
                chunk_id="prepared",
                text=item.text,
                source_uri=f"sealed://{prepared.dataset}/{prepared.query_id}/{item.source_id}",
                retrieval_score=item.retrieval_score,
                retrieval_rank=item.retrieval_rank,
            )
            for item in arm.selected
        ),
    )


_INLINE_GROUP = re.compile(r"\[((?:\s*\d+\s*,?\s*)+)\]")
_FALLBACK_SENTENCE = re.compile(r"[^.!?]*[.!?]+|\S[^.!?]*$")


def _citation_numbers(text: str) -> tuple[int, ...]:
    return tuple(
        dict.fromkeys(
            int(value)
            for group in _INLINE_GROUP.findall(text)
            for value in re.findall(r"\d+", group)
        )
    )


def _sentences(text: str) -> tuple[str, ...]:
    try:
        from nltk.tokenize import sent_tokenize  # type: ignore[import-not-found]

        return tuple(sentence.strip() for sentence in sent_tokenize(text) if sentence.strip())
    except (ImportError, LookupError):
        return tuple(
            match.group().strip()
            for match in _FALLBACK_SENTENCE.finditer(text)
            if match.group().strip()
        )


def citation_sentences_from_inline(
    raw_answer: str,
    selected_source_ids: Sequence[str],
) -> tuple[tuple[CitationSentence, ...], tuple[int, ...]]:
    declared = _citation_numbers(raw_answer)
    rows: list[CitationSentence] = []
    for raw_sentence in _sentences(raw_answer):
        indices = _citation_numbers(raw_sentence)
        source_ids = tuple(
            dict.fromkeys(
                selected_source_ids[index - 1]
                for index in indices
                if 1 <= index <= len(selected_source_ids)
            )
        )
        sentence = " ".join(_INLINE_GROUP.sub(" ", raw_sentence).split())
        if sentence:
            rows.append(CitationSentence(sentence=sentence, source_ids=source_ids))
    return tuple(rows), declared


def citation_sentences_from_routings(
    routings: Iterable[Any],
    selected_source_ids: Sequence[str],
) -> tuple[CitationSentence, ...]:
    selected = set(selected_source_ids)
    rows: list[CitationSentence] = []
    for routing in routings:
        sentence = strip_annotations(str(getattr(routing, "sentence", ""))).strip()
        if not sentence:
            continue
        citation = getattr(routing, "citation", None)
        references = (str(citation),) if citation is not None and str(citation) in selected else ()
        rows.append(CitationSentence(sentence=sentence, source_ids=references))
    return tuple(rows)


def system_output(
    *,
    prepared: PreparedQuery,
    arm_id: str,
    selected_evidence: SelectedEvidenceSet | None = None,
    answer: str,
    citation_indices: Sequence[int],
    citation_sentences: Sequence[CitationSentence],
    failure_stage: Literal["retrieval", "selection", "generation", "system"] | None = None,
    error_code: str | None = None,
) -> SystemOutput:
    arm = prepared.arms[arm_id]
    prepared_by_source = {item.source_id: item for item in arm.selected}
    selected_source_ids = (
        tuple(item.evidence_id for item in selected_evidence.evidence)
        if selected_evidence is not None
        else tuple(prepared_by_source)
    )
    if not set(selected_source_ids) <= set(prepared_by_source):
        raise ValueError("actual generation context must be a subset of prepared selection")
    return SystemOutput(
        schema_version=SYSTEM_OUTPUT_SCHEMA_VERSION,
        dataset=prepared.dataset,
        arm_id=arm_id,
        query_id=prepared.query_id,
        retrieved_unit_ids=arm.retrieved_unit_ids,
        selected_unit_ids=tuple(
            unit_id
            for source_id in selected_source_ids
            for unit_id in prepared_by_source[source_id].unit_ids
        ),
        selected_source_ids=selected_source_ids,
        answer=answer,
        citation_indices=tuple(citation_indices),
        citation_sentences=tuple(citation_sentences),
        failure_stage=failure_stage,
        error_code=error_code,
    )


def freeze_generation_manifest(
    *,
    dataset: str,
    arm_paths: Mapping[str, Path],
    expected_query_ids: Sequence[str],
    runtime_sha256: str,
    attempt_id: str,
) -> dict[str, Any]:
    if set(arm_paths) != set(PRIMARY_ARMS):
        raise ValueError("generation manifest requires exactly seven primary arms")
    arms: dict[str, Any] = {}
    bundle_valid = True
    for arm_id in PRIMARY_ARMS:
        path = arm_paths[arm_id]
        rows = read_jsonl(path)
        parsed = [SystemOutput.model_validate(row) for row in rows]
        observed_ids = [row.query_id for row in parsed]
        if observed_ids != list(expected_query_ids):
            raise ValueError(f"{arm_id} does not cover the exact frozen ordered IDs")
        if any(row.dataset != dataset or row.arm_id != arm_id for row in parsed):
            raise ValueError(f"{arm_id} output identity differs")
        errors = sum(row.failure_stage is not None for row in parsed)
        error_rate = errors / len(parsed)
        valid = error_rate <= 0.01
        bundle_valid &= valid
        arms[arm_id] = {
            "count": len(parsed),
            "ordered_ids_sha256": ordered_id_sha256(observed_ids),
            "file_sha256": file_sha256(path),
            "runtime_errors": errors,
            "runtime_error_rate": error_rate,
            "valid_at_1pct_guard": valid,
        }
    return {
        "schema_version": GENERATION_MANIFEST_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "dataset": dataset,
        "runtime_sha256": runtime_sha256,
        "query_count": len(expected_query_ids),
        "ordered_ids_sha256": ordered_id_sha256(expected_query_ids),
        "arms": arms,
        "outputs_frozen_before_scoring": True,
        "bundle_status": "PASS" if bundle_valid else "INVALID_REQUIRES_FULL_RERUN",
    }


def score_frozen_arm(
    *,
    sidecar_records: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    prepared_records: Sequence[Mapping[str, Any]],
    arm_id: str,
    entails: Callable[[str, str], bool],
) -> tuple[dict[str, Any], int]:
    parsed_outputs = [SystemOutput.model_validate(row) for row in outputs]
    prepared = [PreparedQuery.model_validate(row) for row in prepared_records]
    if [row.query_id for row in parsed_outputs] != [row.query_id for row in prepared]:
        raise ValueError("prepared and frozen output ordered IDs differ")
    examples: list[CitationExample] = []
    for output, prep in zip(parsed_outputs, prepared, strict=True):
        if output.arm_id != arm_id:
            raise ValueError("frozen output contains the wrong arm")
        legal = bool(output.citation_indices) and all(
            1 <= index <= len(output.selected_source_ids)
            for index in output.citation_indices
        )
        if output.failure_stage is not None or not output.answer.strip() or not legal:
            continue
        selected_sources = set(output.selected_source_ids)
        documents = {
            item.source_id: item.text
            for item in prep.arms[arm_id].selected
            if item.source_id in selected_sources
        }
        examples.append(
            CitationExample(
                example_id=output.query_id,
                sentences=tuple(item.sentence for item in output.citation_sentences),
                citations=tuple(item.source_ids for item in output.citation_sentences),
                documents=documents,
            )
        )
    citation_scores: dict[str, dict[str, float | int]] = {}
    scoring_failures: set[str] = set()
    judge_calls = 0
    for example in examples:
        try:
            per_example, calls = score_citation_examples((example,), entails)
        except Exception:  # noqa: BLE001 - scorer failures stay in the denominator
            scoring_failures.add(example.example_id)
            continue
        citation_scores.update(per_example)
        judge_calls += calls
    scorer_outcomes: list[dict[str, Any]] = []
    for output in parsed_outputs:
        citation = citation_scores.get(
            output.query_id,
            {"precision": 0.0, "recall": 0.0},
        )
        scorer_outcomes.append(
            {
                "query_id": output.query_id,
                "retrieved_unit_ids": list(output.retrieved_unit_ids),
                "selected_unit_ids": list(output.selected_unit_ids),
                "selected_source_ids": list(output.selected_source_ids),
                "answer": strip_annotations(output.answer),
                "citation_indices": list(output.citation_indices),
                "failure_stage": (
                    output.failure_stage
                    if output.failure_stage is not None
                    else "scoring"
                    if output.query_id in scoring_failures
                    else None
                ),
                "minicheck": {
                    "precision": float(citation["precision"]),
                    "recall": float(citation["recall"]),
                },
            }
        )
    return score_bundle(sidecar_records, scorer_outcomes), judge_calls


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_component_cluster_bootstrap(
    *,
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    component_ids: Mapping[str, str],
    resamples: int = 10_000,
    seed: int = 13,
) -> dict[str, Any]:
    if set(candidate) != set(baseline) or set(candidate) != set(component_ids):
        raise ValueError("paired bootstrap inputs must cover the same query IDs")
    query_ids = tuple(candidate)
    by_component: dict[str, list[str]] = {}
    for query_id in query_ids:
        by_component.setdefault(component_ids[query_id], []).append(query_id)
    components = tuple(sorted(by_component))
    if not components:
        raise ValueError("paired bootstrap requires at least one component")
    differences = {
        query_id: float(candidate[query_id]) - float(baseline[query_id])
        for query_id in query_ids
    }
    rng = random.Random(seed)
    replicates: list[float] = []
    for _ in range(resamples):
        sampled = [components[rng.randrange(len(components))] for _ in components]
        values = [differences[query_id] for component in sampled for query_id in by_component[component]]
        replicates.append(sum(values) / len(values))
    point = sum(differences.values()) / len(differences)
    return {
        "method": "paired_component_cluster_percentile_bootstrap",
        "seed": seed,
        "resamples": resamples,
        "n_queries": len(query_ids),
        "n_components": len(components),
        "difference": point,
        "ci95_low": _quantile(replicates, 0.025),
        "ci95_high": _quantile(replicates, 0.975),
    }


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
