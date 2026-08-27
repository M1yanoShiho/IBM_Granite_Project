"""Unified 10-arm runner contract for Experiment 04."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidence_rag.contracts.models import (
    GenerationResult,
    Query,
    SelectedEvidenceSet,
)
from evidence_rag.contracts.protocols import Generator, Retriever, Selector
from evidence_rag.contracts.validation import resolve_selection, validate_generation
from evidence_rag.evaluation.system_scorer import score_bundle
from evidence_rag.query_analysis import QueryAnalyzer, RuleBasedQueryAnalyzer
from evidence_rag.selector.provence import ProvenceSelector

NonEmpty = Annotated[str, Field(min_length=1)]

ARM_MATRIX: dict[str, tuple[str, str, str, int | None]] = {
    "dense_rag": ("dense", "keep-all", "direct", None),
    "hybrid_rag": ("hybrid", "keep-all", "direct", None),
    "granite_rerank_rag": ("granite-rerank", "keep-all", "direct", None),
    "provence_rag": ("hybrid", "provence", "direct", None),
    "ours_seed13": ("hybrid", "nli-risk-controlled", "grounded-grc", 13),
    "ours_seed42": ("hybrid", "nli-risk-controlled", "grounded-grc", 42),
    "ours_seed73": ("hybrid", "nli-risk-controlled", "grounded-grc", 73),
    "ablation_dense_retriever": ("dense", "nli-risk-controlled", "grounded-grc", 13),
    "ablation_top10": ("hybrid", "keep-all", "grounded-grc", 13),
    "ablation_direct_generator": ("hybrid", "nli-risk-controlled", "direct", None),
}


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class ModuleIdentity(FrozenModel):
    retriever: Literal["dense", "hybrid", "granite-rerank"]
    selector: Literal["keep-all", "nli-risk-controlled", "provence"]
    generator: Literal["direct", "grounded-grc"]
    generator_seed: Literal[13, 42, 73] | None = None


class RunBudget(FrozenModel):
    final_top_k: Literal[10]
    hybrid_pool_size: Literal[10]
    reranker_pool_size: Literal[40]
    max_selected: Literal[10]
    generator_max_input_tokens: Literal[2304]
    generator_max_new_tokens: Literal[256]
    temperature: float
    top_p: float
    do_sample: Literal[False]
    overflow_policy: Literal["error_no_silent_truncation"]

    @model_validator(mode="after")
    def decoding_values_are_frozen(self) -> RunBudget:
        if self.temperature != 0.0 or self.top_p != 1.0:
            raise ValueError("Experiment 04 freezes temperature=0.0 and top_p=1.0")
        return self


class FormatContract(FrozenModel):
    context_template: Literal["[{index}] ({evidence_id}) {text}"]
    citation_format: Literal["inline_square_bracket_1_based"]
    prompt_id: Literal["experiment04_shared_inline_citation_v1"]
    output_schema: Literal["experiment04.system_output.v1"]


class ArmConfig(FrozenModel):
    schema_version: Literal["experiment04.arm.v1"]
    arm_id: NonEmpty
    table_role: Literal["baseline", "ours", "ablation"]
    model_manifest: NonEmpty
    development_fixture: NonEmpty
    modules: ModuleIdentity
    budget: RunBudget
    format: FormatContract

    @model_validator(mode="after")
    def identity_matches_frozen_matrix(self) -> ArmConfig:
        expected = ARM_MATRIX.get(self.arm_id)
        if expected is None:
            raise ValueError(f"unknown Experiment 04 arm_id: {self.arm_id}")
        observed = (
            self.modules.retriever,
            self.modules.selector,
            self.modules.generator,
            self.modules.generator_seed,
        )
        if observed != expected:
            raise ValueError(f"arm {self.arm_id} differs from the frozen module identity")
        role = (
            "baseline"
            if self.arm_id in {
                "dense_rag",
                "hybrid_rag",
                "granite_rerank_rag",
                "provence_rag",
            }
            else "ours"
            if self.arm_id.startswith("ours_")
            else "ablation"
        )
        if self.table_role != role:
            raise ValueError(f"arm {self.arm_id} has the wrong table_role")
        return self


def load_arm_config(path: Path) -> ArmConfig:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"invalid Experiment 04 arm config {path}: {exc}") from exc
    return ArmConfig.model_validate(raw)


def load_arm_matrix(directory: Path) -> dict[str, ArmConfig]:
    configs: dict[str, ArmConfig] = {}
    for path in sorted(directory.glob("*.toml")):
        config = load_arm_config(path)
        if config.arm_id in configs:
            raise ValueError(f"duplicate Experiment 04 arm_id: {config.arm_id}")
        configs[config.arm_id] = config
    if set(configs) != set(ARM_MATRIX):
        raise ValueError(
            f"Experiment 04 arm set differs: missing={sorted(set(ARM_MATRIX) - set(configs))}, "
            f"unexpected={sorted(set(configs) - set(ARM_MATRIX))}"
        )
    shared_budgets = {config.budget.model_dump_json() for config in configs.values()}
    shared_formats = {config.format.model_dump_json() for config in configs.values()}
    shared_manifests = {config.model_manifest for config in configs.values()}
    shared_fixtures = {config.development_fixture for config in configs.values()}
    if len(shared_budgets) != 1 or len(shared_formats) != 1:
        raise ValueError("Experiment 04 arms do not share one budget/format contract")
    if len(shared_manifests) != 1 or len(shared_fixtures) != 1:
        raise ValueError("Experiment 04 arms do not share one manifest/development fixture")
    return configs


@dataclass(frozen=True)
class ArmComponents:
    retriever: Retriever
    selector: Selector
    generator: Generator


CitationJudge = Callable[
    [Query, SelectedEvidenceSet, GenerationResult],
    tuple[float, float],
]


def _recursive_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key).casefold())
            keys.update(_recursive_keys(child))
    elif isinstance(value, list | tuple):
        for child in value:
            keys.update(_recursive_keys(child))
    return keys


class Experiment04ArmRunner:
    """One entry for every formal system identity; components remain injectable for smoke."""

    def __init__(
        self,
        config: ArmConfig,
        components: ArmComponents,
        *,
        citation_judge: CitationJudge,
        query_analyzer: QueryAnalyzer | None = None,
    ) -> None:
        self.config = config
        self.components = components
        self.citation_judge = citation_judge
        self.query_analyzer = query_analyzer or RuleBasedQueryAnalyzer()

    def run(
        self,
        query: Query,
        *,
        scorer_sidecar: Mapping[str, Any],
    ) -> dict[str, Any]:
        checklist = self.query_analyzer.analyze(query)
        if checklist.query_id != query.query_id:
            raise ValueError("query analyzer returned the wrong query ID")
        candidates = self.components.retriever.retrieve(
            query,
            self.config.budget.final_top_k,
        )
        if candidates.query_id != query.query_id:
            raise ValueError("retriever returned the wrong query ID")
        if isinstance(self.components.selector, ProvenceSelector):
            pruned = self.components.selector.select_with_context(
                query,
                candidates,
                self.config.budget.max_selected,
            )
            selection = pruned.selection
            selected = pruned.selected
            selector_trace: dict[str, Any] = {
                "kind": "provence",
                "dropped_empty": pruned.dropped_empty,
            }
        else:
            selection = self.components.selector.select(
                query,
                candidates,
                self.config.budget.max_selected,
            )
            selected = resolve_selection(candidates, selection)
            selector_trace = {"kind": self.config.modules.selector}
        result = self.components.generator.generate(query, checklist, selected)
        validate_generation(selected, result)
        precision, recall = self.citation_judge(query, selected, result)
        selected_ids = [item.evidence_id for item in selected.evidence]
        cited_indices = [
            selected_ids.index(evidence_id) + 1
            for evidence_id in result.cited_evidence_ids
            if evidence_id in selected_ids
        ]
        outcome = {
            "query_id": query.query_id,
            "retrieved_unit_ids": [
                f"{candidate.document_id}:u000" for candidate in candidates.candidates
            ],
            "selected_unit_ids": [
                f"{candidate.document_id}:u000" for candidate in selected.evidence
            ],
            "selected_source_ids": selected_ids,
            "answer": result.answer,
            "citation_indices": cited_indices,
            "minicheck": {"precision": precision, "recall": recall},
        }
        score = score_bundle([scorer_sidecar], [outcome])
        runtime_output = {
            "schema_version": self.config.format.output_schema,
            "arm_id": self.config.arm_id,
            "query_id": query.query_id,
            "retrieval": {
                "candidate_count": len(candidates.candidates),
                "candidate_ids": [candidate.evidence_id for candidate in candidates.candidates],
            },
            "selection": {
                "selected_count": len(selected.evidence),
                "selected_ids": selected_ids,
                "trace": selector_trace,
            },
            "generation": result.model_dump(mode="json"),
            "scorer_input": outcome,
            "score": {
                "n_queries": score["n_queries"],
                "metrics": score["per_query"][0]["metrics"],
                "failure_reason": score["per_query"][0]["failure_reason"],
            },
        }
        forbidden = _recursive_keys(runtime_output) & {
            "gold",
            "gold_answer",
            "gold_answers",
            "gold_answer_aliases",
            "support_units",
            "component_id",
        }
        if forbidden:
            raise ValueError(f"runtime output leaked scorer-only fields: {sorted(forbidden)}")
        return runtime_output


def canonical_output_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
