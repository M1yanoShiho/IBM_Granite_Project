from __future__ import annotations

from pathlib import Path

from evidence_rag.contracts.models import (
    CandidateSet,
    Document,
    EvidenceCandidate,
    Query,
    RetrieverProvenance,
    SelectionItem,
    SelectionResult,
)
from evidence_rag.evaluation.selector_experiment import (
    align_inputs,
    audit_candidate_pool,
    compare_arm_results,
    run_selector_arm,
    sha256_file,
    validate_pool_manifest,
)
from evidence_rag.infrastructure.datasets import (
    DatasetBundle,
    DatasetManifest,
    GoldCase,
)
from evidence_rag.materializer.source_parent import ParentIndex
from evidence_rag.selector.top_k import TopKSelector

RETRIEVER = RetrieverProvenance(
    name="hybrid",
    implementation_version="hybrid-v1",
    parameters_sha256="a" * 64,
)


def evidence(document_id: str, rank: int) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=f"{document_id}#0",
        document_id=document_id,
        chunk_id="0",
        text=f"text {document_id}",
        source_uri=f"fixture://{document_id}",
        retrieval_score=1.0 / rank,
        retrieval_rank=rank,
    )


def bundle() -> DatasetBundle:
    queries = (
        Query(query_id="q1", text="question one"),
        Query(query_id="q2", text="question two"),
    )
    documents = tuple(
        Document(document_id=item, text=f"text {item}", source_uri=f"fixture://{item}")
        for item in ("d1", "d2", "h1", "h2", "x1", "x2")
    )
    return DatasetBundle(
        manifest=DatasetManifest(
            dataset_id="fixture/selector",
            dataset_version="v1",
            split="test",
            documents_file="documents.jsonl",
            queries_file="queries.jsonl",
            gold_cases_file="gold_cases.jsonl",
        ),
        dataset_signature="dataset-signature",
        documents=documents,
        queries=queries,
        gold_cases=(
            GoldCase(query_id="q1", relevant_document_ids=("d1",)),
            GoldCase(query_id="q2", relevant_document_ids=("d2",)),
        ),
    )


def pools() -> tuple[CandidateSet, ...]:
    return (
        CandidateSet(
            query_id="q1",
            candidates=(evidence("h1", 1), evidence("d1", 2), evidence("x1", 3)),
            retriever=RETRIEVER,
        ),
        CandidateSet(
            query_id="q2",
            candidates=(evidence("d2", 1), evidence("x2", 2), evidence("x1", 3)),
            retriever=RETRIEVER,
        ),
    )


class RequiredOnlySelector:
    def select(self, query: Query, candidates: CandidateSet, max_selected: int) -> SelectionResult:
        target = "d1#0" if query.query_id == "q1" else "d2#0"
        return SelectionResult(
            query_id=query.query_id,
            items=(SelectionItem(evidence_id=target, selection_score=1.0, selection_rank=1),),
        )


class ExplodingSelector:
    def select(self, query: Query, candidates: CandidateSet, max_selected: int) -> SelectionResult:
        raise AssertionError("completed checkpoints must be reused")


def write_pool(path: Path) -> None:
    path.write_text(
        "".join(item.model_dump_json() + "\n" for item in pools()),
        encoding="utf-8",
    )


def test_pool_audit_binds_sha_recall_harm_and_parents(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate_sets.jsonl"
    dataset_path = tmp_path / "manifest.json"
    config_path = tmp_path / "config.toml"
    write_pool(candidate_path)
    dataset_path.write_text("{}\n", encoding="utf-8")
    config_path.write_text("[retriever]\nname='hybrid'\n", encoding="utf-8")
    parent_index = ParentIndex(
        parent_by_document={item: item for item in ("d1", "d2", "h1", "x1", "x2")}
    )

    manifest = audit_candidate_pool(
        bundle(),
        pools(),
        parent_index,
        candidate_pool_path=candidate_path,
        dataset_manifest_path=dataset_path,
        retriever_config_path=config_path,
        top_n=3,
        seed=13,
        harm_by_query={"q1": "h1", "q2": "h2"},
        embedding_model_id="fixture-embedding",
        embedding_revision="fixture-revision",
        git_root=tmp_path,
    )
    assert manifest["candidate_pool"]["sha256"] == sha256_file(candidate_path)  # type: ignore[index]
    assert manifest["candidate_pool"]["exact_top_n_rate"] == 1.0  # type: ignore[index]
    assert manifest["audit"]["required_recall_at_top_n"] == 1.0  # type: ignore[index]
    assert manifest["audit"]["harmful_pool_hit_rate"] == 0.5  # type: ignore[index]
    assert manifest["audit"]["unresolved_parent_count"] == 0  # type: ignore[index]
    assert validate_pool_manifest(
        manifest,
        candidate_pool_path=candidate_path,
        expected_top_n=3,
    ) == sha256_file(candidate_path)


def test_two_arms_use_same_pool_and_statistics_pair_by_query(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate_sets.jsonl"
    source_parent = tmp_path / "source_parent.jsonl"
    write_pool(candidate_path)
    source_parent.write_text("fixture\n", encoding="utf-8")
    pool_sha = sha256_file(candidate_path)
    parent_sha = sha256_file(source_parent)
    aligned = align_inputs(bundle(), pools())
    harm = {"q1": "h1", "q2": "h2"}

    top_rows, _top_selection, _top_selected = run_selector_arm(
        "top-k",
        TopKSelector(),
        aligned,
        output_directory=tmp_path / "arms" / "top-k",
        candidate_pool_sha256=pool_sha,
        source_parent_sha256=parent_sha,
        max_selected=2,
        run_seed=13,
        dataset_signature=bundle().dataset_signature,
        harm_by_query=harm,
        git_root=tmp_path,
    )
    mis_rows, _mis_selection, _mis_selected = run_selector_arm(
        "reliability-mis",
        RequiredOnlySelector(),
        aligned,
        output_directory=tmp_path / "arms" / "reliability-mis",
        candidate_pool_sha256=pool_sha,
        source_parent_sha256=parent_sha,
        max_selected=2,
        run_seed=13,
        dataset_signature=bundle().dataset_signature,
        harm_by_query=harm,
        git_root=tmp_path,
    )
    summary = compare_arm_results(top_rows, mis_rows, stats_seed=20260809, iterations=200)

    assert summary["statistical_unit"] == "query"
    assert summary["paired_harm"]["n_paired"] == 1  # type: ignore[index]
    assert summary["paired_required_recall"]["n_paired"] == 2  # type: ignore[index]
    assert all(set(row["selected_ids"]) <= set(row["candidate_ids"]) for row in mis_rows)
    assert (
        (tmp_path / "arms/top-k/run_manifest.json").read_text()
        .count(pool_sha)
        == 1
    )
    assert pool_sha in (tmp_path / "arms/reliability-mis/run_manifest.json").read_text()

    resumed, _selection, _selected = run_selector_arm(
        "reliability-mis",
        ExplodingSelector(),
        aligned,
        output_directory=tmp_path / "arms" / "reliability-mis",
        candidate_pool_sha256=pool_sha,
        source_parent_sha256=parent_sha,
        max_selected=2,
        run_seed=13,
        dataset_signature=bundle().dataset_signature,
        harm_by_query=harm,
        resume=True,
        git_root=tmp_path,
    )
    assert resumed == mis_rows
