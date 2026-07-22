from pathlib import Path

from evidence_rag.contracts.models import Document, Query
from evidence_rag.infrastructure.datasets import (
    DatasetBundle,
    DatasetManifest,
    GoldCase,
    JsonlDatasetAdapter,
)
from evidence_rag.materializer.answer_bank import build_answer_bank
from evidence_rag.materializer.injector import (
    InjectionTarget,
    find_injection_target,
    inject_counterfactual,
    materialize_counterfactuals,
    write_injected_dataset,
)
from evidence_rag.materializer.provenance import read_provenance


def doc(document_id: str, text: str) -> Document:
    return Document(document_id=document_id, text=text, source_uri=f"fixture://{document_id}")


def test_finds_target_when_alias_appears_exactly_once() -> None:
    documents = {"d1": doc("d1", "The operating margin was 18% last year.")}
    gold = GoldCase(
        query_id="q1",
        relevant_document_ids=("d1",),
        reference_answers=("18%", "18 percent"),
    )
    target = find_injection_target(gold, documents)
    assert isinstance(target, InjectionTarget)
    assert target.needle.document_id == "d1"
    assert target.alias_used == "18%"
    assert target.needle.text[target.span[0] : target.span[1]] == "18%"


def test_rejects_multi_valued_gold() -> None:
    documents = {"d1": doc("d1", "Alice or Bob won.")}
    gold = GoldCase(query_id="q1", relevant_document_ids=("d1",), reference_answers=("Alice", "Bob"))
    assert find_injection_target(gold, documents) is None


def test_rejects_when_alias_appears_more_than_once() -> None:
    documents = {"d1": doc("d1", "18% here and 18% there.")}
    gold = GoldCase(query_id="q1", relevant_document_ids=("d1",), reference_answers=("18%",))
    assert find_injection_target(gold, documents) is None


def test_rejects_when_no_alias_in_any_gold_passage() -> None:
    documents = {"d1": doc("d1", "No number appears here.")}
    gold = GoldCase(query_id="q1", relevant_document_ids=("d1",), reference_answers=("18%",))
    assert find_injection_target(gold, documents) is None


def test_injection_swaps_the_single_alias_and_records_provenance() -> None:
    documents = {"d1": doc("d1", "The operating margin was 18% last year.")}
    gold = GoldCase(query_id="q1", relevant_document_ids=("d1",), reference_answers=("18%",))
    target = find_injection_target(gold, documents)
    assert target is not None
    bank = build_answer_bank(["18%", "23", "44"], seed=42)
    result = inject_counterfactual(target, bank, seed=42)
    assert result is not None
    twin, record = result
    assert twin.document_id == "cf::q1::d1"
    assert "18%" not in twin.text
    assert record.replacement_value in {"23", "44"}
    assert record.needle_document_id == "d1"
    assert record.char_span == target.span


def test_injection_is_reversible_by_mutation_log() -> None:
    documents = {"d1": doc("d1", "Margin was 18% overall.")}
    gold = GoldCase(query_id="q1", relevant_document_ids=("d1",), reference_answers=("18%",))
    target = find_injection_target(gold, documents)
    assert target is not None
    bank = build_answer_bank(["18%", "23"], seed=42)
    injection = inject_counterfactual(target, bank, seed=42)
    assert injection is not None
    twin, record = injection
    start, _ = record.char_span
    restored = (
        twin.text[:start]
        + record.gold_alias_used
        + twin.text[start + len(record.replacement_value) :]
    )
    assert restored == documents["d1"].text


def test_injection_returns_none_when_bank_has_no_same_class_alternative() -> None:
    documents = {"d1": doc("d1", "Margin was 18% overall.")}
    gold = GoldCase(query_id="q1", relevant_document_ids=("d1",), reference_answers=("18%",))
    target = find_injection_target(gold, documents)
    assert target is not None
    bank = build_answer_bank(["Alice"], seed=42)
    assert inject_counterfactual(target, bank, seed=42) is None


def _bundle() -> DatasetBundle:
    documents = (
        doc("d1", "The operating margin was 18% last year."),
        doc("d2", "An unrelated distractor passage about rain."),
    )
    queries = (Query(query_id="q1", text="operating margin?"),)
    golds = (GoldCase(query_id="q1", relevant_document_ids=("d1",), reference_answers=("18%",)),)
    manifest = DatasetManifest(
        dataset_id="niah/nq",
        dataset_version="v1",
        split="dev",
        documents_file="documents.jsonl",
        queries_file="queries.jsonl",
        gold_cases_file="gold_cases.jsonl",
    )
    return DatasetBundle(
        manifest=manifest,
        dataset_signature="sig",
        documents=documents,
        queries=queries,
        gold_cases=golds,
    )


def test_materialize_produces_twin_and_record() -> None:
    bank = build_answer_bank(["18%", "23", "44"], seed=42)
    result = materialize_counterfactuals(_bundle(), bank, seed=42)
    assert len(result.injected_documents) == 3
    assert len(result.records) == 1
    assert result.records[0].counterfactual_document_id == "cf::q1::d1"


def test_write_injected_dataset_loads_back_through_adapter(tmp_path: Path) -> None:
    bank = build_answer_bank(["18%", "23"], seed=42)
    result = materialize_counterfactuals(_bundle(), bank, seed=42)
    manifest_path = write_injected_dataset(_bundle(), result, tmp_path, seed=42)
    reloaded = JsonlDatasetAdapter.load(manifest_path)
    ids = {document.document_id for document in reloaded.documents}
    assert "cf::q1::d1" in ids
    assert reloaded.manifest.dataset_version == "v1+cf42"
    provenance = read_provenance(tmp_path / "provenance.jsonl")
    assert len(provenance) == 1
