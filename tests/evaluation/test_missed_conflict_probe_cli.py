import json
from pathlib import Path

from evidence_rag.evaluation.missed_conflict_probe_cli import main
from evidence_rag.materializer.provenance import MutationRecord, write_provenance


class FakeLLM:
    """Separates the twins: gold from the needle passage, the replacement from the counterfactual."""

    def generate(self, prompt: str) -> str:
        if "MARK_NEEDLE" in prompt:
            return "Kennedy"
        if "MARK_CF" in prompt:
            return "Nixon"
        return "NONE"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _dataset(tmp_path: Path) -> Path:
    _write_jsonl(tmp_path / "documents.jsonl", [
        {"schema_version": "1.0", "document_id": "needle", "text": "MARK_NEEDLE Kennedy won",
         "source_uri": "s://needle"},
        {"schema_version": "1.0", "document_id": "cf::needle", "text": "MARK_CF Nixon won",
         "source_uri": "s://cf"},
    ])
    _write_jsonl(tmp_path / "queries.jsonl", [
        {"schema_version": "1.0", "query_id": "q1", "text": "who won?"}
    ])
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": "1.0", "dataset_id": "niah", "dataset_version": "t+cf42", "split": "dev",
        "documents_file": "documents.jsonl", "queries_file": "queries.jsonl",
        "gold_cases_file": "gold_cases.jsonl",
    }), encoding="utf-8")
    return tmp_path / "manifest.json"


def test_cli_reports_per_prompt(tmp_path) -> None:
    manifest = _dataset(tmp_path)
    provenance = tmp_path / "provenance.jsonl"
    write_provenance(provenance, [
        MutationRecord(
            query_id="q1", needle_document_id="needle", counterfactual_document_id="cf::needle",
            gold_value="Kennedy", gold_alias_used="Kennedy", replacement_value="Nixon",
            string_class="name-1", seed=42, char_span=(0, 7), text_hash_before="a",
            text_hash_after="b", answer_bank_hash="h",
        )
    ])
    output = tmp_path / "probe.json"

    exit_code = main(
        ["--manifest", str(manifest), "--provenance", str(provenance), "--output", str(output)],
        llm=FakeLLM(),
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["n_records"] == 1
    assert "model" in report
    # decoupled (2-stage) runs alongside the single-stage strategies
    assert set(report["strategies"]) == {"baseline", "attribute", "decoupled"}
    for strategy in report["strategies"].values():
        assert strategy["missed_conflict_rate"] == 0.0  # twins separated: needle->Kennedy, cf->Nixon
        assert strategy["needle_gold_rate"] == 1.0
        assert strategy["cf_replacement_rate"] == 1.0


def test_cli_limit_subsamples(tmp_path) -> None:
    manifest = _dataset(tmp_path)
    provenance = tmp_path / "provenance.jsonl"
    write_provenance(provenance, [
        MutationRecord(
            query_id="q1", needle_document_id="needle", counterfactual_document_id="cf::needle",
            gold_value="Kennedy", gold_alias_used="Kennedy", replacement_value="Nixon",
            string_class="name-1", seed=42, char_span=(0, 7), text_hash_before="a",
            text_hash_after="b", answer_bank_hash="h",
        )
        for _ in range(5)
    ])
    output = tmp_path / "probe.json"
    main(
        ["--manifest", str(manifest), "--provenance", str(provenance), "--output", str(output),
         "--limit", "2"],
        llm=FakeLLM(),
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["n_records"] == 2  # limited to the first 2 records
