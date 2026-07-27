import json
from pathlib import Path

from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate
from evidence_rag.evaluation.needle_probe_cli import main
from evidence_rag.materializer.provenance import MutationRecord, write_provenance


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    def generate(self, prompt: str) -> str:
        return self._reply


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _manifest(tmp_path: Path) -> Path:
    documents = [
        {"schema_version": "1.0", "document_id": "needle", "text": "Kennedy won the race",
         "source_uri": "s://needle"},
    ]
    queries = [{"schema_version": "1.0", "query_id": "q1", "text": "who won?"}]
    gold_cases = [
        {"query_id": "q1", "relevant_document_ids": ["needle"], "reference_answers": ["Kennedy"]}
    ]
    _write_jsonl(tmp_path / "documents.jsonl", documents)
    _write_jsonl(tmp_path / "queries.jsonl", queries)
    _write_jsonl(tmp_path / "gold_cases.jsonl", gold_cases)
    manifest = {
        "schema_version": "1.0", "dataset_id": "niah", "dataset_version": "test+cf42",
        "split": "dev", "documents_file": "documents.jsonl", "queries_file": "queries.jsonl",
        "gold_cases_file": "gold_cases.jsonl",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path / "manifest.json"


def _candidates(tmp_path: Path) -> Path:
    candidate_set = CandidateSet(
        query_id="q1",
        candidates=(
            EvidenceCandidate(
                evidence_id="e_needle", document_id="needle", chunk_id="needle::c0",
                text="Kennedy won the race", source_uri="s://needle",
                retrieval_score=2.0, retrieval_rank=1,
            ),
        ),
    )
    path = tmp_path / "candidate_sets.jsonl"
    path.write_text(candidate_set.model_dump_json() + "\n", encoding="utf-8")
    return path


def _provenance(tmp_path: Path) -> Path:
    path = tmp_path / "provenance.jsonl"
    write_provenance(
        path,
        [
            MutationRecord(
                query_id="q1", needle_document_id="needle",
                counterfactual_document_id="cf::needle", gold_value="Kennedy",
                gold_alias_used="Kennedy", replacement_value="Nixon", string_class="name-1",
                seed=42, char_span=(0, 7), text_hash_before="a", text_hash_after="b",
                answer_bank_hash="h",
            )
        ],
    )
    return path


def test_probe_classifies_visible_none(tmp_path):
    # gold "Kennedy" is visible in the needle chunk, but the extractor abstains -> visible NONE.
    output = tmp_path / "probe_summary.json"
    dump = tmp_path / "probe_dump.jsonl"

    exit_code = main(
        [
            "--manifest", str(_manifest(tmp_path)),
            "--candidates", str(_candidates(tmp_path)),
            "--provenance", str(_provenance(tmp_path)),
            "--output", str(output),
            "--dump", str(dump),
        ],
        llm=FakeLLM("NONE"),
    )

    assert exit_code == 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["n"] == 1
    assert summary["n_visible"] == 1
    assert summary["visible_none"] == 1
    assert summary["visible_recovered"] == 0
    row = json.loads(dump.read_text(encoding="utf-8").splitlines()[0])
    assert row["outcome"] == "none"
    assert row["visible"] is True
    assert row["question"] == "who won?"
