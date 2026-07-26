import json
from pathlib import Path

from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate
from evidence_rag.evaluation.cluster_eval_cli import main
from evidence_rag.materializer.provenance import MutationRecord, write_provenance


class FakeLLM:
    """Deterministic extractor: returns the answer keyed by a marker in the passage."""

    def __init__(self, answer_by_marker: dict[str, str]) -> None:
        self._answers = answer_by_marker

    def generate(self, prompt: str) -> str:
        for marker, answer in self._answers.items():
            if marker in prompt:
                return answer
        return "NONE"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _manifest(tmp_path: Path) -> Path:
    documents = [
        {"schema_version": "1.0", "document_id": "needle", "text": "MARK_NEEDLE Kennedy won",
         "source_uri": "s://needle"},
        {"schema_version": "1.0", "document_id": "cf::needle", "text": "MARK_CF Nixon won",
         "source_uri": "s://cf"},
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
                text="MARK_NEEDLE Kennedy won", source_uri="s://needle",
                retrieval_score=2.0, retrieval_rank=1,
            ),
            EvidenceCandidate(
                evidence_id="e_cf", document_id="cf::needle", chunk_id="cf::needle::c0",
                text="MARK_CF Nixon won", source_uri="s://cf",
                retrieval_score=1.0, retrieval_rank=2,
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


def test_cli_writes_report_with_fake_extractor(tmp_path, capsys):
    manifest = _manifest(tmp_path)
    candidates = _candidates(tmp_path)
    provenance = _provenance(tmp_path)
    output = tmp_path / "cluster_eval_report.json"

    exit_code = main(
        [
            "--manifest", str(manifest),
            "--candidates", str(candidates),
            "--provenance", str(provenance),
            "--output", str(output),
        ],
        llm=FakeLLM({"MARK_NEEDLE": "Kennedy", "MARK_CF": "Nixon"}),
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    # both scorings are reported so verbosity bias is visible, not hidden (S5)
    for scoring in ("exact", "lenient"):
        assert report[scoring]["needle_gold_recovery"]["rate"] == 1.0
        assert report[scoring]["missed_conflict"]["rate"] == 0.0
    assert report["selection_bias"]["n_injected"] == 1
    assert report["extraction"] == "single"
    printed = capsys.readouterr().out
    assert "missed_conflict" in printed


def test_cli_lenient_credits_verbose_extraction(tmp_path) -> None:
    """A wordier but correct extraction fails exact scoring and passes lenient."""

    class VerboseLLM:
        def generate(self, prompt: str) -> str:
            if "MARK_NEEDLE" in prompt:
                return "The winner was Kennedy"
            if "MARK_CF" in prompt:
                return "The winner was Nixon"
            return "NONE"

    output = tmp_path / "report.json"
    main(
        [
            "--manifest", str(_manifest(tmp_path)),
            "--candidates", str(_candidates(tmp_path)),
            "--provenance", str(_provenance(tmp_path)),
            "--output", str(output),
        ],
        llm=VerboseLLM(),
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["exact"]["needle_gold_recovery"]["rate"] == 0.0
    assert report["lenient"]["needle_gold_recovery"]["rate"] == 1.0


def test_cli_decoupled_extraction_runs(tmp_path) -> None:
    """The decoupled path issues a Stage A target call and Stage B per passage."""
    prompts: list[str] = []

    class TracingLLM:
        def generate(self, prompt: str) -> str:
            prompts.append(prompt)
            if "Target:" in prompt:
                return "a person's name"
            if "MARK_NEEDLE" in prompt:
                return "Kennedy"
            if "MARK_CF" in prompt:
                return "Nixon"
            return "NONE"

    output = tmp_path / "report.json"
    exit_code = main(
        [
            "--manifest", str(_manifest(tmp_path)),
            "--candidates", str(_candidates(tmp_path)),
            "--provenance", str(_provenance(tmp_path)),
            "--output", str(output),
            "--extraction", "decoupled",
        ],
        llm=TracingLLM(),
    )
    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["extraction"] == "decoupled"
    assert any("Target:" in p for p in prompts)  # Stage A ran
    # Stage B carries both the named target and the original question
    stage_b = [p for p in prompts if "a person's name" in p]
    assert stage_b and all("who won?" in p for p in stage_b)
