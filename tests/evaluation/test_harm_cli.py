import json
from pathlib import Path

import pytest

from evidence_rag.contracts.models import EvidenceCandidate, SelectedEvidenceSet
from evidence_rag.evaluation.harm_cli import main
from evidence_rag.materializer.provenance import MutationRecord, write_provenance


def ev(document_id: str) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=f"e-{document_id}",
        document_id=document_id,
        chunk_id=f"c-{document_id}",
        text=f"text {document_id}",
        source_uri=f"fixture://{document_id}",
        retrieval_score=1.0,
        retrieval_rank=1,
    )


def _write_selected(path: Path, sets: tuple[SelectedEvidenceSet, ...]) -> None:
    path.write_text("\n".join(item.model_dump_json() for item in sets) + "\n", encoding="utf-8")


def record(query_id: str, cf_doc: str) -> MutationRecord:
    return MutationRecord(
        query_id=query_id,
        needle_document_id="n",
        counterfactual_document_id=cf_doc,
        gold_value="18",
        gold_alias_used="18%",
        replacement_value="23",
        string_class="integer",
        seed=42,
        char_span=(0, 3),
        text_hash_before="a" * 64,
        text_hash_after="b" * 64,
        answer_bank_hash="c" * 64,
    )


def test_cli_emits_harm_comparison(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_provenance(tmp_path / "provenance.jsonl", (record("q1", "cf1"), record("q2", "cf2")))
    on = (
        SelectedEvidenceSet(query_id="q1", evidence=(ev("d1"),)),
        SelectedEvidenceSet(query_id="q2", evidence=(ev("d2"),)),
    )
    off = (
        SelectedEvidenceSet(query_id="q1", evidence=(ev("cf1"),)),
        SelectedEvidenceSet(query_id="q2", evidence=(ev("cf2"),)),
    )
    _write_selected(tmp_path / "on.jsonl", on)
    _write_selected(tmp_path / "off.jsonl", off)
    exit_code = main(
        (
            "--provenance",
            str(tmp_path / "provenance.jsonl"),
            "--selected-on",
            str(tmp_path / "on.jsonl"),
            "--selected-off",
            str(tmp_path / "off.jsonl"),
            "--dataset-signature",
            "sig",
            "--iterations",
            "500",
        )
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["harm_on"] == 0.0
    assert payload["harm_off"] == 1.0
    assert payload["delta"] == -1.0
    assert payload["n_paired"] == 2
