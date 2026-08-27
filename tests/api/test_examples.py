from __future__ import annotations

from pathlib import Path

from evidence_rag.api.schemas import QueryResponse

_ROOT = Path(__file__).resolve().parents[2]
_MOCK_RESPONSE = _ROOT / "examples/mock_frontend_response.json"


def test_mock_frontend_response_matches_public_schema_and_stage_boundary() -> None:
    response = QueryResponse.model_validate_json(_MOCK_RESPONSE.read_text(encoding="utf-8"))

    assert response.diagnostics.candidate_count == len(response.candidates)
    assert response.diagnostics.selected_count == len(response.selected_evidence)
    assert response.diagnostics.citation_count == len(response.citations)
    selected_ids = {item.evidence_id for item in response.selected_evidence}
    assert "ev-poison-demo" not in selected_ids
    assert {item.evidence_id for item in response.citations} <= selected_ids
