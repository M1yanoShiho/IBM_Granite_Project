from __future__ import annotations

from fastapi.testclient import TestClient

from evidence_rag.api.app import create_app
from evidence_rag.api.service import PipelineApiService
from evidence_rag.cli.smoke import build_cpu_smoke_pipeline


def _client() -> TestClient:
    pipeline, _config = build_cpu_smoke_pipeline()
    return TestClient(create_app(PipelineApiService(lambda: pipeline)))


def test_health_does_not_eagerly_load_models() -> None:
    response = _client().get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "schema_version": "1.0",
        "status": "ok",
        "model_loaded": False,
        "runtime": "hybrid-nli-grc-seed13",
    }


def test_query_endpoint_returns_frontend_schema() -> None:
    client = _client()

    response = client.post(
        "/v1/query",
        json={
            "query_id": "frontend-http-q1",
            "session_id": "demo-session",
            "query": "What company did IBM acquire in 2019?",
            "top_k": 10,
            "max_selected": 10,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "1.0"
    assert payload["query_id"] == "frontend-http-q1"
    assert payload["session_id"] == "demo-session"
    assert payload["answer"] == "IBM acquired Red Hat in 2019."
    assert payload["candidates"]
    assert payload["selected_evidence"]
    assert payload["citations"]
    assert payload["diagnostics"] == {
        "retriever": "hybrid",
        "selector": "nli-risk-controlled",
        "generator": "grounded-grc",
        "candidate_count": 10,
        "selected_count": 9,
        "dropped_count": 1,
        "citation_count": 1,
    }


def test_query_endpoint_rejects_blank_queries() -> None:
    response = _client().post("/v1/query", json={"query": ""})

    assert response.status_code == 422


def test_query_endpoint_does_not_expose_private_runtime_errors() -> None:
    def unavailable_pipeline():
        raise ValueError("private model path and internal loader details")

    client = TestClient(create_app(PipelineApiService(unavailable_pipeline)))

    response = client.post("/v1/query", json={"query": "Where is the evidence?"})

    assert response.status_code == 503
    assert response.json() == {
        "detail": "runtime unavailable; check server configuration and model assets"
    }
    assert "private model path" not in response.text
