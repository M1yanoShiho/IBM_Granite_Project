"""FastAPI application factory for the final three-module runtime."""

from collections.abc import Sequence

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from evidence_rag.api.schemas import HealthResponse, QueryRequest, QueryResponse
from evidence_rag.api.service import PipelineApiService


def create_app(
    service: PipelineApiService,
    *,
    allowed_origins: Sequence[str] = (),
) -> FastAPI:
    app = FastAPI(
        title="Evidence RAG API",
        version="1.0.0",
        description="Hybrid Retriever → trained NLI Selector → grounded GR-C Generator",
    )
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(allowed_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return service.health()

    @app.post("/v1/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        try:
            return service.query(request)
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(
                status_code=503,
                detail="runtime unavailable; check server configuration and model assets",
            ) from error

    return app
