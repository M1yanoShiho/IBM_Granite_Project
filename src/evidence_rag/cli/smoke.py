from evidence_rag.composition import build_baseline
from evidence_rag.contracts.models import Document, Query


def main() -> None:
    pipeline = build_baseline(
        (
            Document(
                document_id="smoke-doc",
                text="The clean pipeline passes evidence between three modules.",
                source_uri="fixture://smoke-doc",
            ),
        )
    )
    result = pipeline.run(
        Query(query_id="smoke-query", text="clean pipeline evidence"),
        top_k=3,
        max_selected=2,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
