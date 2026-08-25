# Frontend integration

The browser talks only to the Evidence RAG HTTP service. It does not need an HPC account
and must not download or parse model checkpoints.

## Start a local backend

For UI development without models, use `examples/mock_frontend_response.json`. For an
integrated run, install the API extra, configure the external assets described in
`docs/three-module-runtime-handoff.md`, and start:

```bash
evidence-rag-serve --host 127.0.0.1 --port 8000 \
  --allow-origin http://localhost:3000
```

## Health check

`GET /health` returns without loading the pipeline:

```json
{
  "schema_version": "1.0",
  "status": "ok",
  "model_loaded": false,
  "runtime": "hybrid-nli-grc-seed13"
}
```

`model_loaded` becomes `true` after the first successful query.

## Query

Send `POST /v1/query` with JSON:

```json
{
  "schema_version": "1.0",
  "query_id": "frontend-q1",
  "session_id": "demo-session",
  "query": "What company did IBM acquire in 2019?",
  "top_k": 10,
  "max_selected": 10
}
```

`query_id` and `session_id` are optional. The server generates a query ID when omitted.
`top_k` is limited to 1–100 and `max_selected` to 1–10.

The response contains:

- `candidates`: evidence returned by the Hybrid Retriever;
- `selected_evidence`: only evidence retained by the trained NLI Selector;
- `answer`: output from the grounded GR-C Generator;
- `citations`: selected evidence actually cited by the answer;
- `diagnostics`: frozen module names and stage counts.

The UI should display citations from `citations`, not infer them from candidate rank. A
candidate can appear in `candidates` but be deliberately absent from
`selected_evidence` after Selector processing.

## Errors

- `422`: malformed input, blank query, or a limit outside the allowed range.
- `503`: the configured dataset/model/index assets are missing, unreadable, or fail
  integrity validation.

Do not expose the backend's environment variables, local paths, or Python exception text
to the browser. The public 503 response is intentionally generic.

## Examples

- `examples/api_request.py` sends one request using only the Python standard library.
- `examples/mock_frontend_response.json` is a schema-validated response for UI tests.
