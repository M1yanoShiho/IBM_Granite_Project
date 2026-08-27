"""Send one request to a running Evidence RAG API using the standard library."""

from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen


def main() -> None:
    base_url = os.environ.get("EVIDENCE_RAG_API_URL", "http://127.0.0.1:8000")
    payload = json.dumps(
        {
            "query_id": "example-q1",
            "query": "What company did IBM acquire in 2019?",
            "top_k": 10,
            "max_selected": 10,
        }
    ).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/v1/query",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:  # noqa: S310 - user-configured URL
        print(json.dumps(json.load(response), indent=2))


if __name__ == "__main__":
    main()
