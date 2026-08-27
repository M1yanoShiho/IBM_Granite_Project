"""Run the public HTTP API for the final three-module pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from evidence_rag.api.app import create_app
from evidence_rag.api.service import PipelineApiService, load_runtime_pipeline

_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_CONFIG = _ROOT / "configs/runtime/final_seed13.toml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            os.environ.get("EVIDENCE_RAG_RUNTIME_CONFIG", DEFAULT_RUNTIME_CONFIG)
        ),
        help="Final runtime TOML config path.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        help="Frontend origin allowed by CORS; repeat for multiple origins.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    service = PipelineApiService(lambda: load_runtime_pipeline(config_path))
    app = create_app(service, allowed_origins=args.allow_origin)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
