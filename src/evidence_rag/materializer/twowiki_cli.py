"""CLI: materialize 2WikiMultihopQA (dev) into the JSONL dataset bundle.

Production reads the dev parquet with pandas (downloading it once on the login node,
which has internet); tests inject ``rows`` directly so no pandas/network is needed.
"""

import argparse
import json
import os
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from evidence_rag.materializer.twowiki_loader import materialize_2wiki

TWIKI_URL = "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/dev.parquet"


def _download(url: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[data] downloading {url} -> {destination}", flush=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
    except Exception as exc:  # noqa: BLE001 - a download failure must be loud
        partial.unlink(missing_ok=True)
        raise SystemExit(f"could not download {url}: {exc}") from exc
    partial.replace(destination)
    return destination


def _load_rows(parquet: Path) -> list[Mapping[str, object]]:
    try:
        import pandas as pd  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "reading the 2Wiki parquet needs pandas + pyarrow: pip install pandas pyarrow"
        ) from exc
    frame = pd.read_parquet(parquet)
    return list(frame.to_dict("records"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize 2WikiMultihopQA dev")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--query-limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="local dev.parquet; if omitted, download to $TWIKI_PARQUET_DIR or data/2wiki",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    rows: Iterable[Mapping[str, object]] | None = None,
) -> int:
    arguments = _parser().parse_args(argv)
    if rows is None:
        parquet = arguments.parquet or (
            Path(os.getenv("TWIKI_PARQUET_DIR", "data/2wiki")) / "dev.parquet"
        )
        _download(TWIKI_URL, parquet)
        rows = _load_rows(parquet)
    result = materialize_2wiki(
        rows,
        arguments.output,
        query_limit=arguments.query_limit,
        seed=arguments.seed,
    )
    print(
        json.dumps(
            {
                "manifest": str(result.manifest_path),
                "documents": result.document_count,
                "queries": result.query_count,
                "gold_cases": result.gold_case_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
