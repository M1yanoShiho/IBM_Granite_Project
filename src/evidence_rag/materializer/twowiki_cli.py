"""CLI: materialize a labeled 2WikiMultihopQA split into the JSONL dataset bundle.

Production reads parquet with pandas (downloading it once on the login node,
which has internet); tests inject ``rows`` directly so no pandas/network is needed.
"""

import argparse
import json
import os
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from evidence_rag.materializer.twowiki_loader import materialize_2wiki

TWIKI_URL_TEMPLATE = (
    "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/{split}.parquet"
)


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
    parser = argparse.ArgumentParser(description="Materialize 2WikiMultihopQA")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--query-limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument(
        "--output-split",
        default=None,
        help="manifest split label; defaults to --source-split (use heldout for a frozen dev subset)",
    )
    parser.add_argument(
        "--exclude-queries",
        type=Path,
        default=None,
        help="queries.jsonl whose query IDs must be excluded before deterministic sampling",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="local parquet; if omitted, download to $TWIKI_PARQUET_DIR or data/2wiki",
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
            Path(os.getenv("TWIKI_PARQUET_DIR", "data/2wiki"))
            / f"{arguments.source_split}.parquet"
        )
        _download(TWIKI_URL_TEMPLATE.format(split=arguments.source_split), parquet)
        rows = _load_rows(parquet)
    rows = list(rows)
    if arguments.exclude_queries is not None:
        excluded = {
            str(json.loads(line)["query_id"])
            for line in arguments.exclude_queries.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        rows = [row for row in rows if str(row.get("_id", "")) not in excluded]
    output_split = arguments.output_split or arguments.source_split
    result = materialize_2wiki(
        rows,
        arguments.output,
        query_limit=arguments.query_limit,
        seed=arguments.seed,
        split=output_split,
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
