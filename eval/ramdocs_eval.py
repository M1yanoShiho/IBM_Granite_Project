# eval/ramdocs_eval.py
"""V4 external validation: corroboration reranking on RAMDocs (COLM 2025).

RAMDocs (arXiv:2504.13079, MIT) gives per-query document SETS labelled
correct / misinfo / noise — conflicting evidence WE did not construct. Each set is
reranked directly (no index): relevance = Granite dense query-doc cosine;
corroboration = the SAME cross-source answer votes as the certified method (one
extraction pass; both arms derived arithmetically). Pre-registered: alpha is
FROZEN at 0.6 (the certified blend) — no tuning on this benchmark, either way the
outcome is reported.

Metrics per query, over the label of the ranked docs: correct-doc@1, correct-doc@3,
MRR of the first correct doc, misinfo@1 (lower = better). Paired CSVs feed
``eval.significance`` (reference arm: relevance).

    python -m eval.ramdocs_eval --data /user/work/$USER/ramdocs/RAMDocs_test.jsonl \
        --alpha 0.6 --out-prefix results/ramdocs_corrob
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

from src.retrieval.fusion import minmax_normalize

_REQUIRED_DOC_KEYS = {"text", "type"}


def load_ramdocs(path: Path) -> List[dict]:
    """Parse RAMDocs_test.jsonl; fail loud (with the line number) on schema drift."""
    examples: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "question" not in obj or not isinstance(obj.get("documents"), list) \
                    or not obj["documents"]:
                raise ValueError(f"RAMDocs line {i}: need 'question' + non-empty 'documents'.")
            for d in obj["documents"]:
                if not _REQUIRED_DOC_KEYS <= set(d):
                    raise ValueError(f"RAMDocs line {i}: document missing {_REQUIRED_DOC_KEYS}.")
            examples.append(obj)
    return examples


def rank_order(relevance: List[float], votes: List[float], alpha: float) -> List[int]:
    """Doc indices best-first under ``alpha*relevance + (1-alpha)*corroboration``
    (both min-max normalised — exactly the certified blend). Stable: ties keep the
    input order."""
    n = len(relevance)
    rel_n = minmax_normalize({i: relevance[i] for i in range(n)})
    vote_n = minmax_normalize({i: votes[i] for i in range(n)})
    final = {i: alpha * rel_n[i] + (1.0 - alpha) * vote_n[i] for i in range(n)}
    return sorted(range(n), key=lambda i: final[i], reverse=True)


def correct_at_k(ranked_types: List[str], k: int) -> float:
    return 1.0 if any(t == "correct" for t in ranked_types[:k]) else 0.0


def misinfo_at_1(ranked_types: List[str]) -> float:
    return 1.0 if ranked_types and ranked_types[0] == "misinfo" else 0.0


def rr_first_correct(ranked_types: List[str]) -> float:
    for i, t in enumerate(ranked_types):
        if t == "correct":
            return 1.0 / (i + 1)
    return 0.0


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m eval.ramdocs_eval",
        description="Corroboration blend vs relevance-only on RAMDocs doc sets (V4).",
    )
    p.add_argument("--data", type=Path, required=True, help="Path to RAMDocs_test.jsonl.")
    p.add_argument("--alpha", type=float, default=0.6, help="FROZEN blend weight (no tuning).")
    p.add_argument("--limit", type=int, default=None, help="Cap examples (smoke run).")
    p.add_argument("--out-prefix", type=Path, default=Path("results/ramdocs_corrob"),
                   dest="out_prefix")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)
    examples = load_ramdocs(args.data)
    if args.limit is not None:
        examples = examples[: args.limit]
    print(f"RAMDocs examples: {len(examples)} (alpha={args.alpha} frozen)")

    # Heavy deps lazily, mirroring the repo's eval tools.
    from eval.run_benchmark import write_per_query_csv
    from src.llm_client import LLMClient
    from src.retrieval.base import RetrievedChunk
    from src.retrieval.embedder import Embedder
    from src.retrieval.reranker import CorroborationReranker

    embedder = Embedder()
    llm = LLMClient()

    metrics: Dict[str, Dict[str, Dict[str, float]]] = {
        m: {"relevance": {}, "corroborate": {}}
        for m in ("correct1", "correct3", "mrr_correct", "misinfo1")
    }
    for i, ex in enumerate(examples):
        qid = f"r{i}"
        texts = [d["text"] for d in ex["documents"]]
        types = [d["type"] for d in ex["documents"]]
        qv = embedder.embed_query(ex["question"])
        dvs = embedder.embed_documents(texts)
        relevance = [_cosine(qv, dv) for dv in dvs]
        chunks = [RetrievedChunk(f"d{j}", texts[j], relevance[j]) for j in range(len(texts))]
        rr = CorroborationReranker(llm, top_n=len(chunks), alpha=args.alpha)
        _, votes, _, _ = rr.score_docs_with_answers(ex["question"], chunks)

        for arm, alpha in (("relevance", 1.0), ("corroborate", args.alpha)):
            ranked_types = [types[j] for j in rank_order(relevance, votes, alpha)]
            metrics["correct1"][arm][qid] = correct_at_k(ranked_types, 1)
            metrics["correct3"][arm][qid] = correct_at_k(ranked_types, 3)
            metrics["mrr_correct"][arm][qid] = rr_first_correct(ranked_types)
            metrics["misinfo1"][arm][qid] = misinfo_at_1(ranked_types)

    n = len(examples)
    print(f"{'metric':<12} {'relevance':>10} {'corroborate':>12}")
    for m, cols in metrics.items():
        r = sum(cols["relevance"].values()) / n
        c = sum(cols["corroborate"].values()) / n
        print(f"{m:<12} {r:>10.4f} {c:>12.4f}")
        path = Path(f"{args.out_prefix}_{m}.csv")
        write_per_query_csv(cols, path)
    print(f"wrote {args.out_prefix}_<metric>.csv — significance: "
          f"python -m eval.significance --per-query-csv {args.out_prefix}_correct1.csv "
          "--reference relevance")


if __name__ == "__main__":
    main()
