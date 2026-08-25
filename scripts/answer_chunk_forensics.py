"""R16: for every conditional miss, find where the answer-bearing chunk stopped.

R15 established that CMR jumps from 0.0000 to 0.0668 once chunk_size drops below the
corpus's native passage length, and the chunk counts (1.000 against 2.002 chunks per
document) fixed the split itself as fact. Its account of *why* did not survive the same
day: a span survives the window whenever its length is at most `overlap`, which is 10, 20,
33 and 50 words against NQ's longest 5-word answer, so no answer was ever cut. The chunk
holding it exists throughout -- it just never reaches the final evidence set.

Three properties of the scoring chain decide what "never reaches" can mean, and all three
come from code rather than inference:

* `ExtractiveGenerator` concatenates each selected chunk verbatim, separated by markers
  whose identifiers survive normalisation as words, so `answer_match` is exactly "the gold
  string lands inside a *single* selected chunk".
* It cites every selected item, so the cited and selected document sets are identical --
  which is why R15's two conditioning events agreed to the last digit.
* The top-k selector truncates the ranked list at `max_selected`, so "not selected" means
  precisely "ranked past `max_selected`".

Each miss therefore falls into one of four classes, and they call for different fixes:

* `sibling`  -- an answer-bearing candidate shares a document with a selected chunk. The
               passage was split and the half carrying the query terms won. (H16 bets >60%.)
* `other`    -- an answer-bearing candidate sits below the cut in a *different* document.
* `retrieval`-- no answer-bearing candidate is in the pool at all; top_k is the binding
               constraint and the diagnosis moves to the retrieval side.
* `absent`   -- no chunk of any gold document holds the answer. Above 5% of cases this
               refutes the arithmetic above, and nothing else in the entry may be trusted.

Normalisation is imported from `evaluation.scoring`, never reimplemented: every criterion
here rests on sharing `answer_match`'s own definition of a match, and a second copy would
silently become a second metric.

    PYTHONPATH=src python scripts/answer_chunk_forensics.py \
        --run runs/chunk-nq-b-c60o10 \
        --report runs/chunk-nq-b-c60o10/evaluation_report.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

# Private by name, deliberate by use: R16's criteria are only meaningful while this is the
# same normaliser answer_match applies. Copying it would fork the metric.
from evidence_rag.evaluation.scoring import _normalise

CONDITIONAL = "generator.core.conditional_answer_match"
ANSWER = "system.core.answer_match"


def _metric(group: dict[str, object], name: str) -> float | None:
    entry = group[name]
    if isinstance(entry, dict):
        return entry["value"]  # type: ignore[return-value]
    return entry  # type: ignore[return-value]


def read_population(report: Path) -> set[str]:
    """Query IDs where a gold document was selected yet the answer was not found."""
    payload = json.loads(report.read_text(encoding="utf-8"))
    return {
        case["query_id"]
        for case in payload["per_case"]
        if _metric(case["generator"], CONDITIONAL) is not None
        and _metric(case["system"], ANSWER) == 0
    }


def read_gold(path: Path) -> dict[str, tuple[tuple[str, ...], frozenset[str]]]:
    """{query_id: (normalised reference answers, relevant document ids)}."""
    gold: dict[str, tuple[tuple[str, ...], frozenset[str]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            answers = tuple(
                normalised
                for reference in (record.get("reference_answers") or ())
                if (normalised := _normalise(reference))
            )
            documents = frozenset(record.get("relevant_document_ids") or ())
            gold[record["query_id"]] = (answers, documents)
    return gold


def _bears_answer(text: str, answers: tuple[str, ...]) -> bool:
    normalised = _normalise(text)
    return any(answer in normalised for answer in answers)


def _bearing(run: dict[str, object], answers: tuple[str, ...]) -> list[dict[str, object]]:
    """Answer-bearing candidates in the pool, in retrieval-rank order."""
    candidates = sorted(
        run["candidates"]["candidates"],  # type: ignore[index]
        key=lambda item: item["retrieval_rank"],
    )
    return [item for item in candidates if _bears_answer(item["text"], answers)]


def classify_run(
    run: dict[str, object],
    answers: tuple[str, ...],
) -> tuple[str, int | None]:
    """Return (class, the rank that class is about) for one pipeline run.

    The rank is the SIBLING's whenever the class is `sibling`, not the shallowest
    answer-bearing candidate: R16 asks how deep the split half fell, and a nearer candidate
    sitting in some other document does not answer that. The two coincide whenever only one
    candidate bears the answer, which is why they never disagreed at `top_k=50`. Once R18
    widened the pool to 1000 they came apart, and the difference is not cosmetic -- a case
    can be reported at rank 30 while the answer was already visible at rank 11.

    `shallowest_bearing_rank` is the other quantity. Read it, not this one, for "how deep
    would a pool have to reach". This one stays as it is because R16's published shares and
    ranks were computed from it and must remain reproducible.
    """
    selected = run["selected"]["evidence"]  # type: ignore[index]
    selected_documents = {item["document_id"] for item in selected}
    bearing = _bearing(run, answers)
    if not bearing:
        return "retrieval", None
    best = bearing[0]
    # A sibling loss is the split passage itself: the answer sits in another chunk of a
    # document some selected chunk already came from.
    if any(item["document_id"] in selected_documents for item in bearing):
        sibling = next(item for item in bearing if item["document_id"] in selected_documents)
        return "sibling", sibling["retrieval_rank"]
    return "other", best["retrieval_rank"]


def shallowest_bearing_rank(
    run: dict[str, object],
    answers: tuple[str, ...],
) -> int | None:
    """Rank of the shallowest answer-bearing candidate, whichever document it sits in.

    This is the depth a wider pool -- or a reranker over one -- would have to reach before
    the answer is visible at all, which is the question `classify_run`'s rank silently
    answers differently for `sibling` cases. `None` when nothing in the pool bears it.
    """
    bearing = _bearing(run, answers)
    return int(bearing[0]["retrieval_rank"]) if bearing else None  # type: ignore[arg-type]


def corpus_holds_answer(
    corpus: Path,
    wanted: dict[str, tuple[tuple[str, ...], frozenset[str]]],
) -> set[str]:
    """Query IDs whose answer appears in no chunk of any of their gold documents."""
    documents = frozenset().union(*(docs for _, docs in wanted.values())) if wanted else frozenset()
    payload = json.loads(corpus.read_text(encoding="utf-8"))
    by_document: dict[str, list[str]] = {}
    for chunk in payload["chunks"]:
        if chunk["document_id"] in documents:
            by_document.setdefault(chunk["document_id"], []).append(_normalise(chunk["text"]))
    absent = set()
    for query_id, (answers, docs) in wanted.items():
        texts = [text for document in docs for text in by_document.get(document, ())]
        if not any(answer in text for text in texts for answer in answers):
            absent.add(query_id)
    return absent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="a run directory")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--skip-corpus-check",
        action="store_true",
        help="skip the `absent` class; it leaves R16's falsification B unresolved",
    )
    parser.add_argument(
        "--dump",
        type=Path,
        help="write per-case class, rank and gold answers as JSONL, so a class can be "
        "cross-tabulated against the answers themselves",
    )
    arguments = parser.parse_args(argv)

    population = read_population(arguments.report)
    gold = read_gold(arguments.run / "gold_cases.jsonl")

    classes: dict[str, str] = {}
    ranks: dict[str, int] = {}
    shallowest: dict[str, int] = {}
    max_selected: set[int] = set()
    top_k: set[int] = set()
    with (arguments.run / "pipeline_runs.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            run = json.loads(line)
            query_id = run["query"]["query_id"]
            if query_id not in population:
                continue
            max_selected.add(run["max_selected"])
            top_k.add(run["top_k"])
            answers, _ = gold[query_id]
            label, rank = classify_run(run, answers)
            classes[query_id] = label
            if rank is not None:
                ranks[query_id] = rank
            nearest = shallowest_bearing_rank(run, answers)
            if nearest is not None:
                shallowest[query_id] = nearest

    missing = population - set(classes)
    if missing:
        raise SystemExit(f"{len(missing)} population cases absent from pipeline_runs.jsonl")

    if not arguments.skip_corpus_check:
        retrieval = {q: gold[q] for q, label in classes.items() if label == "retrieval"}
        for query_id in corpus_holds_answer(arguments.run / "corpus_snapshot.json", retrieval):
            classes[query_id] = "absent"

    if arguments.dump:
        with arguments.dump.open("w", encoding="utf-8") as handle:
            for query_id in sorted(classes):
                handle.write(
                    json.dumps(
                        {
                            "query_id": query_id,
                            "class": classes[query_id],
                            # `best_rank` keeps R16's meaning -- the rank the class is about
                            # -- so dumps written before R18 stay comparable field for field.
                            "best_rank": ranks.get(query_id),
                            "shallowest_rank": shallowest.get(query_id),
                            "reference_answers": list(gold[query_id][0]),
                        }
                    )
                    + "\n"
                )

    counts = Counter(classes.values())
    total = len(classes)
    print(f"run          : {arguments.run}")
    print(f"selector     : top_k {sorted(top_k)}  max_selected {sorted(max_selected)}")
    print(f"conditional misses: {total}")
    print()
    for label in ("sibling", "other", "retrieval", "absent"):
        n = counts.get(label, 0)
        share = n / total if total else float("nan")
        print(f"  {label:<10} {n:>5}  {share:>7.1%}")
    if arguments.skip_corpus_check:
        print("  (absent not computed: --skip-corpus-check; falsification B unresolved)")

    if ranks:
        ordered = sorted(ranks.values())
        print()
        print("rank the class is about -- sibling's own rank for siblings (misses only)")
        print(f"  n {len(ordered)}  min {ordered[0]}  median {ordered[len(ordered) // 2]}  max {ordered[-1]}")
        cut = max(max_selected) if max_selected else 0
        just_past = sum(1 for rank in ordered if rank <= cut + 5)
        print(f"  within 5 ranks of the cut ({cut}): {just_past} ({just_past / len(ordered):.1%})")

    if shallowest:
        ordered = sorted(shallowest.values())
        divergent = sum(1 for query_id, rank in shallowest.items() if ranks.get(query_id) != rank)
        print()
        print("shallowest answer-bearing candidate -- the depth a wider pool must reach")
        print(f"  n {len(ordered)}  min {ordered[0]}  median {ordered[len(ordered) // 2]}  max {ordered[-1]}")
        print(f"  cases where this is shallower than the class rank: {divergent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
