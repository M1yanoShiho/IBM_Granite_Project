"""Run the trained relation model over a REAL top-20 window, not over isolated pairs.

    export PYTHONPATH=src
    python scripts/pool_relation_probe.py \
      --checkpoint runs/r013/seed-13/fold-0 \
      --dataset-manifest runs/niah-injected/manifest.json \
      --provenance runs/niah-injected/provenance.jsonl \
      --candidates runs/e2-gate-off/candidate_sets.jsonl \
      --output results/r013/pool-probe-seed13.json

Everything R013 has been judged on so far is one passage against one claim. The OOF is pairs,
the surface baseline was pairs, the ablation was pairs, and Gate 0B's 0B-2 probe is pairs too --
so passing that gate would not answer this either. S6 is the standing reason to care: a method
that looked good on a two-passage probe put false conflicts up 38.3 points once it met a real
twenty-passage pool, because the eighteen irrelevant passages were forced to take a position.

So this asks the pool-shaped question. For each query, one claim built from the question and its
gold answer -- the same `build_hypothesis` the training rows used -- scored against ALL twenty
retrieved passages, and the reading is what happens to the eighteen that are neither the needle
nor its twin. They should be UNKNOWN. Every one that comes back SUPPORTS or REFUTES is a false
conflict, which is the mechanism S3 measured as recall loss.

There is a reason to expect this to hurt: UNKNOWN is the model's weakest class (F1 .7885), and
it is ~14% of the training chain against roughly ninety percent of a real window. The training
distribution and the deployment distribution disagree exactly where the model is weakest.

DIAGNOSTIC, NOT A GATE READING. One fold of one seed, scored on the dev pool, reported as a rate
rather than against a threshold. Gate 0B is a separate, once-only measurement with its own
registry line, and nothing here substitutes for it or licenses skipping it.
"""

import argparse
import collections
import importlib
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evidence_rag.cli.train_relations import _score_fn  # noqa: E402
from evidence_rag.contracts.models import CandidateSet  # noqa: E402
from evidence_rag.evaluation.harm import provenance_harm_map  # noqa: E402
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter  # noqa: E402
from evidence_rag.materializer.provenance import read_provenance  # noqa: E402
from evidence_rag.relations.claims import build_hypothesis  # noqa: E402
from evidence_rag.relations.predictor import NLIRelationPredictor  # noqa: E402
from evidence_rag.relations.training import derive_label_order  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, type=Path, help="one fold directory")
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="queries to score; omit for all")
    arguments = parser.parse_args(argv)

    bundle = JsonlDatasetAdapter.load(arguments.dataset_manifest)
    question_by_query = {query.query_id: query.text for query in bundle.queries}
    gold_by_query = {case.query_id: case for case in bundle.gold_cases}
    twin_by_query = provenance_harm_map(read_provenance(arguments.provenance))

    pools = [
        CandidateSet.model_validate_json(line)
        for line in arguments.candidates.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    transformers = importlib.import_module("transformers")
    torch = importlib.import_module("torch")
    cross_encoder = importlib.import_module("sentence_transformers.cross_encoder")
    model = cross_encoder.CrossEncoder(str(arguments.checkpoint))
    saved = transformers.AutoConfig.from_pretrained(str(arguments.checkpoint))
    label_order = derive_label_order(
        {int(key): str(value) for key, value in saved.id2label.items()}
    )
    predictor = NLIRelationPredictor(
        score_fn=_score_fn(model, torch, label_order),
        model_version=f"pool-probe/{arguments.checkpoint.name}",
    )
    print(f"label_order {label_order}", flush=True)

    roles = ("needle", "twin", "other")
    counts: dict[str, collections.Counter[str]] = {role: collections.Counter() for role in roles}
    scored_queries = 0
    skipped = collections.Counter[str]()

    for pool in pools:
        if arguments.limit is not None and scored_queries >= arguments.limit:
            break
        gold = gold_by_query.get(pool.query_id)
        question = question_by_query.get(pool.query_id)
        twin_id = twin_by_query.get(pool.query_id)
        # Every skip is counted and reported. A query dropped in silence would quietly change the
        # denominator, which is the failure this repo keeps finding rather than the loud kind.
        if question is None or gold is None or not gold.reference_answers:
            skipped["no question or gold answer"] += 1
            continue
        if not gold.relevant_document_ids:
            skipped["no labelled relevant document"] += 1
            continue
        if twin_id is None:
            skipped["no injected twin"] += 1
            continue

        claim = build_hypothesis(question, gold.reference_answers[0])
        needles = set(gold.relevant_document_ids)
        candidates = tuple(sorted(pool.candidates, key=lambda item: item.retrieval_rank))
        predictions = predictor.predict([(item.text, claim) for item in candidates])
        for item, prediction in zip(candidates, predictions, strict=True):
            if item.document_id == twin_id:
                role = "twin"
            elif item.document_id in needles:
                role = "needle"
            else:
                role = "other"
            counts[role][prediction.label.value] += 1
        scored_queries += 1
        if scored_queries % 100 == 0:
            print(f"  scored {scored_queries} queries", flush=True)

    report = {
        "checkpoint": str(arguments.checkpoint),
        "label_order": list(label_order),
        "n_queries_scored": scored_queries,
        "skipped": dict(skipped),
        "by_role": {role: dict(counts[role]) for role in roles},
    }
    other = counts["other"]
    other_total = sum(other.values())
    if other_total:
        false_conflict = 1.0 - other.get("UNKNOWN", 0) / other_total
        report["other_n"] = other_total
        report["other_unknown_rate"] = other.get("UNKNOWN", 0) / other_total
        report["false_conflict_rate"] = false_conflict
        report["mean_false_conflicts_per_window"] = false_conflict * other_total / scored_queries
    for role in roles:
        total = sum(counts[role].values())
        if not total:
            continue
        shares = ", ".join(
            f"{label} {count / total:6.2%}" for label, count in sorted(counts[role].items())
        )
        print(f"{role:<8} n={total:<7} {shares}")
    if other_total:
        print(
            f"\nfalse-conflict rate on the non-needle non-twin passages: "
            f"{report['false_conflict_rate']:.4f}"
            f"   (~{report['mean_false_conflicts_per_window']:.2f} per window)"
        )
    print(
        "\nREAD: these passages should be UNKNOWN. Every SUPPORTS or REFUTES among them is a"
        " false conflict,\n      which is the mechanism S3 measured as recall loss and S6"
        " measured as a 38.3pt collapse.\n      Diagnostic only: one fold, one seed, dev pool."
        " It is not a Gate 0B reading."
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {arguments.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
