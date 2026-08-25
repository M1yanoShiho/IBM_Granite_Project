"""QAMPARI plumbing dry run: schema, counts, lengths, and claim splitting.

Same discipline as the three-set dry run -- **no metric is computed and no
generated output is inspected**. The one deliberate exception is Task 3's
largest unknown, which cannot be answered any other way:

  * **claim splitting on entity lists.** QAMPARI answers enumerate entities. One
    atomic claim per entity verifies well; one claim carrying five entities needs
    all five supported and would depress everything. The splitter is run over
    *gold* answers -- not over anything this system generated -- so what is being
    looked at is a property of the dataset and the splitter, not a result.

    On CPU with no model weights this uses a stub LLM, so it measures the
    splitter's ANCHORING and suppression behaviour rather than Granite's split
    quality; the GPU pass repeats it with the real model.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from heldout_dryrun import check_schema, to_selected  # noqa: E402
from qampari_data import load_qampari  # noqa: E402

from evidence_rag.contracts.models import Query, QueryChecklist  # noqa: E402


def _profile(rows: list[dict[str, Any]], top_k: int) -> None:
    passages = [len(r["passages"]) for r in rows]
    gold = [len(r["gold_answers"]) for r in rows]
    words = [sum(len(p["text"].split()) for p in r["passages"][:top_k]) for r in rows]
    per_passage = [len(p["text"].split()) for r in rows for p in r["passages"][:top_k]]
    question = [len(r["question"].split()) for r in rows]

    def line(name: str, values: list[int]) -> None:
        ordered = sorted(values)
        print(
            f"  {name:26s} min={min(values):5d} median={statistics.median(values):7.1f} "
            f"p90={ordered[int(0.9 * len(ordered))]:6d} max={max(values):6d}",
            flush=True,
        )

    print(f"  records: {len(rows)}", flush=True)
    line("passages per record", passages)
    line("gold answers per record", gold)
    line(f"words in top-{top_k}", words)
    line("words per passage", per_passage)
    line("question words", question)
    many = sum(1 for g in gold if g > 5)
    print(
        f"  records with >5 gold answers: {many} ({many / len(rows):.1%}) "
        "-- why the pre-registered metric also reports rec@5",
        flush=True,
    )


class _StubLLM:
    """Returns a splitter-shaped payload built from the text, so the splitter's
    anchoring and suppression logic can be exercised without model weights."""

    def __init__(self) -> None:
        self.mode = "split"

    def generate(self, prompt: str) -> str:
        if prompt.lstrip().startswith("Check whether"):
            items = json.loads(prompt.split("Items:\n", 1)[1])
            return json.dumps(
                {"results": [{"claim_id": i["claim_id"], "faithful": True} for i in items]}
            )
        answer = prompt.split("Answer:\n", 1)[1].strip()
        from evidence_rag.contracts.models import split_sentences

        parts = split_sentences(answer) or [answer]
        return json.dumps({"claims": [{"source_text": p, "text": p} for p in parts]})


def _claim_split_probe(rows: list[dict[str, Any]], n: int) -> None:
    """Run the splitter over QAMPARI GOLD answers rendered as a sentence.

    Not over generated text: nothing this system produced is read here.
    """
    from evidence_rag.generator.claim_splitter import ClaimSplitter

    splitter = ClaimSplitter(llm=_StubLLM())
    print("\n=== claim splitting on entity-list answers (gold, stub LLM) ===", flush=True)
    counts: list[tuple[int, int]] = []
    for row in rows[:n]:
        entities = [aliases[0] for aliases in row["gold_answers"]]
        rendered = ", ".join(entities) + "."
        claims = splitter.split(rendered)
        counts.append((len(entities), len(claims)))
        if len(counts) <= 5:
            print(f"\n  question: {row['question'][:80]!r}")
            print(f"  {len(entities)} gold entities -> {len(claims)} claims")
            for claim in claims[:3]:
                print(f"      {claim.text[:110]!r}")
    if counts:
        ratio = [c / e for e, c in counts if e]
        print(
            f"\n  over {len(counts)} probes: entities/record median "
            f"{statistics.median([e for e, _ in counts]):.1f}, claims median "
            f"{statistics.median([c for _, c in counts]):.1f}, claims-per-entity median "
            f"{statistics.median(ratio):.2f}",
            flush=True,
        )
    print(
        "  NOTE: with a stub LLM this shows anchoring/suppression only; the GPU\n"
        "  pass repeats it with Granite doing the actual splitting.",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--probe", type=int, default=20)
    parser.add_argument("--schema-only", action="store_true")
    args = parser.parse_args()

    print("=== qampari ===", flush=True)
    rows = load_qampari(top_k=args.top_k)
    problems = check_schema("qampari", rows)
    _profile(rows, args.top_k)
    print(f"  schema problems: {problems or 'none'}", flush=True)

    for row in rows[: args.slice]:
        selected = to_selected(row, args.top_k)
        Query(query_id=str(row["query_id"]), text=row["question"])
        QueryChecklist(query_id=str(row["query_id"]), focus=row["question"], required_facts=())
        assert len(selected.evidence) == min(args.top_k, len(row["passages"]))
    print(f"  contract objects built for {args.slice} records: ok", flush=True)

    _claim_split_probe(rows, args.probe)

    if not args.schema_only:
        from evidence_rag.generator.granite import GraniteGenerator, GraniteLLMClient
        from evidence_rag.generator.nli import build_nli_model
        from evidence_rag.generator.verify_annotate import VerifyAnnotateGenerator

        print("\n  [models] loading Granite + TRUE", flush=True)
        llm = GraniteLLMClient()
        nli = build_nli_model("true")
        arms = {
            "baseline": GraniteGenerator(llm=llm),
            "verify-annotate-nogate": VerifyAnnotateGenerator(
                llm=llm, nli=nli, entity_gate="observe"
            ),
        }
        tokenizer = getattr(llm, "_tokenizer", None)
        for arm, generator in arms.items():
            ok = 0
            lengths: list[int] = []
            for row in rows[: args.slice]:
                try:
                    result = generator.generate(
                        Query(query_id=str(row["query_id"]), text=row["question"]),
                        QueryChecklist(
                            query_id=str(row["query_id"]),
                            focus=row["question"],
                            required_facts=(),
                        ),
                        to_selected(row, args.top_k),
                    )
                    assert result.query_id == str(row["query_id"])
                    ok += 1
                    # length only -- the text itself is not printed
                    if tokenizer is not None and result.answer:
                        lengths.append(len(tokenizer.encode(result.answer)))
                except Exception as exc:  # noqa: BLE001 - this is the check
                    print(f"  {arm} FAILED: {type(exc).__name__}: {exc}", flush=True)
            note = (
                f", answer tokens max={max(lengths)} (cap 256)" if lengths else ""
            )
            print(f"  {arm}: {ok}/{args.slice} valid result objects{note}", flush=True)
            # Task 3's real question, answerable only with the real model: does a
            # many-answer question yield one claim per entity, or one claim
            # carrying several? Counts only -- no claim text is printed.
            routings = getattr(generator, "last_routings", None)
            if routings:
                print(
                    f"      last answer: {len(routings)} claims routed, "
                    f"outcomes={ {o: [r.outcome for r in routings].count(o) for o in {r.outcome for r in routings}} }",
                    flush=True,
                )

    print("\nplumbing OK -- no metrics computed, no generated output inspected", flush=True)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
