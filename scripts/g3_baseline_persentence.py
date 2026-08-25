"""G3 remediation Task 2 -- baseline variant that cites per sentence.

ALCE's setting asks the model to place citations at the end of each sentence; our
baseline emits one trailing `Evidence: [n], [m]` line for the whole answer. The
G3 comparison was therefore between two different citation *conventions*, not
just two systems. This arm fixes that: same model, same decoding, same evidence,
same queries, same case construction -- only the citation convention changes.

This is the one deliberate per-arm prompt change the remediation guide allows,
and it applies to the baseline alone. The original baseline is kept and reported
alongside it, because silently switching to whichever convention is more
favourable would be dishonest.

Needs Granite only (no TRUE, no verifier), so it is cheap: 400 generations.

HOLD-OUT: ALCE/ASQA only.

Usage:
  PYTHONPATH=src python scripts/g3_baseline_persentence.py \
    --limit 400 --seed 13 --dump results/g3-remediation/baseline-persentence.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import g3_baseline_comparison as g3  # noqa: E402, I001  identical case construction
from evidence_rag.generator.granite import GraniteLLMClient  # noqa: E402

# Minimal edit of CITATION_RAG_PROMPT: only the citation convention differs.
PER_SENTENCE_CITATION_PROMPT = (
    "Answer the question using only the evidence below.\n"
    "End every sentence of your answer with the bracketed numbers of the evidence "
    "that supports that sentence, for example: The sky is blue [1][3].\n\n"
    "Use this exact format:\n"
    "Answer: <answer, with citations at the end of every sentence>\n\n"
    "If the evidence does not contain the answer, use:\n"
    "Answer: I don't know\n\n"
    "Evidence:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

UNKNOWN = {"", "unknown", "i don't know", "i do not know", "not in the evidence"}


def _strip_answer_prefix(raw: str) -> str:
    text = raw.strip()
    if text.lower().startswith("answer:"):
        text = text[len("answer:") :].strip()
    # drop any trailing whole-answer Evidence: line the model emits out of habit
    return re.split(r"\n\s*Evidence:\s*", text, maxsplit=1)[0].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dump", type=Path, required=True)
    args = parser.parse_args()

    print("[data] loading ALCE/ASQA", flush=True)
    cases = g3.build_cases(g3.vt.ensure_asqa(), args.limit, random.Random(args.seed), top_k=args.top_k)
    print(f"[data] built {len(cases)} cases (identical construction to G3)", flush=True)

    llm = GraniteLLMClient()
    rows = []
    answered = 0
    for n, case in enumerate(cases, start=1):
        context = "\n".join(
            f"[{index}] ({item.evidence_id}) {item.text}"
            for index, item in enumerate(case.selected.evidence, start=1)
        )
        prompt = PER_SENTENCE_CITATION_PROMPT.format(context=context, question=case.question)
        try:
            answer = _strip_answer_prefix(llm.generate(prompt))
        except Exception as exc:  # noqa: BLE001 -- record, keep the sample aligned
            answer = ""
            print(f"[warn] {case.query_id}: {type(exc).__name__}", flush=True)
        if answer.strip().lower().strip(".!?\"' ") in UNKNOWN:
            answer = ""
        if answer:
            answered += 1
        rows.append(
            {
                "arm": "baseline-persentence",
                "query_id": case.query_id,
                "question": case.question,
                "answer": answer,
                "evidence": [
                    {"evidence_id": item.evidence_id, "text": item.text}
                    for item in case.selected.evidence
                ],
            }
        )
        if n % 50 == 0:
            print(f"[gen] {n}/{len(cases)}", flush=True)

    args.dump.parent.mkdir(parents=True, exist_ok=True)
    with args.dump.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[done] {answered}/{len(rows)} answered -> {args.dump}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
