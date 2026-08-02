"""G3 remediation Task 1 (+2) -- re-score the existing G3 run at sentence level.

The answer-level metric used in G3 judges every citation against the *whole*
answer, so entailment gets strictly harder as answers lengthen -- and the arms
differ systematically on length (median 97 vs 54 chars) and citation count (1.94
vs 1.24). The follow-up diagnosis showed the same draft citations score 0.869
against the claim they support and 0.539 against the whole answer. This pass
re-scores with ALCE's sentence-level definitions to size that artifact.

Metrics come from `alce_metrics.py`, a verified port of `compute_autoais` in
princeton-nlp/ALCE. Judge is **MiniCheck**; TRUE is excluded because it selected
the verified arms' citations.

Arms are RECONSTRUCTED from the existing dumps -- no chain is re-run:

  baseline       g3-human-subsample.jsonl (373 answered) -- answer + cited chunks
  verify-only    diagnosis dump: repaired_answer + draft_origin_citations, over
                 every case with a non-empty repaired answer INCLUDING those that
                 errored later in completeness/recheck, which verify-only never
                 reaches. Validated: this yields exactly 218, G3's reported count.
  verified-full  diagnosis dump: final_answer (repaired + recheck fragments), 125.

Baseline citations are answer-level, so a citation-to-sentence mapping is needed
and the choice is not neutral. Both are produced:
  (a) generous-to-baseline: every baseline citation applies to every sentence --
      lenient toward the arm we are trying to beat, so a win under it is the
      conservative result;
  (b) matched protocol: the per-sentence-citation baseline variant (Task 2),
      which emits inline [n] markers parsed exactly as ALCE does.

HOLD-OUT: ALCE/ASQA only. HotpotQA, RGB, MuSiQue-Full never loaded.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from alce_metrics import ScoredExample, compute_citation_metrics  # noqa: E402, I001

CITE_RE = re.compile(r"\[(\d+)")
STOPWORDS = frozenset(
    "a an the of in on at to for and or is are was were be been by with as that this it its "
    "from has have had also his her their he she they there which who whom but not into than "
    "then when while about over under after before".split()
)
CLAIM_OVERLAP = 0.5
"""Share of a claim's content tokens that must appear in a sentence before the
claim's citations are attached to it. Mapping errors mostly *remove* support, so
this stays conservative toward the verified arms."""


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    }


_FALLBACK_SENT = re.compile(r"[^.!?]*[.!?]+|\S[^.!?]*$")
SPLITTER_USED = "unset"


def sent_split(text: str) -> list[str]:
    """ALCE uses NLTK `sent_tokenize`. One splitter is applied identically to every
    arm; which one was available is recorded in the report so the choice is
    visible rather than silent."""
    global SPLITTER_USED
    try:
        from nltk.tokenize import sent_tokenize
    except Exception:  # noqa: BLE001 - environment-dependent; fall back, but say so
        SPLITTER_USED = "regex-fallback"
        return [m.group().strip() for m in _FALLBACK_SENT.finditer(text) if m.group().strip()]
    SPLITTER_USED = "nltk.sent_tokenize"
    return [s.strip() for s in sent_tokenize(text) if s.strip()]


def remove_citations(sentence: str) -> str:
    """ALCE's `remove_citations`: the judge sees prose, not bracket markers."""
    return re.sub(r"\[\d+", " ", re.sub(r" \[\d+", " ", sentence)).replace(" |", "").replace("]", "")


def _normalize_answer_part(text: str) -> str:
    """Mirrors VerifiedGenerator._normalize_answer_part so segments can be located."""
    normalized = " ".join(text.split())
    if normalized and normalized[-1] not in ".!?":
        normalized = f"{normalized}."
    return normalized


# --------------------------------------------------------------------------
# reconstruction
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Part:
    """A stretch of the answer as VerifiedGenerator assembled it. Parts are joined
    verbatim with single spaces, so the reconstruction is byte-identical to the
    recorded answer; sentence splitting happens afterwards on the joined text,
    which keeps it ALCE-faithful (ALCE splits the whole output)."""

    text: str
    kind: str  # "repaired" (citations resolved per sentence) | "fragment" (fixed)
    citations: tuple[str, ...] = ()


def _claims_for_sentence(record: dict[str, Any], sentence: str) -> tuple[str, ...]:
    """Citations for a sentence of the repaired answer.

    Every sentence of the repaired answer came from some trusted claim (repair
    emits only trusted claim spans), so the best-overlapping claim always
    attributes -- a threshold alone strands sentences whose claim was *rewritten*
    away from the source span ("They were inducted..." vs "The Everly Brothers
    received their induction..."). Additional claims attach only when clearly
    present, which is how a sentence merging two claims gets both citations.
    """
    trusted = [v for v in record["verified_claims"] if v["status"] == "supported"]
    by_id = {c["claim_id"]: c["text"] for c in record["draft_claims"]}
    tokens = _content_tokens(sentence)
    scored: list[tuple[float, dict[str, Any]]] = []
    for verification in trusted:
        claim_tokens = _content_tokens(by_id.get(verification["claim_id"], ""))
        if claim_tokens:
            scored.append((len(claim_tokens & tokens) / len(claim_tokens), verification))
    if not scored:
        return ()
    best_index = max(range(len(scored)), key=lambda i: scored[i][0])
    refs: list[str] = []
    for index, (overlap, verification) in enumerate(scored):
        if index == best_index or overlap >= CLAIM_OVERLAP:
            for evidence_id in verification["supporting_evidence_ids"]:
                if evidence_id not in refs:
                    refs.append(evidence_id)
    return tuple(refs)


def _build_parts(record: dict[str, Any], include_fragments: bool) -> list[Part]:
    """Replicate VerifiedGenerator's assembly, fragment dedupe included."""
    repaired = _normalize_answer_part(record["repaired_answer"])
    parts: list[Part] = []
    if repaired:
        parts.append(Part(repaired, "repaired"))
    if not include_fragments:
        return parts
    seen = {repaired.casefold()} if repaired else set()
    for gap in record["gaps"]:
        if not gap.get("recheck_found"):
            continue
        fragment = _normalize_answer_part(gap.get("recheck_fragment", ""))
        if not fragment or fragment.casefold() in seen:
            continue
        seen.add(fragment.casefold())
        parts.append(Part(fragment, "fragment", tuple(gap.get("recheck_evidence_ids", []))))
    return parts


def _score_parts(
    record: dict[str, Any], parts: list[Part]
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...], str]:
    joined = " ".join(part.text for part in parts if part.text)
    if not joined.strip():
        return (), (), joined
    bounds: list[tuple[int, int, Part]] = []
    cursor = 0
    for part in parts:
        if not part.text:
            continue
        bounds.append((cursor, cursor + len(part.text), part))
        cursor += len(part.text) + 1  # the joining space

    sentences: list[str] = []
    citations: list[tuple[str, ...]] = []
    search_from = 0
    for sentence in sent_split(joined):
        start = joined.find(sentence, search_from)
        if start < 0:
            start = search_from
        search_from = start + len(sentence)
        owner = next((part for begin, end, part in bounds if begin <= start < end), None)
        if owner is None:
            refs: tuple[str, ...] = ()
        elif owner.kind == "fragment":
            refs = owner.citations
        else:
            refs = _claims_for_sentence(record, sentence)
        sentences.append(remove_citations(sentence).strip())
        citations.append(refs)
    return tuple(sentences), tuple(citations), joined


def reconstruct_verify_only(records: list[dict[str, Any]]) -> list[ScoredExample]:
    examples: list[ScoredExample] = []
    for record in records:
        # verify-only never runs completeness/recheck, so cases that errored THERE
        # still produced a valid verify-only answer.
        if not record["repaired_answer"].strip():
            continue
        sentences, citations, _ = _score_parts(record, _build_parts(record, include_fragments=False))
        if not sentences:
            continue
        docs = {item["evidence_id"]: item["text"] for item in record["evidence"]}
        examples.append(ScoredExample(record["query_id"], sentences, citations, docs))
    return examples


def reconstruct_verified_full(records: list[dict[str, Any]]) -> tuple[list[ScoredExample], int]:
    """Returns the examples plus how many rebuilt answers are byte-identical to the
    recorded `final_answer` -- a check that the assembly was replicated exactly."""
    examples: list[ScoredExample] = []
    exact = 0
    for record in records:
        if not record["answered"]:
            continue
        sentences, citations, joined = _score_parts(
            record, _build_parts(record, include_fragments=True)
        )
        if joined == record["final_answer"]:
            exact += 1
        if not sentences:
            continue
        docs = {item["evidence_id"]: item["text"] for item in record["evidence"]}
        examples.append(ScoredExample(record["query_id"], sentences, citations, docs))
    return examples, exact


def reconstruct_baseline_generous(rows: list[dict[str, Any]]) -> list[ScoredExample]:
    """Convention (a): every baseline citation applies to every baseline sentence."""
    examples: list[ScoredExample] = []
    for row in rows:
        if row["arm"] != "baseline" or not row["answer"].strip():
            continue
        docs = {c["evidence_id"]: c["text"] for c in row["cited_chunks"]}
        refs = tuple(docs)
        sentences = tuple(remove_citations(s).strip() for s in sent_split(row["answer"]))
        if not sentences:
            continue
        examples.append(
            ScoredExample(row["query_id"], sentences, tuple(refs for _ in sentences), docs)
        )
    return examples


def reconstruct_baseline_persentence(rows: list[dict[str, Any]]) -> list[ScoredExample]:
    """Convention (b): inline [n] markers, parsed exactly as ALCE parses them."""
    examples: list[ScoredExample] = []
    for row in rows:
        if not row["answer"].strip():
            continue
        docs = {c["evidence_id"]: c["text"] for c in row["evidence"]}
        order = [c["evidence_id"] for c in row["evidence"]]
        sentences: list[str] = []
        citations: list[tuple[str, ...]] = []
        for sentence in sent_split(row["answer"]):
            refs: list[str] = []
            for match in CITE_RE.findall(sentence):
                index = int(match) - 1
                if 0 <= index < len(order) and order[index] not in refs:
                    refs.append(order[index])
            sentences.append(remove_citations(sentence).strip())
            citations.append(tuple(refs))
        if not sentences:
            continue
        examples.append(ScoredExample(row["query_id"], tuple(sentences), tuple(citations), docs))
    return examples


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

OLD_ANSWER_LEVEL = {
    "baseline": (0.602, 0.646, 373),
    "verify-only": (0.762, 0.862, 218),
    "verified-full": (0.575, 0.736, 125),
}


def render(results: dict[str, Any], args: argparse.Namespace, checks: dict[str, Any]) -> str:
    lines = ["# G3 remediation Task 1 — sentence-level re-scoring", ""]
    lines.append(
        "Re-scoring of the existing G3 run with ALCE's sentence-level citation "
        "metrics. No chain was re-run; the arms are reconstructed from the G3 and "
        "diagnosis dumps. Judge: **MiniCheck** (TRUE excluded — it selected the "
        "verified arms' citations)."
    )
    lines.append("")
    lines.append("**Hold-out respected:** ALCE/ASQA only.")
    lines.append("")
    lines.append("## Reconstruction checks")
    lines.append("")
    for key, value in checks.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Answer-level (G3) vs sentence-level (ALCE) — the artifact size")
    lines.append("")
    lines.append("| arm | n | answer-level prec | sentence-level prec | Δ | answer-level rec | sentence-level rec | Δ |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for arm, report in results.items():
        old = OLD_ANSWER_LEVEL.get(arm)
        new_p = report.citation_prec / 100
        new_r = report.citation_rec / 100
        if old:
            lines.append(
                f"| {arm} | {report.n_examples} | {old[0]:.3f} | {new_p:.3f} | "
                f"{new_p - old[0]:+.3f} | {old[1]:.3f} | {new_r:.3f} | {new_r - old[1]:+.3f} |"
            )
        else:
            lines.append(
                f"| {arm} | {report.n_examples} | — | {new_p:.3f} | — | — | {new_r:.3f} | — |"
            )
    lines.append("")
    lines.append("## Diagnostics")
    lines.append("")
    lines.append("| arm | sentences | citations | multi-cite sentences | supported | overcite |")
    lines.append("|---|---|---|---|---|---|")
    for arm, report in results.items():
        lines.append(
            f"| {arm} | {report.n_sentences} | {report.n_citations} | {report.sent_mcite} | "
            f"{report.sent_mcite_support} | {report.sent_mcite_overcite} |"
        )
    lines.append("")
    lines.append(
        f"Seed {args.seed}. Metrics implemented in `scripts/alce_metrics.py`. "
        f"Sentence splitter actually used: `{SPLITTER_USED}` (ALCE uses "
        "`nltk.sent_tokenize`), applied identically to every arm."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--baseline-subsample", type=Path, required=True)
    parser.add_argument("--baseline-persentence", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dump", type=Path, default=None)
    args = parser.parse_args()

    def load(path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    diagnosis = load(args.diagnosis)
    subsample = load(args.baseline_subsample)

    arms: dict[str, list[ScoredExample]] = {}
    arms["baseline"] = reconstruct_baseline_generous(subsample)
    arms["verify-only"] = reconstruct_verify_only(diagnosis)
    verified_full, exact = reconstruct_verified_full(diagnosis)
    arms["verified-full"] = verified_full
    if args.baseline_persentence is not None and args.baseline_persentence.exists():
        arms["baseline-persentence"] = reconstruct_baseline_persentence(load(args.baseline_persentence))

    checks = {
        "baseline examples (G3 reported 373 answered)": len(arms["baseline"]),
        "verify-only examples (G3 reported 218 answered)": len(arms["verify-only"]),
        "verified-full examples (G3 reported 125 answered)": len(arms["verified-full"]),
        "verified-full answers rebuilt byte-identical to the recorded final_answer":
            f"{exact}/{len(arms['verified-full'])}",
        "sentences with no citation attached (verify-only)":
            sum(1 for e in arms["verify-only"] for c in e.citations if not c),
        "sentences with no citation attached (verified-full)":
            sum(1 for e in arms["verified-full"] for c in e.citations if not c),
    }
    for key, value in checks.items():
        print(f"[check] {key}: {value}", flush=True)

    from evidence_rag.generator.nli import build_nli_model

    print("[judge] loading MiniCheck", flush=True)
    minicheck = build_nli_model("minicheck")

    def entails(premise: str, hypothesis: str) -> bool:
        return minicheck.classify(premise=premise, hypothesis=hypothesis) == "entailment"

    results: dict[str, Any] = {}
    for arm, examples in arms.items():
        print(f"[score] {arm}: {len(examples)} examples", flush=True)
        results[arm] = compute_citation_metrics(examples, entails)
        print(
            f"[score] {arm}: prec={results[arm].citation_prec:.1f} rec={results[arm].citation_rec:.1f}",
            flush=True,
        )

    report = render(results, args, checks)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    print("\n" + report, flush=True)

    if args.dump is not None:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with args.dump.open("w", encoding="utf-8") as handle:
            for arm, report_obj in results.items():
                for row in report_obj.per_example:
                    handle.write(json.dumps({"arm": arm, **row}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
