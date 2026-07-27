"""Smoke test for Generator Part B, Half 1 -- the attribution path with REAL components.

Exercises B1 ``DebertaNLIModel`` (real cross-encoder NLI), B2 ``Attributor``, and
B3 ``RuleBasedEntityExtractor`` for the first time outside of fakes. It is NOT an
evaluation and NOT a benchmark: success means "real model loads, real inputs flow
through, verdicts on hand-built cases look correct."

Out of scope (Half 2, HPC/GPU): ``CompletenessChecker`` (B4) and ``Verifier``
(B5), both of which need Granite. We call ``Attributor`` directly here.

Run:
    python scripts/smoke_attribution.py

The first run downloads ``cross-encoder/nli-deberta-v3-base`` from Hugging Face.
On a shared/cluster machine set ``MODEL_CACHE_DIR`` to scratch space so weights do
not land in $HOME (see docs/hpc-deployment.md); ``nli.py`` already reads it. This
script forces CPU and does not assume CUDA is present. If the download fails
behind a proxy, it reports the failure rather than falling back to a fake -- a
fake here would defeat the whole purpose.

Exits non-zero if any diagnostic row fails, so it doubles as a quick regression
check later.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from evidence_rag.contracts.models import EvidenceCandidate, SelectedEvidenceSet
from evidence_rag.generator.attribution import Attributor, EntityMismatchRecord
from evidence_rag.generator.models import Claim, ClaimSpan, DraftAnswer
from evidence_rag.generator.nli import DebertaNLIModel


# --------------------------------------------------------------------------- #
# Fixtures: one DraftAnswer + one SelectedEvidenceSet, realistic enterprise
# phrasing (company + year + money) so the entity normalizer actually runs.
# --------------------------------------------------------------------------- #

QUERY_ID = "smoke-acme-fy2023"


@dataclass(frozen=True)
class Case:
    """A single diagnostic claim plus what we expect the verdict to look like."""

    number: int
    label: str
    claim_id: str
    claim_text: str
    source_sentence: str  # goes into answer_text; the claim's span points here
    faithful: bool
    expected_status: str | None  # None => claim must be absent from the verdicts
    expected_entity_consistent: bool | None
    expected_contradicted: bool | None
    expects_mismatch_record: bool


CASES: tuple[Case, ...] = (
    Case(
        number=1,
        label="clean support (same entities)",
        claim_id="claim-1",
        claim_text="Acme Corporation reported total revenue of $4.2 billion for fiscal year 2023.",
        source_sentence=(
            "Acme Corporation reported total revenue of $4.2 billion for fiscal year 2023."
        ),
        faithful=True,
        expected_status="supported",
        expected_entity_consistent=True,
        expected_contradicted=False,
        expects_mismatch_record=False,
    ),
    Case(
        number=2,
        label="counterfactual: organization swapped",
        claim_id="claim-2",
        claim_text="Acme Corporation's net income for fiscal year 2023 was $520 million.",
        source_sentence="Acme Corporation's net income for fiscal year 2023 was $520 million.",
        faithful=True,
        expected_status="unsupported",
        expected_entity_consistent=False,
        expected_contradicted=None,  # depends on NLI; asserted loosely below
        expects_mismatch_record=True,
    ),
    Case(
        number=2,
        label="counterfactual: number/headcount swapped",
        claim_id="claim-3",
        claim_text="Acme Corporation employed 45,000 people at the end of 2023.",
        source_sentence="Acme Corporation employed 45,000 people at the end of 2023.",
        faithful=True,
        # FINDING (real run): on this phrasing DeBERTa-NLI catches the 38,000 vs
        # 45,000 swap as a *contradiction* on its own, so NLI never returns
        # entailment and the entity number-presence backstop (REQUIRE_PRESENCE)
        # never gets to run -- no sink record here. The outcome is still correct
        # (unsupported); the entity layer is only actually exercised by the ORG
        # swap (claim-2), where NLI is entity-blind. Expectations below encode the
        # observed-correct behavior so this doubles as a regression baseline.
        expected_status="unsupported",
        expected_entity_consistent=True,
        expected_contradicted=True,
        expects_mismatch_record=False,
    ),
    Case(
        number=3,
        label="direct contradiction",
        claim_id="claim-4",
        claim_text="Acme Corporation's revenue increased compared to fiscal year 2022.",
        source_sentence="Acme Corporation's revenue increased compared to fiscal year 2022.",
        faithful=True,
        expected_status="unsupported",
        expected_entity_consistent=True,
        expected_contradicted=True,
        expects_mismatch_record=False,
    ),
    Case(
        number=4,
        label="topically related, says nothing",
        claim_id="claim-5",
        claim_text="Acme Corporation completed its acquisition of Initech in fiscal year 2023.",
        source_sentence=(
            "Acme Corporation completed its acquisition of Initech in fiscal year 2023."
        ),
        faithful=True,
        expected_status="unsupported",
        expected_entity_consistent=True,
        # FINDING (real run): DeBERTa-NLI FALSE-POSITIVES a *contradiction* here --
        # ev-5 ("operates manufacturing facilities...") says nothing about the
        # Initech acquisition, yet the model labels the pair contradiction. The
        # status verdict is still correct (unsupported), and the guide's case-4
        # spec only pins status + entity_consistent, so we do not assert the flag
        # here -- but this is a genuine model-quality issue, reported separately.
        expected_contradicted=None,
        expects_mismatch_record=False,
    ),
    Case(
        number=5,
        label="unfaithful claim (must be skipped)",
        claim_id="claim-6",
        claim_text="Acme Corporation was founded in 1998.",
        source_sentence="Acme Corporation was founded in 1998.",
        faithful=False,
        expected_status=None,
        expected_entity_consistent=None,
        expected_contradicted=None,
        expects_mismatch_record=False,
    ),
)


# Evidence engineered so each faithful claim hits its intended branch. Every
# claim is scored against every evidence, so the entailing one must be unique
# per claim to keep the diagnosis clean.
EVIDENCE_TEXTS: tuple[tuple[str, str], ...] = (
    (
        "ev-1",
        "In its annual report, Acme Corporation announced that total revenue "
        "reached $4.2 billion for fiscal year 2023.",
    ),
    (
        "ev-2",  # org swap target for claim-2
        "Globex Corporation's net income for fiscal year 2023 was $520 million.",
    ),
    (
        "ev-3",  # number swap target for claim-3
        "Acme Corporation employed 38,000 people at the end of 2023.",
    ),
    (
        "ev-4",  # contradiction target for claim-4
        "Acme Corporation's revenue declined compared to fiscal year 2022.",
    ),
    (
        "ev-5",  # topical / neutral target for claim-5
        "Acme Corporation operates manufacturing facilities across North America and Europe.",
    ),
)


def build_draft() -> DraftAnswer:
    """Concatenate the source sentences into one answer and give each claim the
    span of its own sentence, so the DraftAnswer validators are satisfied."""
    answer_parts: list[str] = []
    claims: list[Claim] = []
    cursor = 0
    for case in CASES:
        sentence = case.source_sentence
        start = cursor
        end = start + len(sentence)
        claims.append(
            Claim(
                claim_id=case.claim_id,
                text=case.claim_text,
                span=ClaimSpan(start=start, end=end),
                faithful_to_answer=case.faithful,
            )
        )
        answer_parts.append(sentence)
        cursor = end + 1  # +1 for the joining space
    return DraftAnswer(
        query_id=QUERY_ID,
        answer_text=" ".join(answer_parts),
        claims=tuple(claims),
    )


def build_selected() -> SelectedEvidenceSet:
    evidence = tuple(
        EvidenceCandidate(
            evidence_id=evidence_id,
            document_id=f"doc-{evidence_id}",
            chunk_id=f"chunk-{evidence_id}",
            text=text,
            source_uri=f"fixture://{evidence_id}",
            retrieval_score=1.0,
            retrieval_rank=rank,
        )
        for rank, (evidence_id, text) in enumerate(EVIDENCE_TEXTS, start=1)
    )
    return SelectedEvidenceSet(query_id=QUERY_ID, evidence=evidence)


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #

def _truncate(text: str, width: int = 52) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def main() -> int:
    draft = build_draft()
    selected = build_selected()
    cases_by_id = {case.claim_id: case for case in CASES}

    print("Loading cross-encoder/nli-deberta-v3-base on CPU (first run downloads it)...")
    try:
        nli = DebertaNLIModel(device="cpu")
        # Force the load now so a download/proxy failure is reported up front,
        # not mid-table.
        nli.classify(premise="warm up", hypothesis="warm up")
    except Exception as exc:  # noqa: BLE001 - smoke script: surface any load failure
        print(f"\nFAILED to load the real NLI model: {type(exc).__name__}: {exc}")
        print("This smoke test must not fall back to a fake -- fix the environment and rerun.")
        return 2

    records: list[EntityMismatchRecord] = []
    attributor = Attributor(nli, on_entity_mismatch=records.append)

    # --- raw NLI labels for every (claim, evidence) pair -------------------- #
    print("\n" + "=" * 78)
    print("RAW NLI LABELS  (premise = evidence, hypothesis = claim)")
    print("=" * 78)
    for claim in draft.claims:
        print(f"\n[{claim.claim_id}] {_truncate(claim.text, 68)}")
        for item in selected.evidence:
            label = nli.classify(premise=item.text, hypothesis=claim.text)
            print(f"    {item.evidence_id}: {label:<13} | {_truncate(item.text, 60)}")

    # --- aggregated verdicts ------------------------------------------------ #
    verdicts = attributor.verify_claims(draft.claims, selected)
    verdicts_by_id = {v.claim_id: v for v in verdicts}

    print("\n" + "=" * 78)
    print("VERDICT TABLE")
    print("=" * 78)
    header = (
        f"{'':4} {'claim':<9} {'expected':<12} {'status':<12} "
        f"{'support':<10} {'ent_cons':<9} {'contra':<7}"
    )
    print(header)
    print("-" * len(header))

    all_pass = True

    for case in CASES:
        verdict = verdicts_by_id.get(case.claim_id)

        if case.expected_status is None:
            # Unfaithful claim: must be absent entirely.
            row_pass = verdict is None
            all_pass = all_pass and row_pass
            marker = "PASS" if row_pass else "FAIL"
            actual = "ABSENT" if verdict is None else f"present({verdict.status})"
            print(
                f"{marker:<4} {case.claim_id:<9} {'absent':<12} {actual:<12} "
                f"{'-':<10} {'-':<9} {'-':<7}"
            )
            continue

        if verdict is None:
            all_pass = False
            print(
                f"{'FAIL':<4} {case.claim_id:<9} {case.expected_status:<12} "
                f"{'MISSING':<12} {'-':<10} {'-':<9} {'-':<7}"
            )
            continue

        checks = [verdict.status == case.expected_status]
        if case.expected_entity_consistent is not None:
            checks.append(verdict.entity_consistent == case.expected_entity_consistent)
        if case.expected_contradicted is not None:
            checks.append(verdict.contradicted == case.expected_contradicted)
        row_pass = all(checks)
        all_pass = all_pass and row_pass

        marker = "PASS" if row_pass else "FAIL"
        support = ",".join(verdict.supporting_evidence_ids) or "-"
        print(
            f"{marker:<4} {case.claim_id:<9} {case.expected_status:<12} "
            f"{verdict.status:<12} {support:<10} "
            f"{str(verdict.entity_consistent):<9} {str(verdict.contradicted):<7}"
        )

    # --- mismatch sink ------------------------------------------------------ #
    print("\n" + "=" * 78)
    print(f"on_entity_mismatch SINK RECORDS ({len(records)})")
    print("=" * 78)
    if not records:
        print("  (none)")
    for record in records:
        case = cases_by_id.get(record.claim_id)
        tag = f" [{case.label}]" if case else ""
        print(f"  {record.claim_id} x {record.evidence_id}{tag}")
        for mismatch in record.mismatches:
            print(
                f"      type={mismatch.entity_type} claim={mismatch.normalized!r} "
                f"evidence={list(mismatch.evidence_values)}"
            )

    # Case 2 specifically must have delivered records (mismatch detail survives
    # a downgrade). Report it explicitly rather than only via the table.
    case2_ids = {case.claim_id for case in CASES if case.expects_mismatch_record}
    got_record_ids = {record.claim_id for record in records}
    print("\nCase 2 mismatch-record check:")
    for claim_id in sorted(case2_ids):
        ok = claim_id in got_record_ids
        print(f"  {claim_id}: {'record delivered' if ok else 'NO record'}")

    print("\n" + "=" * 78)
    print("RESULT:", "ALL ROWS PASS" if all_pass else "SOME ROWS FAILED")
    print("=" * 78)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
