from evidence_rag.contracts.models import EvidenceCandidate, SelectedEvidenceSet
from evidence_rag.generator.attribution import Attributor, EntityMismatchRecord
from evidence_rag.generator.models import Claim, ClaimSpan
from evidence_rag.generator.nli import NLILabel

ANSWER = "Acme's revenue was $1.2B in 2024."


class FakeNLI:
    """Deterministic stand-in keyed by the premise (the evidence text)."""

    def __init__(self, labels: dict[str, NLILabel], default: NLILabel = "neutral") -> None:
        self.labels = labels
        self.default = default
        self.pairs: list[tuple[str, str]] = []

    def classify(self, premise: str, hypothesis: str) -> NLILabel:
        self.pairs.append((premise, hypothesis))
        return self.labels.get(premise, self.default)


def evidence(evidence_id: str, text: str) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=f"doc-{evidence_id}",
        chunk_id=f"chunk-{evidence_id}",
        text=text,
        source_uri=f"fixture://{evidence_id}",
        retrieval_score=1.0,
        retrieval_rank=1,
    )


def claim(
    claim_id: str = "claim-1",
    text: str = "Acme's revenue was $1.2B in 2024.",
    *,
    faithful_to_answer: bool = True,
) -> Claim:
    return Claim(
        claim_id=claim_id,
        text=text,
        span=ClaimSpan(start=0, end=len(ANSWER)),
        faithful_to_answer=faithful_to_answer,
    )


def selected(*items: EvidenceCandidate) -> SelectedEvidenceSet:
    return SelectedEvidenceSet(query_id="q-1", evidence=items)


def test_entailing_and_entity_consistent_evidence_supports_the_claim() -> None:
    support = evidence("ev-1", "Acme Inc. reported revenue of 1,200 million dollars in 2024.")
    attributor = Attributor(FakeNLI({support.text: "entailment"}))

    verification = attributor.verify_claim(claim(), selected(support))

    assert verification.status == "supported"
    assert verification.supporting_evidence_ids == ("ev-1",)
    assert verification.entity_consistent is True
    assert verification.contradicted is False


def test_entailing_but_entity_swapped_evidence_downgrades_the_claim() -> None:
    counterfactual = evidence("ev-1", "Globex reported revenue of $1.2B in 2024.")
    records: list[EntityMismatchRecord] = []
    attributor = Attributor(
        FakeNLI({counterfactual.text: "entailment"}),
        on_entity_mismatch=records.append,
    )

    verification = attributor.verify_claim(claim(), selected(counterfactual))

    assert verification.status == "unsupported"
    assert verification.supporting_evidence_ids == ()
    assert verification.entity_consistent is False
    assert [record.evidence_id for record in records] == ["ev-1"]
    assert records[0].mismatches[0].normalized == "acme"


def test_contradicting_evidence_is_recorded_and_leaves_the_claim_unsupported() -> None:
    against = evidence("ev-1", "Acme reported no revenue in 2024.")
    attributor = Attributor(FakeNLI({against.text: "contradiction"}))

    verification = attributor.verify_claim(claim(), selected(against))

    assert verification.status == "unsupported"
    assert verification.contradicted is True
    assert verification.entity_consistent is True


def test_all_neutral_evidence_leaves_the_claim_unsupported_without_entity_blame() -> None:
    attributor = Attributor(FakeNLI({}))

    verification = attributor.verify_claim(
        claim(),
        selected(evidence("ev-1", "Unrelated background."), evidence("ev-2", "More background.")),
    )

    assert verification.status == "unsupported"
    assert verification.entity_consistent is True
    assert verification.contradicted is False


def test_every_evidence_is_scored_so_contradiction_survives_an_earlier_support() -> None:
    support = evidence("ev-1", "Acme Inc. reported revenue of $1.2B in 2024.")
    against = evidence("ev-2", "Acme reported no revenue in 2024.")
    nli = FakeNLI({support.text: "entailment", against.text: "contradiction"})
    attributor = Attributor(nli)

    verification = attributor.verify_claim(claim(), selected(support, against))

    assert verification.status == "supported"
    assert verification.supporting_evidence_ids == ("ev-1",)
    assert verification.contradicted is True
    assert len(nli.pairs) == 2


def test_a_clean_support_keeps_the_claim_consistent_despite_another_swapped_evidence() -> None:
    support = evidence("ev-1", "Acme Inc. reported revenue of $1.2B in 2024.")
    swapped = evidence("ev-2", "Globex reported revenue of $1.2B in 2024.")
    records: list[EntityMismatchRecord] = []
    attributor = Attributor(
        FakeNLI({support.text: "entailment", swapped.text: "entailment"}),
        on_entity_mismatch=records.append,
    )

    verification = attributor.verify_claim(claim(), selected(support, swapped))

    # ClaimVerification requires a supported claim to be entity-consistent; the
    # per-pair mismatch is still observable through the sink.
    assert verification.status == "supported"
    assert verification.entity_consistent is True
    assert [record.evidence_id for record in records] == ["ev-2"]


def test_nli_is_asked_with_the_evidence_as_premise() -> None:
    support = evidence("ev-1", "Acme Inc. reported revenue of $1.2B in 2024.")
    nli = FakeNLI({support.text: "entailment"})

    Attributor(nli).verify_claim(claim(), selected(support))

    assert nli.pairs == [(support.text, claim().text)]


def test_unfaithful_claims_are_never_verified() -> None:
    support = evidence("ev-1", "Acme Inc. reported revenue of $1.2B in 2024.")
    nli = FakeNLI({support.text: "entailment"})

    verifications = Attributor(nli).verify_claims(
        (
            claim("claim-1"),
            claim("claim-2", "Acme was founded by Globex.", faithful_to_answer=False),
        ),
        selected(support),
    )

    assert [item.claim_id for item in verifications] == ["claim-1"]
    assert all(hypothesis == claim().text for _premise, hypothesis in nli.pairs)
