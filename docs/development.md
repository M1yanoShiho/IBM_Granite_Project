# Parallel development

## Ownership

- Retriever team: `src/evidence_rag/retriever` and `tests/retriever`.
- Selector team: `src/evidence_rag/selector`, `tests/selector`, and `docs/selector`.
- Generator team: `src/evidence_rag/generator` and `tests/generator`.
- Integration owner: contracts, pipeline, composition, and architecture tests.

## Independent work

- Retriever developers test against Query and Document fixtures and return CandidateSet.
- Selector developers test against a fixed CandidateSet without waiting for a real Retriever.
- Generator developers test against a fixed SelectedEvidenceSet without waiting for Selector.
- Fixtures prove interface compatibility; they are not model-performance evidence.

## Merge rule

A module improvement may be connected to the baseline only when:

1. it keeps the current Protocol;
2. its own tests pass;
3. the module swap tests pass;
4. the complete Pipeline runs;
5. both module metrics and complete-Pipeline metrics are reported.

Contract changes require agreement from all three module teams.
