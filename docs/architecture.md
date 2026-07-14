# Architecture

The only production flow is:

Query → Retriever → CandidateSet → Selector → SelectionResult
→ Pipeline resolves canonical evidence → Generator → GenerationResult.

## Retriever

Returns candidate evidence text, source, score, and rank.

## Selector

Returns selected evidence IDs, selection scores, and selection ranks.
It does not return evidence text, sufficiency status, missing facts, or conflict status.
Selector algorithms may use any approved internal features, but the external output stays small.

## Pipeline

Connects modules and resolves selected IDs back to the exact Retriever candidates.
It contains no retrieval, selection, or generation algorithm.

## Generator

Receives the selected canonical evidence and returns an answer plus cited evidence IDs.
Its answering, refusal, and uncertainty behavior belongs to Generator development.

## Composition

`composition.py` chooses the concrete Retriever, Selector, and Generator implementations
for one run and passes them into the unchanged Pipeline. It does not contain the connection
logic. Module teams can test their own component directly; composition is used for smoke,
integration, and controlled replacement runs.
