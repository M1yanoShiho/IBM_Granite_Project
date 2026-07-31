"""Relation layer: claim/passage relation prediction for Graph 2.0.

Deliberately does NOT import from `evidence_rag.selector` — the Relation Builder must run
standalone for Gate 0B, where there is no selector. The selector consumes this package, never
the other way round.
"""
