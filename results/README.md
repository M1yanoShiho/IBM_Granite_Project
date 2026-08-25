# Frozen aggregate results

This directory contains only the small final aggregate artifacts used by the dissertation and
public result documentation.

- `experiment04/`: frozen three-module comparison and module ablations.
- `experiment05/`: general RAG reliability/traceability comparison and registered claim decisions.

Raw generations, per-query scores, traces, indexes and intermediate snapshots are not stored in the
release tree. Their archive/recovery boundary is documented in
`docs/RELEASE_TREE_MAPPING.md` and the external artifact manifest prepared later in the release.

`FINAL PASS` in these files means the registered protocol completed. It does not mean the stronger
whole-system superiority claims were supported.
