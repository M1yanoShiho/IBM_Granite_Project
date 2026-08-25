# Models and data

The repository intentionally contains no trained weights, upstream model snapshots, raw datasets,
indexes, or per-query generation bundles. `ARTIFACT_MANIFEST.json` is the machine-readable source
for frozen revisions, byte sizes, SHA-256 digests, licenses, and availability.

## What is publicly downloadable

The Retriever base models, Generator base model, Selector backbone, TRUE verifier, and MiniCheck
scorer remain on their official Hugging Face repositories at immutable revisions. The manifest
links each exact weight file and records the byte size and digest observed from that revision.

Provence is link-only. Its Hub metadata, model card, and bundled license file use inconsistent
Creative Commons labels and additional terms. This project therefore does not mirror it, does not
grant commercial rights, and asks users to review the upstream license file before use.

The final derived assets are not public downloads:

- `selector-seed13`: frozen and fully identified, but redistribution authorization has not been
  recorded.
- `grc-adapter-seed13`, `grc-adapter-seed42`, and `grc-adapter-seed73`: frozen by SHA-256, but
  redistribution authorization has not been recorded. Their retained shared-storage copies were
  verified as 62,332,992-byte weight files plus 1,274-byte configs; those files are not copied into
  this checkout.

This means the CPU smoke test and the mock/API contract are publicly runnable, while real-model
execution still requires an authorized copy of the derived Selector and Generator assets.

## Inspect, download, and verify

List every registered asset:

```bash
evidence-rag-artifacts list
```

Download one public file. Multi-file models require `--file-name` for each shard:

```bash
evidence-rag-artifacts download \
  --asset granite-embedding-r2 \
  --output model.safetensors
```

Verify an existing local file without downloading it:

```bash
evidence-rag-artifacts verify \
  --asset selector-seed13 \
  --file /path/to/authorized/model.safetensors
```

The downloader writes a temporary file in the destination directory, validates its byte count and
SHA-256, and only then publishes the final filename. It refuses to overwrite an existing file.
The same command is available as `python scripts/verify_artifacts.py ...` from a repository clone.

For a retained GR-C adapter, use the verifier against the authorized preserved copy before runtime
use. Do not infer safetensors identity from its LoRA parameter count or filename.

## Dataset boundary

Experiment 04 uses fixed ordered subsets of HotpotQA distractor validation, MuSiQue Full dev, and
RGB English noise. Their upstream source file sizes, hashes, and immutable mirror revisions are in
the manifest. HotpotQA is CC-BY-SA-4.0, MuSiQue is CC-BY-4.0, and RGB is non-commercial
CC-BY-NC-SA-4.0. Source files are not mirrored here.

Experiment 05 uses KILT-NQ, KILT-TriviaQA, ALCE-ASQA, and KILT/DPR Wikipedia corpora. The KILT and
ALCE repositories use MIT for their software, but that does not replace the terms of NQ, TriviaQA,
ASQA, or Wikipedia content. These raw sources and all derived indexes/bundles are therefore marked
`not-redistributed-mixed-terms`.

The checked-in `results/` directory contains only compact aggregate evidence needed to regenerate
the dissertation tables. Full raw recomputation requires the restricted bundles described in the
manifest and the immutable research archive ref.

## Front-end use

A front end should call the versioned HTTP API described in `docs/frontend-api.md`. It should not
download checkpoints into browser code, load safetensors directly, or depend on any individual HPC
account. Backend operators configure authorized model locations through the documented environment
variables.
