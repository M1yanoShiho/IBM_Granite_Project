# Evidence RAG Models and Data

The repository intentionally contains no trained weights, upstream model snapshots, raw datasets,
indexes, or per-query generation bundles. [`ARTIFACT_MANIFEST.json`](../ARTIFACT_MANIFEST.json) is
the machine-readable authority for frozen revisions, byte sizes, SHA-256 digests, licenses, source
links, and availability.

## 📦 Public upstream models

The Retriever base models, Generator base model, Selector backbone, TRUE verifier, and MiniCheck
scorer remain on their official repositories at immutable revisions. The manifest registers every
required weight file and the exact size and digest observed at that revision.

The canonical runtime uses:

- IBM Granite English embedding R2 for dense retrieval;[^granite-embed]
- NLI DeBERTa v3 base as the Selector backbone;[^nli]
- IBM Granite 4.1 3B as the Generator base;[^granite-generator]
- TRUE T5-XXL NLI mixture for frozen answer verification.[^true]

Provence is link-only. Its upstream metadata, model card, and bundled license file use inconsistent
Creative Commons labels and additional terms. This project does not mirror it, grant commercial
rights, or resolve those conflicts for users.[^provence]

## 🔒 Restricted derived models

The final derived assets are frozen but not public downloads:

| Asset | Bytes | Availability |
|---|---:|---|
| Selector seed-13 checkpoint | 737,731,768 | `restricted-not-published` |
| GR-C seed-13 adapter weights | 62,332,992 | `restricted-not-published` |
| GR-C seed-42 adapter weights | 62,332,992 | `restricted-not-published` |
| GR-C seed-73 adapter weights | 62,332,992 | `restricted-not-published` |

Each GR-C adapter also has a 1,274-byte config. All four weight identities were rechecked against
preserved authorized copies. Redistribution authorization has not been recorded, so no public URL
is provided and no permission expansion is implied.

This boundary means the CPU smoke and API contract are publicly runnable, while real-model
execution requires an authorized copy of the trained Selector and Generator adapter.

## 🔍 Inspect, download, and verify

List every registered asset:

```bash
evidence-rag-artifacts list
```

Download one public file. Multi-file models require one command per registered shard:

```bash
evidence-rag-artifacts download \
  --asset granite-embedding-r2 \
  --output model.safetensors
```

Verify an existing authorized file without downloading it:

```bash
evidence-rag-artifacts verify \
  --asset selector-seed13 \
  --file /path/to/authorized/model.safetensors
```

The downloader writes a temporary file beside the destination, validates bytes and SHA-256, and
only then publishes the final filename. It refuses to overwrite an existing file. The same command
is available as `python scripts/verify_artifacts.py ...` from a repository clone.

## 📚 Dataset boundary

Experiment 04 uses fixed ordered subsets of HotpotQA distractor validation, MuSiQue Full dev, and
RGB English noise. HotpotQA is CC-BY-SA-4.0, MuSiQue is CC-BY-4.0, and RGB is non-commercial
CC-BY-NC-SA-4.0.[^hotpot][^musique][^rgb] Source files are not mirrored here.

Experiment 05 uses KILT-NQ, KILT-TriviaQA, ALCE-ASQA, and KILT/DPR Wikipedia corpora. The KILT and
ALCE repositories use MIT for their software, but this does not replace the terms of NQ,
TriviaQA, ASQA, or Wikipedia content.[^kilt][^alce] These raw sources and all derived indexes and
run bundles are therefore marked `not-redistributed-mixed-terms`.

The checked-in [`results/`](../results/) directory contains only compact aggregates needed to
regenerate the dissertation tables. Full raw recomputation requires the restricted bundles in the
manifest and the immutable research archive reference.

## 🌐 Front-end use

A front end calls the versioned HTTP API described in
[front-end integration](frontend-integration.md). It must not download checkpoints into browser
code, parse safetensors, or depend on an individual cluster account. Backend operators configure
authorized model locations through environment variables.

[^granite-embed]: [Granite embedding frozen revision](https://huggingface.co/ibm-granite/granite-embedding-english-r2/tree/47ea694b257b703fee9253d75c2b1f2985180498)
[^nli]: [NLI DeBERTa frozen revision](https://huggingface.co/cross-encoder/nli-deberta-v3-base/tree/6c749ce3425cd33b46d187e45b92bbf96ee12ec7)
[^granite-generator]: [Granite 4.1 3B frozen revision](https://huggingface.co/ibm-granite/granite-4.1-3b/tree/c0650403e44e78ec0262dab1c90914c65b196c4e)
[^true]: [TRUE verifier frozen revision](https://huggingface.co/google/t5_xxl_true_nli_mixture/tree/aa6cfe1dd4257853bfdd772992045f41bfc14988)
[^provence]: [Provence frozen upstream revision](https://huggingface.co/naver/provence-reranker-debertav3-v1/tree/ef49e233e3c6e50efc476c68f1390f8a63add4d4)
[^hotpot]: [HotpotQA official site](https://hotpotqa.github.io/)
[^musique]: [MuSiQue official repository](https://github.com/StonyBrookNLP/musique)
[^rgb]: [RGB official repository](https://github.com/chen700564/RGB)
[^kilt]: [KILT official repository](https://github.com/facebookresearch/KILT)
[^alce]: [ALCE official repository](https://github.com/princeton-nlp/ALCE)
