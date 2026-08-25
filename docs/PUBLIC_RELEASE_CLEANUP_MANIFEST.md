# Public release cleanup inventory

**Status:** draft inventory; no deletion, move, commit, or push is authorized by this file.

**Target:** dissertation submission plus a reproducible public research-software release.

**Execution plan:** [`../task_plan.md`](../task_plan.md). This inventory decides what belongs in
the release; `task_plan.md` is the authority for Goal status, sequencing, acceptance, and execution.

**Release line:** trained three-module Evidence RAG system and the experiments needed to
support every result reported in the dissertation, including negative findings.

## 1. Decision labels

| Label | Meaning |
|---|---|
| `KEEP-CORE` | Required to install and run the final Retriever → Selector → Generator system. |
| `KEEP-REPRO` | Required to reproduce training, baselines, ablations, scoring, or final tables. |
| `KEEP-RESULT` | Small final report or aggregate result that belongs in the public repository. |
| `CONSOLIDATE` | Keep the information, but merge duplicate versions into one canonical file. |
| `EXTERNAL` | Store outside Git with a DOI/download URL, version, licence, and SHA-256. |
| `ARCHIVE` | Preserve in a supplementary research archive or historical tag, not in the default branch. |
| `REMOVE` | Generated/internal/redundant material that is not required in the release tree. |
| `HOLD` | Do not remove until a clean-clone dependency and reproduction test proves it unused. |

The release must preserve code needed to reproduce reported results, not every intermediate
attempt that led to that code. Failed or superseded versions are retained only when the
dissertation explicitly analyses them or when they are needed to reconstruct a final artifact.

## 2. Release acceptance gates

The cleanup is complete only when all of these are true:

- [ ] GitHub default branch points to the final release tree.
- [ ] A fresh clone can install the package using documented commands.
- [ ] CPU smoke tests run without model weights or HPC access.
- [ ] The trained three-module configuration can run when external model paths are supplied.
- [ ] Selector and Generator training commands are documented and reference public inputs.
- [ ] Experiment 04 and Experiment 05 aggregate tables can be regenerated from archived
      machine-readable inputs.
- [ ] Unit, integration, formatting, type, build, and smoke CI jobs are green.
- [ ] No personal account name, absolute HPC path, secret, raw cache, or private infrastructure
      detail exists in the release tree.
- [ ] No model weight, dataset cache, index, raw generation bundle, or per-query trace is tracked.
- [ ] Every external artifact has a stable URL/DOI, licence/provenance note, byte size, and SHA-256.
- [ ] README distinguishes technical execution success from scientific claim support.
- [ ] `LICENSE`, `CITATION.cff`, authorship, repository description, and release tag are present.

Target for the final checked-out tree: under 25 MiB excluding Git history and external release
assets, with no ordinary tracked file above 1 MiB unless it is a justified final figure/table.

## 3. Protect current uncommitted research first

These current worktree groups contain final research work and must not be lost during cleanup:

| Current path group | Decision | Release action |
|---|---|---|
| `configs/experiments/experiment04/` | `KEEP-REPRO` | Preserve the ten frozen system/ablation configs; later rename to a canonical final-evaluation location. |
| `src/evidence_rag/evaluation/experiment04_*.py` | `KEEP-REPRO` | Preserve final Experiment 04 runner/scoring logic. |
| `src/evidence_rag/evaluation/experiment05_*.py` | `KEEP-REPRO` | Preserve final Experiment 05 data, retrieval, generation, scoring, IO, and runtime logic. |
| `src/evidence_rag/evaluation/citation_metrics.py` | `KEEP-REPRO` | Shared final citation metric. |
| `src/evidence_rag/evaluation/sealed_runtime.py` | `KEEP-REPRO` | Frozen-output and hash-boundary checks. |
| `src/evidence_rag/evaluation/system_scorer.py` | `KEEP-REPRO` | Unified system scoring used by final experiments. |
| `src/evidence_rag/retriever/rerank.py` | `KEEP-REPRO` | Granite rerank baseline used in final comparisons. |
| `src/evidence_rag/selector/provence.py` | `KEEP-REPRO` | Provence baseline used in final comparisons. |
| `src/evidence_rag/selector/threshold_only.py` | `KEEP-REPRO` | Final threshold-only ablation used in Experiment 05. |
| `scripts/experiment04_*.py` | `KEEP-REPRO` | Preserve Goals 1–5; consolidate into `experiments/experiment04/` after tests pass. |
| `scripts/experiment05_*.py` | `KEEP-REPRO` | Preserve final data, generation, import, score, freeze, and compile stages. |
| `scripts/run_experiment04_*.slurm` | `CONSOLIDATE` | Use as source for one parameterised Experiment 04 HPC template; remove embedded account paths. |
| `scripts/run_experiment05_*.slurm` | `CONSOLIDATE` | Use as source for one parameterised Experiment 05 HPC template; remove embedded account paths. |
| corresponding untracked tests | `KEEP-REPRO` | Preserve tests for all final experiment and baseline code. |
| Experiment 04/05 final docs and aggregate results | mixed | Apply the explicit document/result rules in section 9 below. |

Before any cleanup deletion, this work must be saved in a dedicated, reviewable commit or an
external immutable snapshot. Do not stage all worktree files together.

## 4. Repository root

| Path | Decision | Required final state |
|---|---|---|
| `README.md` | `CONSOLIDATE` | Rewrite as the single public landing page: contribution, architecture, installation, quickstart, external assets, reproduction, results, limitations, citation. |
| `pyproject.toml` | `KEEP-CORE` | Keep; add README, authors, licence, project URLs, classifiers, and final optional dependency groups. |
| `requirements-dev.lock` | `KEEP-CORE` | Keep a reproducible CI/development lock; document how it was generated. |
| `.python-version` | `KEEP-CORE` | Keep while Python 3.11 is the tested runtime. |
| `.env.example` | `KEEP-CORE` | Keep placeholders only; no account, host, or private path. |
| `.gitignore` | `KEEP-CORE` | Keep model/data/cache/output exclusions and add any newly externalised result paths. |
| `.gitattributes` | `KEEP-CORE` | Keep after checking that it contains only intentional text/binary rules. |
| `.github/workflows/ci.yml` | `KEEP-CORE` | Keep and make green; separate lightweight core CI from optional GPU/HPC reproduction checks. |
| `LICENSE` | add | Licence choice requires agreement from the team/supervisor and any applicable IP owner. |
| `CITATION.cff` | add | Cite the software release and, when available, the dissertation/paper DOI. |
| `CONTRIBUTING.md` | add or omit | Recommended if continued collaboration is expected; not mandatory for submission. |
| `CODEOWNERS` | add or omit | Recommended to make module ownership explicit. |
| `CHANGELOG.md` | add | Record the dissertation release and scientific/result boundaries. |

## 5. Runtime source code

### 5.1 Keep as the final runnable system

| Path | Decision | Reason |
|---|---|---|
| `src/evidence_rag/contracts/` | `KEEP-CORE` | Shared module schemas, protocols, and validation. |
| `src/evidence_rag/pipeline/` | `KEEP-CORE` | Enforces Retriever → Selector → Generator data flow. |
| `src/evidence_rag/composition.py` | `KEEP-CORE` | Final module registry and portable config construction. Split factories later only if behaviour remains identical. |
| `src/evidence_rag/query_analysis.py` | `KEEP-CORE` | Query checklist used by the pipeline and final experiments. |
| `src/evidence_rag/infrastructure/config.py` | `KEEP-CORE` | Portable experiment configuration. |
| `src/evidence_rag/infrastructure/corpus.py` | `KEEP-CORE` | Corpus snapshots and pre-chunked runtime support. |
| `src/evidence_rag/infrastructure/datasets.py` | `KEEP-REPRO` | Dataset manifests used by final evaluation/training. |
| `src/evidence_rag/infrastructure/artifacts.py` | `HOLD` | Keep until final experiment import graph is validated. |
| `src/evidence_rag/infrastructure/benchmarks.py` | `HOLD` | Keep only if final public data download/materialisation uses it. |

### 5.2 Retriever

| Path | Decision | Reason |
|---|---|---|
| `retriever/bm25.py` | `KEEP-REPRO` | BM25 baseline and dependency of Strong BM25. |
| `retriever/strong_bm25.py` | `KEEP-CORE` | Sparse arm of the final Hybrid Retriever. |
| `retriever/granite.py` | `KEEP-CORE` | Granite dense retrieval; also contains historical wrappers currently imported by composition. |
| `retriever/hybrid.py` | `KEEP-CORE` | Final sparse+dense fusion. |
| `retriever/fusion.py` | `KEEP-CORE` | RRF implementation used by final Hybrid. |
| `retriever/chunking.py` | `KEEP-CORE` | Runtime corpus chunking contract. |
| `retriever/indexing.py` | `KEEP-CORE` | Portable index manifest and snapshot loading. |
| `retriever/rerank.py` | `KEEP-REPRO` | Final Granite reranker comparison arm. |

`HyDE`, Query2Doc, decompose, convex-fusion and other earlier Retriever variants should not be
presented as active final methods. Because several currently share `retriever/granite.py` and
`composition.py`, mark them `CONSOLIDATE`: either move them to `experiments/baselines/` or keep them
behind clearly labelled legacy baseline names. Remove their large config sweep only after tests and
the dissertation references confirm they are not used by final tables.

### 5.3 Selector

| Path | Decision | Reason |
|---|---|---|
| `selector/nli_runtime.py` | `KEEP-CORE` | Final trained NLI Selector runtime. |
| `selector/nli_dual_head.py` | `KEEP-CORE` | Model architecture and checkpoint loader. |
| `selector/dual_head.py` | `KEEP-CORE` | Checkpoint schema/model support. |
| `selector/risk_controlled.py` | `KEEP-CORE` | Frozen conservative selection policy. |
| `selector/models.py` | `KEEP-CORE` | Decision and trace models. |
| `selector/top_k.py` | `KEEP-REPRO` | Required keep-all/TopK ablation, not the final trained Selector. |
| `selector/threshold_only.py` | `KEEP-REPRO` | Experiment 05 ablation. |
| `selector/provence.py` | `KEEP-REPRO` | Final baseline. |
| `selector/answer_norm.py` | `HOLD` | Keep while Selector training/evaluation imports are validated. |
| `selector/guidance.py` | `HOLD` | Keep only if final public output/API exposes guidance. |

Earlier Corroboration, Beam, Graph, and adaptive-selector versions should be `ARCHIVE`, unless a
specific dissertation table uses them. Their final scientific findings may be summarized in the
paper/results document without retaining every implementation version in the release tree.

### 5.4 Generator

Keep the active grounded GR-C path and the Direct Granite comparison:

- `generator/granite.py` — `KEEP-CORE`
- `generator/nli.py` — `KEEP-CORE`
- `generator/verify_annotate.py` — `KEEP-CORE`
- `generator/draft.py` — `KEEP-CORE`
- `generator/claim_splitter.py` — `KEEP-CORE`
- `generator/entity_check.py` — `KEEP-CORE`
- `generator/key_facts.py` — `KEEP-CORE`
- `generator/json_parsing.py` — `KEEP-CORE`
- `generator/models.py` — `KEEP-CORE`
- `generator/trace.py` — `KEEP-CORE`
- `generator/extractive.py` — `KEEP-REPRO` for lightweight smoke/baseline use

The retired generate → verify → repair family is not on the final active path:

- `generator/attribution.py`
- `generator/completeness.py`
- `generator/evidence_recheck.py`
- `generator/repair.py`
- `generator/verified.py`
- `generator/verifier.py`

Mark these `ARCHIVE` unless the dissertation explicitly reports the verify-only/delete baseline.
If it does, keep the minimal runnable baseline and its tests under `experiments/baselines/`, not mixed
into the primary Generator documentation.

### 5.5 CLI, data preparation, and older research layers

Keep or replace with a documented final command:

- `cli/smoke.py` — `KEEP-CORE`
- `cli/experiment.py` — `KEEP-CORE`
- `cli/ingest.py` — `KEEP-CORE` only if public users can run the system on their own documents
- `cli/run_selector_lean.py` — `KEEP-REPRO` for the final Selector training/selection procedure
- the minimal label/component materialisation CLIs imported by `run_selector_lean.py` — `KEEP-REPRO`

The remaining old `cli/` commands, `materializer/`, `relations/`, and early evaluation state machines
are `HOLD` pending a clean dependency audit. Default decision after that audit:

- keep only code required to rebuild final Selector training inputs from public datasets;
- move Graph 1/Graph 2, Beam, R001–R005 and superseded state-machine code to `ARCHIVE`;
- keep `loaders/` only if document ingestion remains a claimed project capability;
- never remove a data-preparation step while the public training command still assumes its output.

## 6. Training and evaluation reproduction

### Selector

Keep the final lean training chain:

- `configs/selector/lean_v3.toml`
- `src/evidence_rag/cli/run_selector_lean.py`
- `src/evidence_rag/selector/{dual_head,nli_dual_head,nli_runtime,risk_controlled,models}.py`
- required selector materialisation/evaluation helpers and focused tests
- a new concise `docs/models/selector.md` containing data provenance, base revision, seeds,
  threshold/cap selection, checkpoint checksum, supported finding, and limitations

Old `adaptive_risk_*` and `beam_v1.toml` configs are `ARCHIVE` after final references are checked.

### Generator

Keep the final GR-C training and qualification chain:

- `scripts/full_flow_g300_draft_lora_train.py`
- `scripts/full_flow_g310_seed13_screen.py`
- `scripts/full_flow_g400_niah_qualification.py`
- `scripts/full_flow_g410_cross_data_qualification.py`
- focused tests for those commands
- one canonical frozen recipe derived from the G320/G330 reports
- a new concise `docs/models/generator.md` containing base revision, LoRA recipe, seeds,
  adapter checksums, qualification outcome, final held-out result, and limitations

Earlier G2xx data-repair scripts are `ARCHIVE` once the final training dataset recipe can be rebuilt
from a single deterministic public preparation command. Until then they remain `HOLD`; deleting them
would make the final adapter's training data provenance incomplete.

### Final evaluations

Keep all source and focused tests needed for:

- Experiment 04: three datasets, three Generator seeds, system baselines, and module ablations;
- Experiment 05: NQ/TriviaQA/ASQA, five systems, three module ablations, unified scoring, bootstrap,
  claim labels, and integrity audit;
- dedicated misleading-evidence Selector evaluation supporting the conditional module-level claim.

Consolidate the many stage scripts into documented commands such as:

```text
python -m evidence_rag.reproduction.train_selector ...
python -m evidence_rag.reproduction.train_generator ...
python -m evidence_rag.reproduction.evaluate_experiment04 ...
python -m evidence_rag.reproduction.evaluate_experiment05 ...
python -m evidence_rag.reproduction.build_tables ...
```

This consolidation must wrap or migrate tested logic; it must not silently reimplement the frozen
scorers after results are known.

## 7. Configurations

### Keep

- `configs/models/three_module_seed13.json`
- `configs/experiments/systemf_three_module_smoke_seed13.toml`
- `configs/selector/lean_v3.toml`
- `configs/experiments/experiment04/*.toml`
- one lightweight CPU reference/smoke config
- ingestion config only if ingestion is part of the final project scope

### Consolidate

- Create one all-seed final model manifest covering Selector seed 13 and Generator seeds 13/42/73.
- Repoint Experiment 04/05 configs from documentation artifacts to the canonical model manifest.
- Replace device-specific values with CLI/environment overrides.
- Replace all absolute storage paths with named environment variables.

### Archive/remove from the default branch

- `configs/experiments/chunk_*.toml`
- `configs/experiments/ovl_*.toml`
- `configs/experiments/pipe_*.toml`
- `configs/experiments/pipe8_*.toml`
- most `configs/experiments/retr_*.toml` sweep variants
- `configs/selector/adaptive_risk_*.toml`
- `configs/selector/beam_v1.toml`

Keep a sweep config only when a dissertation result directly names that arm and the final summary
cannot be regenerated from a smaller canonical matrix.

## 8. Scripts and HPC launchers

### Keep as research reproduction code

- all `scripts/experiment04_*.py`
- all `scripts/experiment05_*.py`
- the four final Generator training/qualification scripts listed in section 6
- final Selector training through the package CLI
- only the data download/normalisation scripts needed for public datasets
- table compilation, score aggregation, and integrity-audit commands used by final reports

### Consolidate

- Replace the current Experiment 04 Slurm files with one parameterised launcher plus an example
  environment file.
- Replace the current Experiment 05 Slurm files with one parameterised launcher plus stage flags.
- Move portable orchestration under `scripts/hpc/` or `experiments/*/hpc/`.
- Use `$USER`, environment variables, and CLI arguments; never commit account-specific paths.

### Archive/remove from the default branch

- superseded `full_flow_a*`, `b*`, `f*`, `g*`, and `s*` stage scripts not needed by final training;
- repeated repair candidates after the final deterministic preparation command exists;
- old `run_*`, `submit_*`, recovery, monitoring, and server-copy wrappers;
- one-off forensics, packet, and manual-review scripts not referenced by the dissertation;
- duplicate metrics scripts replaced by the final Experiment 04/05 scorer.

Each deletion candidate must pass this test: no kept module, kept test, final command, final table,
model card, or dissertation reproduction map references it.

## 9. Documents and results

### Canonical public documents to keep or create

- rewritten root `README.md`
- `docs/architecture.md`
- `docs/setup.md`
- `docs/reproduction.md`
- `docs/models-and-data.md`
- `docs/results.md`
- `docs/limitations.md`
- concise Selector and Generator model cards
- `docs/three-module-runtime-handoff.md`, merged into setup/API documentation where duplicated
- a machine-readable artifact manifest with public links and checksums

Use English as the canonical public documentation language. A Chinese overview may be included as a
separate translation, but mixed-language trackers should not be the primary entry point.

### Experiment 04

`KEEP-RESULT`:

- `FINAL_REPORT.md`
- `results/FINAL_TABLES.md`
- `results/summary_metrics.csv`
- `results/final_results.json`
- `results/final_audit.json`
- `results/table1.json`, `results/table2.json`
- `results/TABLE1.tex`, `results/TABLE2.tex`
- concise Goal 3 and Goal 4 result reports when needed for audit traceability

`EXTERNAL`:

- `per_query_metrics.csv`
- `per_query_goal4_metrics.csv`
- full generation, scoring, and trace bundles

`ARCHIVE` or `REMOVE` from default:

- `PLAN.md`, `TRACKER.md`, old handoffs, and all `snapshots/`
- readiness reports duplicated by the final audit
- intermediate goal manifests after their essential hashes are consolidated

The current `goal2_model_config_manifest.json` is `CONSOLIDATE`: move its still-needed all-seed model
metadata into `configs/models/` before removing the documentation artifact.

### Experiment 05

`KEEP-RESULT`:

- `findings.md`
- `FINAL_RESULTS_TABLES.md`
- `results/final/FINAL_REPORT.md`
- `EXPERIMENT_AUDIT.md` and `EXPERIMENT_AUDIT.json`
- `results/final/table1.csv`, `table1.json`, `table2.csv`, `table2.json`
- `results/final/claim_labels.json`
- `results/final/bootstrap.json`
- `results/final/final_audit.json`
- `results/final/tables.tex`
- `results/final/appendix_diagnostics.json` when cited by the dissertation

`EXTERNAL`:

- 12,000 raw generations
- 12,000 per-query scoring rows and traces
- frozen input/output bundles and import staging packages

`ARCHIVE` or `REMOVE` from default:

- `PLAN.md`, `PLAN_SELF_REVIEW.md`, `TRACKER.md`
- recovery/job chronology and intermediate readiness material duplicated by the final audit

### Older documents

Default action for these locations:

| Path | Decision |
|---|---|
| `docs/full-flow/experiments/01_*` | `ARCHIVE`; retain only findings explicitly used in the dissertation. |
| `docs/full-flow/experiments/02_*` | `ARCHIVE`; retain only provenance needed by the final Generator recipe. |
| `docs/full-flow/experiments/03_*` | `CONSOLIDATE`; distil G300/G320/G330/G400/G410 evidence into the Generator model card. |
| `docs/selector/experiments/01_*`–`07_*` | `ARCHIVE`; retain only reported findings and final provenance. |
| `docs/selector/experiments/08_*` | `CONSOLIDATE`; keep L002/L003 results and final policy, remove superseded plans/snapshots. |
| `docs/superpowers/` | `REMOVE` from the public release tree. |
| `.aris/` | `REMOVE` from tracking. |
| `refine-logs/` | `REMOVE` from the public release tree. |
| `docs/hpc-run-log.md` | `CONSOLIDATE`; replace with portable reproduction instructions and a compact provenance table. |
| `docs/presentations/*.{pptx,docx}` | `EXTERNAL` supplementary material unless required by the submission. |

### Existing top-level results and runs

- `results/` — `EXTERNAL` by default. Retain only small aggregates that are cited and not already
  represented by the canonical Experiment 04/05 result files.
- `runs/` — `EXTERNAL`; no runtime cache or source-parent dump belongs in Git.
- local `artifacts/` — `EXTERNAL`; never add the ignored  data/cache tree to Git.
- local `dist/` — `REMOVE`; rebuild from the release tag.

## 10. External model and data inventory

| Asset | Approximate size | Decision |
|---|---:|---|
| trained Selector seed-13 checkpoint | 704 MiB | `EXTERNAL` |
| trained GR-C seed-13 adapter | 60 MiB | `EXTERNAL` |
| GR-C seed-42 and seed-73 adapters | verify | `EXTERNAL` |
| public base-model snapshots and Hugging Face cache | many GiB | reference model IDs/revisions only |
| datasets and source caches | large | reference source/version/licence; provide download/preparation code |
| retrieval indexes | large | rebuild or publish separately with corpus/config checksums |
| raw generations, traces, sidecars, per-query metrics | large | publish as a versioned supplementary dataset if redistribution is allowed |

Preferred destinations, after licence and ownership review: Hugging Face for model artifacts and
data.bris/Zenodo for a DOI-bearing immutable research archive. The Git repository stores only
manifests and download instructions.

## 11. Tests

### Keep

- tests for every retained Retriever, Selector, Generator, contract, and pipeline module;
- the three-module smoke fixture and trained-runtime mocked integration tests;
- focused Selector and Generator training tests;
- Experiment 04/05 scorer, runtime, audit, and table-compilation tests;
- architecture tests preventing cross-module contract violations;
- packaging/typecheck tests.

### Remove or rewrite

- tests whose only purpose is to assert the textual state of old `PLAN`, `TRACKER`, or archival docs;
- artifact-presence tests for files moved to an external DOI archive;
- tests for code versions removed from the release tree.

Do not delete a failing test merely to make CI green. The current G000 document-state failure should
be resolved by separating historical-document validation from public core CI, or by archiving the
entire obsolete G000 route consistently.

## 12. Goal execution cross-reference

Detailed status and acceptance criteria live in [`../task_plan.md`](../task_plan.md). This table maps
the inventory to the executable plan.

| Goal | Scope from this inventory | Transition gate |
|---|---|---|
| G1 | Protect the uncommitted Experiment 04/05 groups in section 3 and freeze the full history. | Archive branch/tag resolve to the verified complete commit. |
| G2 | Apply sections 4–11 to construct the clean release tree without changing frozen science. | Release tree meets size, path, artifact, and recoverability gates. |
| G3 | Finalise the `KEEP-CORE` runtime in section 5 and canonical configs in section 7. | CPU and real-runtime interfaces are stable and tested. |
| G4 | Finalise `KEEP-REPRO` training/evaluation code in sections 6, 8, 9, and 11. | Every dissertation result maps to reproducible code and artifacts. |
| G5 | Publish or document all `EXTERNAL` assets in section 10. | Licence, link/DOI, version, size, and SHA-256 are complete. |
| G6 | Produce the canonical public documents and metadata in sections 4, 9, and 13. | README, licence, citation, results, and limitations are complete. |
| G7 | Execute all release acceptance gates in section 2. | Fresh clone, CI, CPU/HPC smoke, path/secret, and result checks pass. |
| G8 | Integrate the verified release into `main` and publish the dissertation tag/release. | Main, tag, Release, DOI links, and archive recovery are verified. |

## 13. Required claim language in the public release

The final README/results page must state all of the following together:

- the three-module implementation and frozen experiments completed with traceable technical validity;
- the trained Selector showed a conditional evidence-level benefit under misleading-evidence stress;
- the Selector did not show a supported end-answer uplift in its dedicated blind answer gate;
- on ordinary Experiment 05 datasets the frozen Selector was largely inactive;
- the stronger whole-system superiority claims were not supported;
- `FINAL PASS` refers to protocol/execution completion, not scientific superiority.

This prevents the cleaned repository from becoming cleaner at the cost of omitting valid negative
results or overstating the contribution.
