# Experiment 04 Goal 2 — System Wiring Readiness

**Date:** 2026-08-22  
**Result:** `PASS (= READY)`  
**Boundary:** revealed synthetic development fixture only; no formal held-out run or score

## Decision

Goal 2 passes. The exact 10-arm matrix is frozen, every arm completes the shared
development smoke through one runner and one five-metric scorer, the four public
baseline routes have also passed a real-model GPU smoke, and the existing Ours
three-seed artifacts retain reload/qualification evidence. The formal output
directory remained empty throughout Goal 2.

Per the user's 2026-08-22 execution instruction, a Goal PASS is now an automatic
handoff checkpoint. Goal 3 is therefore the next active goal; this report is not
an authorization to change any frozen identity.

## Frozen arms

| Group | Arms | Development result |
|---|---|---|
| Baselines | Dense, Hybrid, Granite Rerank, Provence | 4/4 injected smoke PASS; 4/4 real-model GPU smoke PASS |
| Ours | GR-C seeds 13, 42, 73 | 3/3 unified smoke PASS; frozen adapters reload/qualification PASS |
| Ablations | Dense Retriever, Top-10, Direct Generator | 3/3 unified smoke PASS |
| **Total** | **10** | **10/10 complete one revealed synthetic query** |

The exact module identities are enforced by
`src/evidence_rag/evaluation/experiment04_runner.py` and the ten TOML files under
`configs/experiments/experiment04/`. A config that changes one frozen module,
seed, budget, prompt, or role fails validation before running.

## Model and protocol freeze

`artifacts/goal2_model_config_manifest.json` records the immutable Hugging Face
commit, config SHA-256, and weight SHA-256 for:

- Granite dense embedding r2;
- Granite embedding reranker r2;
- official Provence;
- Granite 4.1-3B base;
- Selector NLI backbone plus the project seed-13 checkpoint;
- TRUE runtime verifier;
- MiniCheck scorer-only judge;
- all three GR-C adapters.

The shared contract fixes Top10, Hybrid RRF k=60, reranker pool 40, input cap
2304, output cap 256, greedy decoding, no silent truncation, one context format,
one inline citation format, and one prompt hash. MiniCheck remains scorer-only and
TRUE remains part of the grounded runtime path.

## Verification evidence

### Unified 10-arm deterministic development smoke

Command:

```text
PYTHONPATH=src .venv/bin/python scripts/experiment04_goal2.py
```

Result: `PASS`, 10/10 arms. Each content-free record confirms a retrieval trace,
selected context, non-empty answer, all five score fields, scorer readability,
and zero forbidden gold-sidecar keys. Full synthetic traces are under the ignored
`runs/experiment04/goal2-development-smoke/` directory; the committed digest-only
record is `artifacts/goal2_smoke_manifest.json`.

### Real baseline GPU smoke

Slurm job `18671419` ran on `bp1-gpu037` (Tesla V100 32GB) with Python 3.11.15,
torch 2.6.0+cu126, and Transformers 4.57.6. It recomputed the frozen model hashes,
loaded the real dense embedder, Granite reranker, official Provence remote code,
Direct Granite, and MiniCheck, then completed the four baseline arms on the same
revealed synthetic case. Result: `COMPLETED`, exit `0:0`, elapsed `00:01:58`,
maximum RSS 9,124,104 KiB, and 4/4 arms `PASS`.

The content-free record is `artifacts/goal2_real_baseline_smoke.json`. Full traces
remain on the server under
`/user/work/fl25387/experiment04_goal2_outputs/real-baselines/full/` and are not
committed.

An earlier job, `18671328`, stopped before producing any arm output because the
official Provence code required undeclared `nltk`. Goal 2 added `nltk` and
`sentencepiece` to the Granite runtime dependencies, installed the NLTK sentence
tokenizer data, and reran the unchanged frozen configuration successfully.

### Ours model evidence

The real seed-13 three-module route already passed the server wiring smoke in
`reports/THREE_MODULE_WIRING_SMOKE_2026-08-21.md` under Experiment 03. The GR-C
seed 13/42/73 adapter files and configs match the hashes recorded by G330/G430;
all three training manifests are complete with reload `PASS`, and the frozen
three-seed family passed the locked G400/G410 responsibility qualifications.
Goal 2 did not retrain or select a seed.

### Automated tests

- Goal 1 + Goal 2 focused set: 23 tests passed before the real-server run.
- Retriever, Generator, runner, and three-module regression set: all tests passed.
- Full repository suite excluding the known unrelated stale Experiment 03 status
  assertion: all selected tests passed, with the existing optional skips only.
- Ruff and strict mypy passed on the new runner, reranker, Provence adapter, local
  smoke, and server smoke sources.

The excluded pre-existing assertion is
`test_g000_without_server_audit_stops_before_next_stage`; it expects Experiment
03 to remain active at G000 and is unrelated to Experiment 04 implementation.

## Leakage and held-out audit

- Only `tests/fixtures/three_module_smoke_dataset/` was used for Goal 2 outputs.
- Runtime outputs contain no gold answers, support labels, support units, or
  component IDs.
- Gold fixture fields exist only in an in-memory scorer sidecar and never enter
  the committed smoke manifest.
- No Retriever, Selector, Generator, TRUE, or MiniCheck call was made on any
  formal held-out record in Goal 2.
- `runs/experiment04/formal/` remained absent/empty.

## PASS checklist

| Requirement | Result |
|---|---|
| Exactly 10 frozen configurations | PASS |
| Correct system and seed identities | PASS |
| Shared budgets, prompt, format, and scorer | PASS |
| At least one development case per arm | PASS (10/10) |
| Retrieval, selection, answer, and score present | PASS (10/10) |
| Real public baseline models load and run | PASS (4/4) |
| Ours checkpoints have real reload/qualification evidence | PASS (3/3) |
| No gold leakage | PASS |
| Formal held-out output directory empty | PASS |

**Final Goal 2 result: `PASS (= READY)`.**
