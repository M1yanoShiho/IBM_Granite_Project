#!/bin/bash
# R013 / R014 / R015 -- the M0 §3.8 relation-model training path, ONE SEED PER JOB.
# Protocol: docs/selector/M0_PROTOCOL_FREEZE.md §3.8, base set by A3 (§11.5), seeds by §5.4.
#
# Three jobs, not one loop inside one job. §5.4's three-seed clause is reinstated for this path
# and each seed must leave its own manifest, log and checkpoints -- a run that quietly produced
# two of three would otherwise be reported as "the three seeds" with nothing on disk to say
# otherwise.
#
# ============================ BEFORE THE FIRST SUBMISSION ============================
#
# 0. §11.9(A)'s six hard prerequisites must be green. They were on bp1 2026-08-06; re-run if the
#    venv or the cache has moved. `任一不过即停` is the protocol's wording, not a suggestion:
#      export PYTHONPATH=src && python scripts/a3_preflight.py
#
# 1. LOGIN NODE, network on -- compute nodes run HF-offline and will die on from_pretrained:
#      export HF_HOME=/user/work/$USER/hf_cache
#      hf download cross-encoder/nli-deberta-v3-base
#      # ONLY if §3.8's pre-registered emergency arm is triggered (all three seeds finished,
#      # gold_supports_recall still under .85, both arms reported side by side):
#      # hf download cross-encoder/nli-deberta-v3-large
#
# 2. LOGIN NODE -- export the decontaminated VitaminC splits. ALL THREE outputs, including dev:
#    the trainer cross-checks the pairs files against the removal log and refuses a pair of
#    files that did not come out of the same decontamination pass. That guard is what catches a
#    path slip onto vitaminc_test.jsonl, which has an identical schema and would silently train
#    on the split §3.8 permits exactly one run on.
#      export PYTHONPATH=src
#      python -m evidence_rag.cli.export_vitaminc \
#        --out-train data/gate0b/vitaminc_train.jsonl \
#        --out-dev   data/gate0b/vitaminc_dev.jsonl \
#        --out-test  data/gate0b/vitaminc_test.jsonl \
#        --removed-log data/gate0b/vitaminc_decontamination.json
#
#    Dev is exported but NEVER trained on. It is loaded only so the trainer can verify that
#    train and dev share no page; §2.4 and §11.2a reserve VitaminC official dev as the one legal
#    surface for freezing a threshold and for a base-selection measurement.
#
# 3. LOGIN NODE -- two cheap checks that fail before a GPU hour is spent:
#      # (a) the whole data path, folds and manifest, without importing torch:
#      python -m evidence_rag.cli.train_relations \
#        --vitaminc-train data/gate0b/vitaminc_train.jsonl \
#        --vitaminc-dev   data/gate0b/vitaminc_dev.jsonl \
#        --decontamination-log data/gate0b/vitaminc_decontamination.json \
#        --seed 13 --output-dir runs/r013/plan --plan-only
#
#      # (b) does the CrossEncoder scaffold KEEP the named three-class head §11.9 item 4
#      #     measured? Item 6 verified only that the scaffold ACCEPTS the checkpoint, and
#      #     `num_labels=3` reaches PretrainedConfig, which can replace a named id2label with
#      #     LABEL_0/1/2. A head that came back re-ordered rather than anonymised would relabel
#      #     every edge and raise nothing, so this is checked, not assumed:
#      python -m evidence_rag.cli.train_relations \
#        --vitaminc-train data/gate0b/vitaminc_train.jsonl \
#        --vitaminc-dev   data/gate0b/vitaminc_dev.jsonl \
#        --decontamination-log data/gate0b/vitaminc_decontamination.json \
#        --seed 13 --output-dir runs/r013/scaffold --scaffold-check-only
#
# 4. A SMOKE JOB before the real one. --max-examples truncates the chain and the manifest says
#    is_smoke_run: true, so a smoke run cannot later be read as the real one. Once sealed-600
#    exists, smoke the NIAH half too -- it is the half with the blocking constraint:
#      sbatch scripts/run_r013_train_relations.slurm 13 runs/r013/smoke "--max-examples 2000"
#      sbatch scripts/run_r013_train_relations.slurm 13 runs/r013/smoke-niah \
#        "--max-examples 2000 --niah-manifest runs/niah-train-injected/manifest.json \
#         --niah-provenance runs/niah-train-injected/provenance.jsonl \
#         --niah-parents runs/niah-train-injected/source_parent.jsonl \
#         --niah-dev-manifest runs/niah-injected/manifest.json \
#         --sealed-dir runs/niah-sealed600 --niah-twin-label REFUTES"
#
#    A4 (M0 §12) APPROVED 2026-08-09, protocol now g2-proto-5. Background: the adaptation pool
#    shares 202 of 2000 queries with the dev evaluation run and no split was ever actually
#    made. The adaptation half therefore REQUIRES --niah-dev-manifest; dev-overlapping
#    families are excluded, the excluded count lands in the manifest, and the guard compares
#    query-id sets, never split labels. The five-flag form is refused outright.
#
#    §3.8(b) freezes the optimisation recipe -- AdamW, lr 2e-5, linear decay, warmup ratio .06,
#    batch 32, 2 epochs, max_length 256, weight decay .01, max_grad_norm 1.0, bf16 -- and records
#    that no legal tuning surface exists for it (dev is reserved by §2.4 and §11.2a; this path
#    trains on train only). The CLI therefore exposes NO flag for any of them and writes all ten
#    into every manifest. bf16 needs sentence-transformers >= 4; on a 3.x the job stops rather
#    than substituting fp16 autocast, which would run a different recipe with every log line and
#    every metric still in range.
#
# ============================== THE NIAH HALF IS BLOCKED ==============================
#
# §3.8 trains on VitaminC PLUS NIAH train-split mutation-log pairs for domain adaptation, under
# one hard constraint: those pairs' parent pages must have ZERO overlap with sealed-600.
# sealed-600 does not exist yet (§8 item 2). The half is implemented and REFUSES to run without
# a sealed-600 parent-page set that the pairs actually clear. There is no flag that disables the
# check. Omitting the NIAH flags runs the VitaminC half alone and the manifest says so in words.
#
# Once sealed-600 exists, add -- all six together, the CLI refuses a partial set:
#   --niah-manifest     runs/niah-train-injected/manifest.json
#   --niah-provenance   runs/niah-train-injected/provenance.jsonl
#   --niah-parents      runs/niah-train-injected/source_parent.jsonl
#   --niah-dev-manifest runs/niah-injected/manifest.json
#   --sealed-dir        runs/niah-sealed600
#   --niah-twin-label   REFUTES
#
# --niah-dev-manifest names the dev evaluation run the adaptation set must stay disjoint from
# (ruling 2026-08-09 / A4). The guard compares query-id SETS and never reads a manifest's
# `split` label -- on 2026-08-09 all three NIAH runs' labels read 'dev', including
# runs/niah-train*; the label carries no meaning, the sets do.
#
# --sealed-dir is the corpus line's BUILD DIRECTORY, not a file: the check reads that build's
# own frozen split fingerprint (`sealed600.read_split_fingerprint(dir).parent_pages`), so the
# set that was audited and the set that is checked are the same object. Both sides normalise
# through `source_parent.normalize_parent`; `ParentIndex.parent_of` is deliberately NOT used
# here, because its self-parent fallback would guarantee zero overlap on a leakage comparison.
#
# --niah-parents is built by `python -m evidence_rag.cli.build_source_parent`.
#
# --niah-twin-label is REFUTES, ruled by §3.8(a): those pairs are genuine contradictions, the
# consumers' collapse still counts REFUTES as not-supported so neither 0B-2 metric moves, and
# UNKNOWN is then trained only by VitaminC's NEI class, which is where 0B-1 needs it. The flag
# has NO default even so -- pass it explicitly, so the ruling is legible in every manifest.
#
# ================================= COST, READ IT ==================================
#
# 5 folds x 3 seeds = FIFTEEN full fine-tunes of a 184M-parameter model over ~370k VitaminC
# rows. That is the pre-registered recipe, not a knob. Budget accordingly and run the smoke job
# first; the walltime below is for ONE seed's five folds.
#
# Submit from the project root. Positional args, in order:
#   SEED  OUTPUT_DIR  EXTRA_FLAGS
#   mkdir -p logs runs
#   sbatch scripts/run_r013_train_relations.slurm 13 runs/r013/seed-13   # R013
#   sbatch scripts/run_r013_train_relations.slurm 42 runs/r014/seed-42   # R014
#   sbatch scripts/run_r013_train_relations.slurm 73 runs/r015/seed-73   # R015
#
# AFTERWARDS. This job writes per-row out-of-fold predictions and a manifest. It does NOT
# compute any Gate 0B number: those predictions are over TRAINING data, and `external_report`
# would happily print the five frozen 0B-1 threshold names against the wrong split. 0B-1 is
# VitaminC official test, run ONCE (§3.8), by `scripts/run_gate0b.slurm`. To score a trained
# checkpoint there, paste the `gate0b_registry_line` from the manifest into
# `cli/gate0b.py::LABEL_ORDER` -- a reviewable one-line diff, which is what §11.10 item 7 means
# by "must go through the registry, not be hardcoded at the call site".
#SBATCH --job-name=r013-train
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
# a100 is pinned, not preferred: §3.8(b) freezes bf16, and this partition is mostly
# rtx_2080 (13 nodes) + V100 (3) which have no bf16 -- an unpinned gpu:1 usually lands
# there (18319801 drew bp1-gpu002, a 2080). Scheduling constraint, not a recipe change.
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
SEED="${1:-13}"
OUTPUT_DIR="${2:-runs/r013/seed-$SEED}"
EXTRA="${3:-}"

module load languages/python/3.12.3
export HF_HOME=/user/work/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IR_DATASETS_HOME=/user/work/$USER/ir_datasets PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Trainer integrations that phone home would fail under HF_HUB_OFFLINE and take the job with
# them; the CLI also sets report_to=[] on the training arguments.
export WANDB_DISABLED=true HF_HUB_DISABLE_TELEMETRY=1
source /user/work/$USER/venv/bin/activate
cd /user/work/$USER/IBM_Granite_Project
export PYTHONPATH=src

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| torch', torch.__version__)"

# torch < 2.6 is a project pin, and the whole of A3 rests on it (§11.1): under it transformers
# refuses to torch.load a .bin, which is why the pre-registered base could not be used. If this
# ever prints >= 2.6 the amendment's premise is false and it should be taken back to the
# protocol, not worked around here. scripts/a3_preflight.py is what checks it properly.

echo "=== R013-R015: §3.8 training, seed $SEED -> $OUTPUT_DIR ==="
# $EXTRA is deliberately unquoted so it disappears entirely when empty and word-splits into
# separate flags when set. No path here contains spaces.
python -m evidence_rag.cli.train_relations \
  --vitaminc-train data/gate0b/vitaminc_train.jsonl \
  --vitaminc-dev data/gate0b/vitaminc_dev.jsonl \
  --decontamination-log data/gate0b/vitaminc_decontamination.json \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR" \
  $EXTRA

echo "=== run manifest ==="
cat "$OUTPUT_DIR/manifest.json"
