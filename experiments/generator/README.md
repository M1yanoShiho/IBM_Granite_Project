# Grounded GR-C Generator

The final Generator uses a Granite draft LoRA only for the draft call, a frozen Granite base
for claim splitting, and a frozen TRUE verifier for routing. The recipe, input hashes, and
three adapter hashes are frozen in `frozen_provenance.json`.

## Training and qualification sequence

The public scripts retain the tested stage boundaries. Each command accepts explicit input
and output paths; use its `--help` output for the complete required argument list.

1. Audit lengths and train the two preregistered seed-13 recipes:

   ```bash
   python scripts/full_flow_g300_draft_lora_train.py audit-lengths --help
   python scripts/full_flow_g300_draft_lora_train.py train \
     --recipe gr-c --run-kind formal --seed 13 \
     --model-snapshot "$GRANITE_SNAPSHOT" \
     --data-manifest "$GENERATOR_DATA_MANIFEST" \
     --train-cases "$GENERATOR_TRAIN_CASES" \
     --validation-cases "$GENERATOR_VALIDATION_CASES" \
     --ordered-ids "$GENERATOR_ORDERED_IDS" \
     --init-adapter "$GENERATOR_INIT_ADAPTER" \
     --output-dir "$GENERATOR_RUNS/grc-seed13"
   ```

2. Run and score the seed-13 GR-F/GR-C screen with
   `full_flow_g310_seed13_screen.py`. Generation is gold-free; references are opened only by
   the separate `score` command.
3. Apply the frozen G320 rule recorded in `frozen_provenance.json`. It selected GR-C before
   formal evaluation; it must not be rerun against held-out results to choose a new recipe.
4. Train GR-C seeds 42 and 73 with the same G300 command and frozen inputs. G330 records all
   three adapter/config/training-manifest hashes.
5. Run and score NIAH qualification with `full_flow_g400_niah_qualification.py` for seeds
   13/42/73.
6. Prepare, run, and score the 2Wiki cross-data qualification with
   `full_flow_g410_cross_data_qualification.py`.

The implementation closure is eight script modules: G300, G310, G400, G410,
`full_flow_b100.py`, `full_flow_g230.py`, `full_flow_joint.py`, and `alce_metrics.py`.
Earlier Full-flow stage scripts are preserved only in `research-archive-2026-08-25`.

## External inputs

Full training requires the Granite and TRUE snapshots, the initial adapter, the G223
controlled-continuation training bundle, and a GPU. These assets are not embedded in Git.
`frozen_provenance.json` pins their content identities and records that neither held-out nor
decision-dev data was read during training.
