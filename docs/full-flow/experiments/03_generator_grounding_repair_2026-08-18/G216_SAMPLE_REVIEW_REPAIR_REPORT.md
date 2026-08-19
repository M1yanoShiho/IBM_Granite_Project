# G216 Sample Review Repair Report

**Status:** `PRE_MANUAL_PASS / NO TRAINING STARTED`

G216 applied the controlled repair authorized after G212M. It did not train Generator, did not generate utility labels, did not change TRUE thresholds, and did not read sealed, held-out, or official dev data.

## Repair Rule

G216 only removed cases that failed G212M sample review/adjudication. For failed 2Wiki train answerable cases, the corresponding unsupported train counterpart was removed as well to keep the train distribution controlled.

This is a narrow repair. It does not claim the data is frozen. The repaired bundle must still pass the next length/sample review stage before G300 can start.

Because G216 only removes rows and does not create or modify any target text, structural and TRUE support status are inherited as a strict subset of the already passed G214/G210R2 lineage. Any future repair that changes target text must rerun structural and TRUE inference instead of using this subset inheritance.

## Removed Rows

- G212M failed sample rows: 8
- Removed train cases: 6
- Removed validation cases: 5
- Total removed cases: 11

The 11 removed cases consist of:

- 3 failed 2Wiki train answerable cases;
- 3 matching 2Wiki unsupported train counterparts;
- 3 failed 2Wiki model-val answerable cases;
- 2 failed NIAH model-val answerable cases.

## Revised Counts

| Dataset / role | Groups |
|---|---:|
| NIAH train answerable | 515 |
| NIAH model-val answerable | 213 |
| 2Wiki train answerable | 824 |
| 2Wiki train unsupported | 1,049 |
| 2Wiki model-val answerable | 103 |

Unsupported update ratio is `0.11292927118096674`, within the 10%-15% gate. Split group overlap and component overlap are both `0`.

## Gate Result

All G216 gates passed:

- NIAH train groups >= 400;
- NIAH model-val groups >= 100;
- 2Wiki train groups >= 400;
- 2Wiki model-val groups >= 100;
- unsupported update ratio within 10%-15%;
- split group overlap = 0;
- split component overlap = 0.

## Runtime Artifacts

Full revised train/validation cases remain on the server:

- `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G216-v1/data/train_cases.jsonl`
- `/scratch/fl25387/IBM_Granite_Project_latest/runs/full-flow/G216-v1/data/validation_cases.jsonl`

Hashes:

- train cases SHA256: `a8d0cd236b912fefd42cc2f06e43ccc71ed31e6349a2d84642c76784ffb309d7`
- validation cases SHA256: `fdbb9b4003dcfa2984ed42b91cdab55db47bd2dae9e7c8ee89490672359f4c6d`
- ordered IDs SHA256: `99dab16a2b776a8466dee1edd5394c8a8ac99e3fef7e58bb3c3a966430fc6c9b`

## Git Artifacts

- Execution audit: `artifacts/G216/G216_EXECUTION_AUDIT.json`, SHA256 `72e61b14031d0fd8958722258fdc1ca06f7f1fcd76470beb039146329c2f8d70`
- Manifest: `artifacts/G216/data/manifest.json`, SHA256 `0782877816719e13f2caa1ce53fd38be38c2e2d18756df91ba9095b0a0301df0`
- Ordered IDs: `artifacts/G216/data/ordered_ids.json`, SHA256 `99dab16a2b776a8466dee1edd5394c8a8ac99e3fef7e58bb3c3a966430fc6c9b`

## Next Step

Run G212R2 on the G216 bundle:

1. rerun length/truncation audit with `max_length=2304`;
2. prepare a new fixed sample packet;
3. complete sample review/adjudication on the new packet;
4. only if freeze readiness passes, unlock G300.
