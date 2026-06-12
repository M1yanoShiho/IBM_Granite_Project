# HPC Deployment — Granite on BluePebble (BP1)

How to run the Granite generative model (RAG layer + long-context experiments) on
Bristol's **BluePebble** HPC. The retrieval + IR-metrics pipeline is CPU-friendly
and does **not** need this; only the generation side does.

> **Golden rule: never run a model on the login node.** Jobs found on the login
> node are killed without notice. The login node is only for: transferring code,
> downloading weights, installing the environment, and submitting jobs. All model
> inference goes through `sbatch` / `srun` to a GPU node.

## Our confirmed settings

| Slurm field | Value | Source |
| --- | --- | --- |
| `--account` | `coms039904` | IT email (project code) |
| `--qos` | `normal` | `sacctmgr` |
| `--partition` | `gpu` (14-day) or `gpu_short` (6h, faster queue) | unrestricted association |
| username / host | `uz25020@bp1-login.acrc.bris.ac.uk` | IT email |

Interactive `srun --pty` is blocked on this cluster ("launch failed: Unspecified
error") — **always submit with `sbatch`**.

### GPUs available (for an 8B model)

| GPU | VRAM | Runs 8B? | `--gres` |
| --- | --- | --- | --- |
| A100 MIG `3g.40gb` | 40 GB | ✅ best (bf16 + some long context) | `gpu:3g.40gb:1` |
| RTX 3090 | 24 GB | ✅ bf16 fits | `gpu:rtx_3090:1` |
| V100 | 16/32 GB | ⚠️ 32G ok, 16G tight | `gpu:V100:1` |
| RTX 2080 Ti | 11 GB | ❌ bf16 too big; 4-bit only | `gpu:rtx_2080:1` |

A plain `--gres=gpu:1` on `gpu_short` gives an **RTX 2080 Ti (11 GB)** — fine for
the 3B smoke test, but for 8B you must request `gpu:rtx_3090:1` / `gpu:3g.40gb:1`
or use 4-bit quantisation. Node driver is **CUDA 12.4**, so install PyTorch
`cu121` (forward-compatible).

Restricted, group-named partitions (`mwvdk`, `mlcnu`, `rmc`, `econ-ssl`,
`geophysics`) are **not** ours — use **`gpu`** / **`gpu_short`**.

---

## One-time setup

### 1. Connect

- **Off campus:** install + connect the **F5 BIG-IP Edge Client** (server
  `uobnet2.bristol.ac.uk`). On campus (eduroam/wired): no VPN.
- SSH (no web login):
  ```bash
  ssh uz25020@bp1-login.acrc.bris.ac.uk     # accept the host key (yes); password is blind-typed
  ```

### 2. Get the code (login node has internet)

```bash
cd /user/work/$USER                          # large files live in /user/work, NOT /user/home (small quota)
git clone https://github.com/M1yanoShiho/IBM_Granite_Project.git
cd IBM_Granite_Project
mkdir -p logs                                # Slurm needs this dir for job output
```

### 3. Python env + **CUDA** PyTorch (the #1 gotcha)

There is no `python` on the login node by default — load a module first. Use
**3.12** (the default 3.14 is too new for prebuilt torch/transformers wheels).
`requirements.txt` does not pin torch, so install the **CUDA** build first; the
rest then see torch as already satisfied.

```bash
module load languages/python/3.12.3      # NOT the 3.14 default
python -m venv /user/work/$USER/venv
source /user/work/$USER/venv/bin/activate
pip install --upgrade pip

pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA torch FIRST
pip install -r requirements.txt                                         # rest; torch already satisfied
# For 4-bit 8B on an 11GB card later:  pip install bitsandbytes
```

The Slurm scripts (`scripts/*.slurm`) also `module load languages/python/3.12.3`
before activating the venv — compute nodes need it too.

### 4. Pre-download the model weights (on the login node)

Compute nodes have **no internet** — download on the login node into the cache on
`/user/work`, then jobs load it offline (`HF_HUB_OFFLINE=1`).

```bash
export HF_HOME=/user/work/$USER/hf_cache
# Start with the small model; confirm the exact 8B *instruct* repo on
# https://huggingface.co/ibm-granite before downloading it.
huggingface-cli download ibm-granite/granite-4.1-3b
# huggingface-cli download ibm-granite/<8B-INSTRUCT>
```

> **Model choice:** RAG answer generation needs an **instruct** model. The 8B
> **base** model is for the long-context "needle" experiments (~512K context).
> Granite is Apache-2.0 / open — usually no token needed; if a repo is gated, run
> `huggingface-cli login` first.

---

## Run

### Smoke test (do this first)

```bash
sbatch scripts/smoke_8b.slurm        # loads granite-4.1-3b on a 2080 Ti, generates one sentence
squeue -u $USER                      # watch it queue/run (PD -> R -> gone)
cat logs/granite-smoke-*.out         # check the output
```
Success here = `llm_client.py` works against a real model. Then bump
`GRANITE_MODEL_ID` to the 8B instruct repo and switch `--gres` to a 24GB+ card.

### Full RAG benchmark (later)

`scripts/run_rag.slurm` targets a 24GB RTX 3090. It only produces a table once
`eval/run_rag.py`, `eval/rag_metrics.py`, a gold-answer dataset loader, and the
`DenseRetriever` are implemented (see `docs/dev-log.md`).

---

## Gotchas checklist

1. **Never run on the login node** — always `sbatch` (interactive `srun --pty` is blocked here).
2. **Install CUDA torch** (`cu121`), don't let `requirements.lock`'s `+cpu` win.
3. **Download weights on the login node**; jobs run with `HF_HUB_OFFLINE=1`.
4. **`HF_HOME` on `/user/work`**, not `/user/home` (small quota).
5. **`mkdir -p logs`** before `sbatch` (Slurm won't create the output dir).
6. **Queue is shared / fair-use** — jobs don't start instantly; submit well before
   the 2026-09-04 deadline.

## Handy commands

```bash
squeue -u $USER                      # your jobs
scontrol show job <JOBID> | grep StartTime   # when will it start
scancel <JOBID>                      # cancel
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,ExitCode   # post-mortem
```
