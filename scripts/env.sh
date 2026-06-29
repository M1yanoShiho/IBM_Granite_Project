#!/bin/bash
# BluePebble LOGIN-NODE setup. Source this once per login session before you
# prefetch / verify datasets or models, or submit jobs:
#
#     source scripts/env.sh
#
# Why: every fresh login session has no `python` until the module is loaded, and
# by default HuggingFace + ir_datasets download into /user/home (tiny quota) where
# the Slurm jobs — which read /user/work — can't find them. This points both caches
# at /user/work and activates the venv, so downloads land where jobs look.
#
# LOGIN NODE ONLY: caches stay ONLINE here so you can download. The Slurm scripts
# set HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE themselves for the offline compute
# nodes — do NOT set those here.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Run with:  source scripts/env.sh   (it must be sourced, not executed)" >&2
    exit 1
fi

module load languages/python/3.12.3
source "/user/work/$USER/venv/bin/activate"

export HF_HOME="/user/work/$USER/hf_cache"
export IR_DATASETS_HOME="/user/work/$USER/ir_datasets"

echo "env ready: python=$(command -v python)"
echo "  HF_HOME=$HF_HOME"
echo "  IR_DATASETS_HOME=$IR_DATASETS_HOME"
