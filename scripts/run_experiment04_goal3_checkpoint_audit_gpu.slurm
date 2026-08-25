#!/usr/bin/env bash
#SBATCH --job-name=exp04-g3-checkpoint-audit-gpu
#SBATCH --account=coms039904
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/user/work/fl25387/experiment04_goal3_checkpoint_audit_gpu-%j.out

set -euo pipefail

audit_file() {
  local label="$1"
  local path="$2"
  if [[ -f "$path" ]]; then
    local digest
    digest="$(sha256sum "$path" | cut -d' ' -f1)"
    printf 'CHECK %s PRESENT %s\n' "$label" "$digest"
  else
    printf 'CHECK %s MISSING\n' "$label"
  fi
}

root=/scratch/fl25387/IBM_Granite_Project_latest
audit_file selector_seed13 "$root/runs/selector-lean-v3/L002/NLI-base/seed13/model.safetensors"
audit_file grc_seed13_weights "$root/runs/full-flow/G310-v1/formal-grc-seed13/adapter/adapter_model.safetensors"
audit_file grc_seed13_config "$root/runs/full-flow/G310-v1/formal-grc-seed13/adapter/adapter_config.json"
audit_file grc_seed42_weights "$root/runs/full-flow/G330-v1/formal-grc-seed42/adapter/adapter_model.safetensors"
audit_file grc_seed42_config "$root/runs/full-flow/G330-v1/formal-grc-seed42/adapter/adapter_config.json"
audit_file grc_seed73_weights "$root/runs/full-flow/G330-v1/formal-grc-seed73/adapter/adapter_model.safetensors"
audit_file grc_seed73_config "$root/runs/full-flow/G330-v1/formal-grc-seed73/adapter/adapter_config.json"
