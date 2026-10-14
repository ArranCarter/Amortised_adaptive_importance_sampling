#!/bin/bash
# Shifted families for the capacity ablation. Tag exp:AHA.
# 48 jobs. 
set -e
cd "$(dirname "$0")/.."

while read -r params; do
  .venv/bin/python run.py device=auto threads=1 wandb.project=${AAIS_WANDB_ABLATIONS:-aais-ablations} wandb.mode=online $params </dev/null
done < run_params/run_params_capacity_shiftA.txt
