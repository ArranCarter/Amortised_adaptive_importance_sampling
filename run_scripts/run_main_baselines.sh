#!/bin/bash
# Evaluate the baselines on every main-table task. Tag exp:M.
# 64 jobs. 
set -e
cd "$(dirname "$0")/.."

while read -r params; do
  .venv/bin/python run.py device=auto threads=1 wandb.project=${AAIS_WANDB_MAIN:-aais-main} wandb.mode=online wandb.rollout=true $params </dev/null
done < run_params/run_params_main_baselines.txt
