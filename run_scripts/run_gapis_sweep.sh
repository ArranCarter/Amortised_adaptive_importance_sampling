#!/bin/bash
# GAPIS step-size sweep: 11 sizes on every task. Tag exp:GS.
# 176 jobs. 
set -e
cd "$(dirname "$0")/.."

while read -r params; do
  .venv/bin/python run.py device=auto threads=1 wandb.project=${AAIS_WANDB_MAIN:-aais-main} wandb.mode=online $params </dev/null
done < run_params/run_params_gapis_sweep.txt
