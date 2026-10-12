#!/bin/bash
# Ablation: the EM update features. Tag exp:AE.
# 24 jobs. 
set -e
cd "$(dirname "$0")/.."

while read -r params; do
  .venv/bin/python train.py device=auto threads=1 wandb.project=${AAIS_WANDB_ABLATIONS:-aais-ablations} wandb.mode=online wandb.figure_every=250 train.checkpoint_every=250 $params </dev/null
done < run_params/run_params_em_ablation.txt
