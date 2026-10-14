#!/bin/bash
# Extra bananahard_d5 checkpoints for the figures.
# 6 jobs. 
set -e
cd "$(dirname "$0")/.."

while read -r params; do
  .venv/bin/python train.py device=auto threads=1 wandb.project=${AAIS_WANDB_MAIN:-aais-main} wandb.mode=online wandb.figure_every=250 train.checkpoint_every=250 $params </dev/null
done < run_params/run_params_bananahard5_train.txt
