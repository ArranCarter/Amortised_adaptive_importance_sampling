#!/bin/bash
# Sample figures for bananahard_d5. Writes PDFs to results/figures/.
# 18 jobs. 
set -e
cd "$(dirname "$0")/.."

while read -r params; do
  .venv/bin/python run.py device=auto threads=1 wandb.project=null wandb.samples_pdf=true wandb.density_instance=0 eval.seeds=[0] $params </dev/null
done < run_params/run_params_bananahard5_figs.txt
