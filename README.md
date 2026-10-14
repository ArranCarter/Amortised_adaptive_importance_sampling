# Amortised adaptive importance sampling
Code for <a href="https://arxiv.org/abs/2610.16736" target="_blank">amortised adaptive importance sampling</a>.

---

## Project structure

```
run.py                     Evaluate one sampler on one task
train.py                   Train the learnt sampler on one task
pyproject.toml  uv.lock    Dependencies

amortised_adaptive_is/    
  core/                    Runs and evaluates a sampler, with target family definitions.
  learnt/                  The amortised sampler, training, objectives, network definition.
  baselines/               Baselines for comparison.
  reporting/               Plotting and W&B logging.

configs/                   Contains the default parameters for running each sampler and task, and for training the learnt sampler.

experiments/               Utilities for running the experiments, like generating configs.

run_params/ run_scripts/   Parameters for experiments and bash scripts to run them.
```
---

## Setup

We use `uv` to manage the project. Install it following the [instructions](https://docs.astral.sh/uv/getting-started/installation/).

Once installed, run the following command in the root of the project to install the dependencies:
```bash
uv sync
```

This will create `.venv` in the root of the project. To activate the environment, run:
```bash
source .venv/bin/activate
```
---

## Running

`run.py` evaluates a sampler on a task. `train.py` trains the learnt sampler on a task and evaluates it.

```bash
# evaluate a sampler on a task
.venv/bin/python run.py sampler=dm_pmc_resampling task=gmm5_d25

# train the learnt sampler
.venv/bin/python train.py task=gmm5_d5 tag=my-run
```

**Samplers** (`configs/sampler/`): `learnt`, `dm_pmc_resampling`, `m_pmc`, `gapis`, `true_sampler`.
The GAPIS baseline is sensitive to its step-size parameter, for the problems we considered we provide tuned step-sizes in `experiments/gapis_lambda/tuned_lam.json`.

**Tasks** (`configs/task/`): six families -- `bananahard`, `blr`, `funnel`,
`hierarchical`, `corners`, `gmm5` -- at `d` in {2, 5, 25}, with `bananahard` and `funnel` also at 50, plus
a `*fix` single-target variant of the first four.

---



## Reproducing the results

Results are: the **main table**, **generalisation**, the **ablations**, and
**single target vs family**. Scripts for running are in `run_scripts/`, which runs one job per line of its parameter file in `run_params/`. There are a large number of individual jobs, the training jobs took around 3-5 minutes on an NVIDIA L40S for most runs, but up to 15 minutes for the larger models (highest dimension problems). Evaluation jobs took a couple seconds per job. 

For just the learnt sampler on the main 4 families:
```bash
./run_scripts/run_main_learnt.sh      # all 48 jobs
```

For all experiments:

**Stage 1 -- train.** These train and evaluate the learnt samplers, checkpoints are saved and are needed for some evaluations in the second set of runs.

```bash
./run_scripts/run_main_learnt.sh         # 48 jobs
./run_scripts/run_single_train.sh        # 24 jobs 
./run_scripts/run_grad_ablation.sh       # 12 jobs 
./run_scripts/run_em_ablation.sh         # 24 jobs 
./run_scripts/run_obj_ablation.sh        # 24 jobs 
./run_scripts/run_loss_ablation.sh       # 36 jobs 
./run_scripts/run_arch_ablation.sh       # 24 jobs 
./run_scripts/run_cov_ablation.sh        # 24 jobs 
./run_scripts/run_capacity_ablation.sh   # 36 jobs 
./run_scripts/run_bananahard5_train.sh   #  6 jobs (figures only)
```

**Stage 2 -- evaluation**, once the checkpoints exist.

```bash
./run_scripts/run_main_baselines.sh      #  64 jobs, doesnt need any learnt checkpoints
./run_scripts/run_single_eval.sh         #  96 jobs
./run_scripts/run_gen_shiftA.sh          # 612 jobs
./run_scripts/run_gen_shiftB.sh          #  96 jobs
./run_scripts/run_arch_shiftA.sh         #  48 jobs
./run_scripts/run_capacity_shiftA.sh     #  48 jobs
```

### Figures
```bash
./run_scripts/run_funnel50_figs.sh       #  6 jobs
./run_scripts/run_bananahard5_figs.sh    # 18 jobs
```
---

## Citation
```bibtex
@article{carter2026aais,
    title    = {Amortised adaptive importance sampling},
    author   = {Carter, Arran and Elvira, Víctor and Whitammer, Esmeralda S},
    year     = {2026},
    journal  = {arXiv preprint arXiv:2610.16736},
}
```
