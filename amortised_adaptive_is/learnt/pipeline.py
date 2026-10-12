"""Train the learnt sampler, score the result, and record both.
"""

import dataclasses
import json
import pathlib
import time
from typing import Any

import torch

from amortised_adaptive_is.core.paths import RESULTS_DIR, write_per_instance
from amortised_adaptive_is.core.runner import (
    DEV_SEEDS,
    N_REF_SAMPLES_DEFAULT,
    score_sampler,
)
from amortised_adaptive_is.core.specs import TaskSpec
from amortised_adaptive_is.learnt.train import (
    TrainConfig,
    make_learnt_runner,
    save,
    train,
)
from amortised_adaptive_is.reporting import format_table


def _wandb_probe(cfg: TrainConfig, logger, density_dir=None, label: str = "run",
                 density_instance: int = 0):
    """Feeds the rollout diagnostics to W&B.
    """

    def probe(epoch, row, records, val_target, beta):
        modes = getattr(val_target, "centres", None)
        var = getattr(val_target, "variances", None)
        sd = float(var.clamp_min(1e-30).sqrt().mean()) if var is not None else 1.0
        logger.on_validate(epoch, row, records, modes, sd, beta, cfg.n_epochs, target=val_target)
        if density_dir is not None:
            every = getattr(logger, "figure_every", 0) or 0
            last = epoch >= cfg.n_epochs - 1
            if last or (every and epoch % every == 0):
                from amortised_adaptive_is.reporting.wandb_rollout import save_density_pdfs

                save_density_pdfs(records, density_dir, f"{label}-e{epoch:04d}",
                                  instance=density_instance)

    return probe


def train_and_score(
    spec: TaskSpec,
    cfg: TrainConfig,
    label: str,
    eval_seeds=DEV_SEEDS,
    eval_batch_size: int = 64,
    wandb: dict[str, Any] | None = None,
    verbose: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Train on one task, evaluate the selected checkpoint, and write the record.

    Args:
        spec: Task specification.
        cfg: Training configuration.
        label: Run name; also the checkpoint and record filename.
        eval_seeds: Evaluation seeds.
        eval_batch_size: Instances per evaluation seed.
        wandb: Logging options (`project`, `mode`, `group`, `tags`, `figure_every`, `density_pdf`, `density_dir`).
        verbose: Print progress and the score table.

    Returns:
        The run record, as written to `results/runs/<label>.json`.
    """
    if verbose:
        print(f"=== {label} ===")
        print(f"task {spec.name}: N={spec.n_components} T={spec.n_steps} K={spec.k_particles}")
        print(f"cfg  {dataclasses.asdict(cfg)}")

    logger, probe = None, None
    if wandb and wandb.get("project"):
        from amortised_adaptive_is.reporting.wandb_rollout import WandbLogger

        logger = WandbLogger(
            project=wandb["project"],
            name=label,
            config={**dataclasses.asdict(cfg), "task": spec.name,
                    "spatial_dim": spec.spatial_dim, "n_components": spec.n_components,
                    "n_steps": spec.n_steps, "k_particles": spec.k_particles},
            mode=wandb.get("mode", "online"), group=wandb.get("group"),
            tags=tuple(wandb.get("tags") or ()),
            figure_every=int(wandb.get("figure_every") or 0),
        )

        ddir = (pathlib.Path(wandb["density_dir"]) if wandb.get("density_dir")
                else RESULTS_DIR / "figures" / label) if wandb.get("density_pdf") else None
        probe = _wandb_probe(cfg, logger, density_dir=ddir, label=label,
                             density_instance=int(wandb.get("density_instance") or 0))

    res = train(spec, cfg, on_validate=probe, device=device, dtype=dtype)
    if verbose:
        print(
            f"trained {res.net.n_params()} params in {res.seconds:.1f}s, "
            f"{res.target_evals / 1e6:.1f}M target evals, "
            f"val_ess {res.best_val_ess:.4f}, val_xent {res.best_val_xent:.4f} "
            f"(selected epoch {res.best_epoch})"
        )

    t0 = time.time()
    # Per-instance metrics
    row, per_vec = score_sampler(
        make_learnt_runner(res.net, cfg), spec,
        seeds=eval_seeds, batch_size=eval_batch_size, per_instance=True,
        device=device, dtype=dtype,
    )
    rows, per = {"learnt": row}, {"learnt": per_vec}
    if verbose:
        print(f"\nbench [{time.time() - t0:.1f}s]")
        print(format_table(rows))


    pf = write_per_instance(spec.name, f"learnt/{label}", per["learnt"])
    if verbose:
        print(f"per-instance -> {pf}")

    if logger is not None:
        logger.summary({f"score/{k}": v for k, v in rows["learnt"].items()}
                       | {"best_epoch": res.best_epoch, "diverged": res.diverged,
                          "nan_abort_epoch": res.nan_abort_epoch,
                          "train_target_evals": res.target_evals})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = RESULTS_DIR / f"{label}.pt"
    save(ckpt, res.net, cfg, spec, extra={"scores": rows["learnt"]}, snapshots=res.snapshots)

    record = {
        "label": label,
        "task": spec.name,
        "spec": {k: v for k, v in spec.__dict__.items()},
        "cfg": dataclasses.asdict(cfg),
        "n_params": res.net.n_params(),
        "train_seconds": res.seconds,
        "train_target_evals": res.target_evals,
        "best_val_ess": res.best_val_ess,
        "best_val_xent": res.best_val_xent,
        "best_epoch": res.best_epoch,
        "diverged": res.diverged,
        "nan_abort_epoch": res.nan_abort_epoch,
        "metric_settings": {"n_ref_samples": N_REF_SAMPLES_DEFAULT,
                            "eval_seeds": list(eval_seeds),
                            "eval_batch_size": eval_batch_size},
        "history": res.history,
        "scores": rows,
        "checkpoint": str(ckpt),
    }
    # One file per run
    runs_dir = RESULTS_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out = runs_dir / f"{label}.json"
    out.write_text(json.dumps(record, indent=2))
    if verbose:
        print(f"\nwrote {out}")
    if logger is not None:
        logger.finish()
    return record
