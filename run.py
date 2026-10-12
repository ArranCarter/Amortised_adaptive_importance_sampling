"""The entrypoint: evaluate one sampler on one task, selected by config.

    .venv/bin/python run.py sampler=dm_pmc_resampling task=gmm5_d25
    .venv/bin/python run.py sampler=gapis task=blr_d5 sampler.lam=0.001 task.n_components=10
    .venv/bin/python run.py sampler=learnt task=gmm5_d2 sampler.checkpoint=results/C-....pt
"""

import json
import pathlib

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from amortised_adaptive_is.core.paths import CONFIG_DIR, RESULTS_DIR, write_per_instance
from amortised_adaptive_is.core.registry import build as build_sampler
from amortised_adaptive_is.core.runner import score_sampler
from amortised_adaptive_is.core.specs import TaskSpec
from amortised_adaptive_is.reporting import format_table


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _dtype(name: str) -> torch.dtype:
    try:
        return {"float32": torch.float32, "float64": torch.float64}[name]
    except KeyError:
        raise ValueError(f"dtype must be float32 or float64, got {name!r}") from None


# Samplers that carry no per-step (proposal, particles) trace
NO_TRACE = ("true_sampler",)


def _log_wandb(cfg, spec, name, row, runner, device, dtype) -> None:
    """Report this job's scores to W&B, and/or write the per-step density PDFs.
    """
    wb = OmegaConf.to_container(cfg.wandb, resolve=True) if "wandb" in cfg else None
    if not wb:
        return
    want_wandb = bool(wb.get("project"))
    want_density = bool(wb.get("density_pdf"))
    want_samples = bool(wb.get("samples_pdf"))
    if not (want_wandb or want_density or want_samples):
        return

    logger = None
    if want_wandb:
        from amortised_adaptive_is.reporting.wandb_rollout import BaselineWandbLogger

        fam, _, dim = spec.name.rpartition("_d")
        logger = BaselineWandbLogger(
            project=wb["project"],
            name=wb.get("name") or f"{name}-{spec.name}-n{spec.n_components}",
            config={"task": spec.name, "method": name, "n_components": spec.n_components,
                    "k_particles": spec.k_particles, "n_steps": spec.n_steps,
                    "dim": spec.spatial_dim,
                    "sampler": OmegaConf.to_container(cfg.sampler, resolve=True),
                    "eval": OmegaConf.to_container(cfg.eval, resolve=True)},
            mode=wb.get("mode", "online"), group=wb.get("group") or spec.name,
            tags=tuple(wb.get("tags") or ()) or (name, fam, f"d{dim}", f"n{spec.n_components}"),
        )
    try:

        want_trace = ((wb.get("rollout") or want_density or want_samples)
                      and name not in NO_TRACE)
        recs = None
        if want_trace:
            from amortised_adaptive_is.core.runner import make_eval_inputs
            from amortised_adaptive_is.reporting.wandb_rollout import baseline_records

            target, c0, v0 = make_eval_inputs(
                spec, int(cfg.eval.seeds[0]), int(cfg.eval.batch_size), device, dtype)
            same = wb.get("density_same_init")
            if same is not None:
                # Every instance starts from instance `same`'s proposal, so a pair of rollouts
                # differs only in the target.
                k = int(same)
                c0 = c0[k:k + 1].expand_as(c0).clone()
                v0 = v0[k:k + 1].expand_as(v0).clone()
            result = runner(target, init_centres=c0, init_variances=v0,
                            n_steps=spec.n_steps, k_particles=spec.k_particles)
            recs = baseline_records(result) if "per_step" in result else None
            modes = getattr(target, "centres", None)
            var = getattr(target, "variances", None)
            if recs is not None and logger is not None and wb.get("rollout"):
                logger.log_rollout(
                    recs, target, modes,
                    float(var.clamp_min(1e-30).sqrt().mean()) if var is not None else 1.0,
                    f"{name} on {spec.name}")
            if recs is not None and want_density:
                # produce plots
                from amortised_adaptive_is.reporting.wandb_rollout import save_density_pdfs

                out_dir = wb.get("density_dir") or (RESULTS_DIR / "figures" / name)
                paths = save_density_pdfs(recs, out_dir, f"{name}-{spec.name}", target=target,
                                          instance=int(wb.get("density_instance") or 0),
                                          view=wb.get("density_view"))
                print(f"  wrote {len(paths)} density pdf(s) -> {out_dir}")
            if recs is not None and want_samples:
                from amortised_adaptive_is.reporting.wandb_rollout import save_clean_samples

                out_dir = wb.get("density_dir") or (RESULTS_DIR / "figures" / name)
                fs = save_clean_samples(
                    recs, out_dir, f"{name}-{spec.name}", target=target,
                    instance=int(wb.get("density_instance") or 0),
                    view=wb.get("density_view"))
                for f in fs:
                    print(f"  wrote {f}")
        if recs is None and (want_density or want_samples or wb.get("rollout")):
            print(f"  {name} has no per-step trace, nothing to draw")
        if logger is not None:
            logger.summary({f"score/{k}": v for k, v in row.items()})
    finally:
        if logger is not None:
            logger.finish()


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    """Score the configured sampler on the configured task."""
    if cfg.threads:
        torch.set_num_threads(int(cfg.threads))
    device, dtype = _device(str(cfg.device)), _dtype(str(cfg.dtype))

    spec = TaskSpec(**OmegaConf.to_container(cfg.task, resolve=True))
    name = str(cfg.sampler.name)
    runner = build_sampler(name, cfg.sampler, device=device, dtype=dtype)

    # `per_instance=True` so this run can be compared with the others on the same target instances.
    row, per_vec = score_sampler(
        runner,
        spec,
        seeds=tuple(cfg.eval.seeds),
        batch_size=int(cfg.eval.batch_size),
        n_ref_samples=int(cfg.eval.n_ref_samples),
        n_expectation_samples=int(cfg.eval.n_expectation_samples),
        expectation_chunk=int(cfg.eval.expectation_chunk),
        device=device,
        dtype=dtype,
        per_instance=True,
    )
    rows = {name: row}
    pf = write_per_instance(spec.name, name, per_vec)
    print(f"per-instance -> {pf}")
    print(f"\n{spec.name}  N={spec.n_components} T={spec.n_steps} K={spec.k_particles}")
    print(format_table(rows))

    _log_wandb(cfg, spec, name, rows[name], runner, device, dtype)

    
    out = pathlib.Path(cfg.out) if cfg.out else RESULTS_DIR / "runs" / f"{name}-{spec.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "task": spec.name,
        "sampler": OmegaConf.to_container(cfg.sampler, resolve=True),
        "spec": OmegaConf.to_container(cfg.task, resolve=True),
        "eval": OmegaConf.to_container(cfg.eval, resolve=True),
        "scores": rows,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
