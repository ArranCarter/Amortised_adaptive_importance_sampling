"""Run a sampler on fixed seeds and evaluate it.

A runner is any callable with the signature:
    runner(target, init_centres, init_variances, n_steps, k_particles) -> dict
"""

from collections.abc import Callable, Iterable
from typing import Any

import torch

from amortised_adaptive_is.core.counters import (
    CountingTarget,
    DensityEvalCounter,
    density_eval_counter,
)
from amortised_adaptive_is.core.evaluate import evaluate_method
from amortised_adaptive_is.core.gmm import draw_initial_centres
from amortised_adaptive_is.core.seeding import set_seed
from amortised_adaptive_is.core.specs import TaskSpec
from amortised_adaptive_is.core.targets import build_target


def make_eval_inputs(
    spec: TaskSpec,
    seed: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """Build the target batch and initial proposal for one evaluation run.

    Args:
        spec: Task specification.
        seed: RNG seed.
        batch_size: Number of independent target instances `B`.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        Tuple of (target, init_centres, init_variances).
    """
    set_seed(int(seed))
    target = build_target(
        spec.target_cfg,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        n_components=spec.n_components,
        spatial_dim=spec.spatial_dim,
    )
    
    init_centres = draw_initial_centres(
        batch_size=batch_size,
        n_components=spec.n_components,
        spatial_dim=spec.spatial_dim,
        centre_scale=spec.centre_scale,
        centre_sample_type=spec.centre_sample_type,
        device=device,
        dtype=dtype,
    )
    init_variances = torch.ones_like(init_centres) * float(spec.init_variance)
    return target, init_centres, init_variances


def _reduce(metrics: dict[str, Any]) -> dict[str, float]:
    """Collapse the per-target-instance vectors to a mean.
    """
    out: dict[str, float] = {}
    for k, v in metrics.items():
        out[k] = float(v.mean().item()) if torch.is_tensor(v) else float(v)
    return out


def _per_instance(metrics: dict[str, Any]) -> dict[str, list[float]]:
    """Keep the per-target-instance vectors instead of collapsing them to a mean.

    Args:
        metrics: Output of `evaluate_method`, values either (B,) tensors or scalars.

    Returns:
        Metric name to per-instance values, for the tensor-valued entries only.
    """
    out: dict[str, list[float]] = {}
    for k, v in metrics.items():
        if torch.is_tensor(v) and v.dim() >= 1:
            out[k] = [float(z) for z in v.detach().flatten()]
    return out


def _replace_non_finite(result: dict) -> float:
    """Replace non-finite log-weights and report how many there were.

    Args:
        result: Sampler output dictionary, modified in place.

    Returns:
        Fraction of non-finite entries.
    """
    bad, total = 0, 0
    for key in ("final_log_w", "all_log_w"):
        lw = result.get(key)
        if lw is None:
            continue
        finite = torch.isfinite(lw)
        bad += int((~finite).sum())
        total += int(finite.numel())
        if not bool(finite.all()):
            result[key] = torch.where(finite, lw, torch.full_like(lw, -1e30))
    
    for key in ("final_samples", "all_samples", "centres"):
        x = result.get(key)
        if x is not None and not bool(torch.isfinite(x).all()):
            result[key] = torch.nan_to_num(x, nan=0.0, posinf=1e30, neginf=-1e30)

    # Variances must stay strictly positive and the logits must stay a valid distribution.
    var = result.get("variances")
    if var is not None and not bool(torch.isfinite(var).all()):
        result["variances"] = torch.nan_to_num(var, nan=1.0, posinf=1e30, neginf=1.0).clamp(min=1e-12)

    logits = result.get("log_component_logits")
    if logits is not None and not bool(torch.isfinite(logits).all()):
        finite = torch.isfinite(logits)
        # A row with no finite entry falls back to uniform.
        row_ok = finite.any(dim=-1, keepdim=True)
        logits = torch.where(finite, logits, torch.full_like(logits, -1e30))
        result["log_component_logits"] = torch.where(row_ok, logits, torch.zeros_like(logits))

    return bad / max(1, total)



N_REF_SAMPLES_DEFAULT = 300

# Draws to compute true values for expectations.
N_EXPECTATION_SAMPLES_DEFAULT = 200000


def score_runner(
    runner: Callable[..., dict],
    spec: TaskSpec,
    seed: int = 0,
    batch_size: int = 64,
    n_ref_samples: int = N_REF_SAMPLES_DEFAULT,
    compute_transport: bool = True,
    n_expectation_samples: int = N_EXPECTATION_SAMPLES_DEFAULT,
    expectation_chunk: int = 5000,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    n_steps: int | None = None,
    return_per_instance: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, list[float]]]:
    """Run one sampler and score it.

    Args:
        runner: Callable `(target, init_centres, init_variances, n_steps, k_particles) -> result dict`.
        spec: Task specification.
        seed: RNG seed for targets, initial proposal, and the run itself.
        batch_size: Number of target instances.
        n_ref_samples: Points on each side of the transport distances.
        compute_transport: Whether to compute the transport distances.
        n_expectation_samples: Number of draws to compute the reference `E[f]`.
        expectation_chunk: Chunk the expectation calculation into this many samples.
        device: Defaults to CPU.
        dtype: Defaults to float32.
        n_steps: Override the task's number of steps.
        return_per_instance: Also return the per-target-instance metric vectors.

    Returns:
        Flat dict of scalar metrics. When `return_per_instance`, a tuple of (flat dict, per-instance vectors).
    """
    device = device or torch.device("cpu")
    target, init_centres, init_variances = make_eval_inputs(spec, seed, batch_size, device, dtype)
    steps = spec.n_steps if n_steps is None else int(n_steps)

    counter = DensityEvalCounter()
    with density_eval_counter(counter):
        result = runner(
            target=CountingTarget(target, counter),
            init_centres=init_centres,
            init_variances=init_variances,
            n_steps=steps,
            k_particles=spec.k_particles,
        )

    nan_frac = _replace_non_finite(result)
    raw = evaluate_method(
        result=result,
        target=target,
        n_ref_samples=n_ref_samples,
        compute_transport=compute_transport,
        n_expectation_samples=n_expectation_samples,
        expectation_chunk=expectation_chunk,
        ref_seed=10000 + int(seed),
    )
    metrics = _reduce(raw)
    metrics["nan_frac"] = nan_frac

    # Targets are normalised, so the ground truth is log Z = 0
    metrics["logz_err"] = metrics.get("logz_abserr_final", float("nan"))
    metrics["logz_err_all"] = metrics.get("logz_abserr_all", float("nan"))

    # cost counting
    metrics["target_evals"] = float(counter.target_log_prob_evals)
    metrics["target_grad_evals"] = float(counter.target_grad_evals)
    metrics["target_hess_evals"] = float(counter.target_hess_evals)
    metrics["target_cost"] = float(counter.target_cost)
    metrics["proposal_evals"] = float(counter.proposal_log_prob_evals)
    metrics["n_steps"] = float(steps)
    if not return_per_instance:
        return metrics

    per = _per_instance(raw)
    return metrics, per


# Evaluation seed splits. Four seed-batches of 64 to reduce memory requirement.
DEV_SEEDS: tuple[int, ...] = (0, 1, 2, 3)


TEST_SEEDS: tuple[int, ...] = (4, 5, 6, 7)


def score_sampler(
    runner: Callable[..., dict],
    spec: TaskSpec,
    seeds: Iterable[int] = DEV_SEEDS,
    per_instance: bool = False,
    **kwargs: Any,
) -> dict[str, float] | tuple[dict[str, float], dict[str, list[float]]]:
    """Score one sampler over several seeds and aggregate.

    Args:
        runner: A `run`-signature callable.
        spec: Task specification.
        seeds: Seeds to aggregate over.
        per_instance: Also return the concatenated per-instance vectors, in seed order.
        **kwargs: Forwarded to `score_runner`.

    Returns:
        The aggregated metrics, or a tuple of (metrics, per-instance vectors) when `per_instance` is set.
    """
    per_seed = []
    merged: dict[str, list[float]] = {}
    for s in seeds:
        if per_instance:
            m, p = score_runner(runner, spec, seed=s, return_per_instance=True, **kwargs)
            for k, v in p.items():
                merged.setdefault(k, []).extend(v)
        else:
            m = score_runner(runner, spec, seed=s, **kwargs)
        per_seed.append(m)

    agg: dict[str, float] = {}

    for k in per_seed[0]:
        vals = torch.tensor([p[k] for p in per_seed], dtype=torch.float64)
        agg[k] = float(vals.mean())

    # Compute quantile metrics
    QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
    for k, v in merged.items():
        t = torch.tensor(v, dtype=torch.float64)
        finite = t[torch.isfinite(t)]
        if finite.numel():
            q = torch.quantile(finite, torch.tensor(QUANTILES, dtype=torch.float64))
            (agg[f"{k}_p05"], agg[f"{k}_p25"], agg[f"{k}_median"],
             agg[f"{k}_p75"], agg[f"{k}_p95"]) = (float(z) for z in q)
            
            agg[f"{k}_iqr"] = float(q[3] - q[1])
    agg["n_instances"] = float(len(next(iter(merged.values()), [])) or 0)

    return (agg, merged) if per_instance else agg
