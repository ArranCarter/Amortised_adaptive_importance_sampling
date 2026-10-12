"""Evaluation utilities for sampler outputs.
"""

import torch

from amortised_adaptive_is.core.metrics.is_metrics import compute_ess, compute_log_z
from amortised_adaptive_is.core.metrics.transport import sliced_wasserstein, wasserstein
from amortised_adaptive_is.core.seeding import temp_seed


def _estimate_expectation(x: torch.Tensor, fn, log_w: torch.Tensor | None = None) -> torch.Tensor:
    """Estimate E[fn(x)] using either unweighted (reference) or weighted (IS) samples.

    Args:
        x: Samples, shape (B, S, d).
        fn: Function mapping (B, S, d) -> (B, S, ...).
        log_w: Optional unnormalised log weights, shape (B, S).

    Returns:
        Estimated expectation, shape (B, ...).
    """

    y = fn(x)
    if log_w is None:
        return y.mean(dim=1)

    w = torch.softmax(log_w, dim=-1)
    while w.ndim < y.ndim:
        w = w.unsqueeze(-1)
    return (w * y).sum(dim=1)


def _transport_metrics(x: torch.Tensor, gt: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every transport distance between a sampler's points and the shared reference.
    """
    return {
        "w1": wasserstein(x, gt, p=1),
        "w2": wasserstein(x, gt, p=2),
        "sliced_w1": sliced_wasserstein(x, gt, p=1),
    }


def _reference_expectations(target, eval_fns, n_samples: int, chunk: int) -> list[torch.Tensor]:
    """Reference `E[f]` from many draws, accumulated in chunks.

    Args:
        target: Target exposing `sample`.
        eval_fns: Test functions, each mapping (B, S, d) -> (B, S, k).
        n_samples: Total draws.
        chunk: Draws per step.

    Returns:
        One tensor per test function, shape (B, k).
    """
    totals = None
    drawn = 0
    while drawn < n_samples:
        m = min(chunk, n_samples - drawn)
        x = target.sample(m)
        sums = [fn(x).sum(dim=1) for fn in eval_fns]
        totals = sums if totals is None else [t + s for t, s in zip(totals, sums)]
        drawn += m
    return [t / drawn for t in totals]


def evaluate_method(
    result: dict,
    target,
    n_ref_samples: int = 1000,
    compute_transport: bool = True,
    n_expectation_samples: int = 200000,
    expectation_chunk: int = 5000,
    ref_seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Compute evaluation metrics for a sampler result.

    Args:
        result: Sampler output dictionary.
        target: Target distribution with `sample`.
        n_ref_samples: Points on each side of the transport comparisons.
        compute_transport: Whether to compute the transport distances.
        n_expectation_samples: Draws used for the reference `E[f]`.
        expectation_chunk: Draws per accumulation step.
        ref_seed: Seed for the reference draws.

    Returns:
        Dict of metric name to tensor (shape (B,)).
    """
    if ref_seed is None:
        raise ValueError("`ref_seed` must be provided.")
    device = result["centres"].device
    B = result["centres"].shape[0]

    with temp_seed(ref_seed):
        gt_samples = target.sample(n_ref_samples) # (B, n_ref_samples, d)

    # Test functions f0..fK used to probe integral-estimation quality.
    # - f0: mean (vector), E[x]
    # - f1: second moment of norm, E[||x||^2]
    # - f2: fourth moment of norm, E[||x||^4]
    # - f3: E[exp(-||x||^2 / 2)]
    def f0(x: torch.Tensor) -> torch.Tensor:
        return x

    def f1(x: torch.Tensor) -> torch.Tensor:
        return (x**2).sum(dim=-1, keepdim=True)

    def f2(x: torch.Tensor) -> torch.Tensor:
        sq = (x**2).sum(dim=-1, keepdim=True)
        return sq**2

    def f3(x: torch.Tensor) -> torch.Tensor:
        sq = (x**2).sum(dim=-1, keepdim=True)
        return torch.exp(-0.5 * sq)

    eval_fns = [f0, f1, f2, f3]
    with temp_seed(ref_seed + 1):
        gt_expectations = _reference_expectations(
            target, eval_fns, n_expectation_samples, expectation_chunk)

    metrics: dict[str, torch.Tensor] = {}

    # IS methods.
    #
    # _all pools every sample drawn across the trajectory and weights each by its own
    # generating proposal, w = pi(x) / q_t(x).
    x_all = result["all_samples"]
    lw_all = result["all_log_w"]

    for i, fn in enumerate(eval_fns):
        est_all = _estimate_expectation(x_all, fn, log_w=lw_all)
        diff = est_all - gt_expectations[i]
        metrics[f"f{i}_err_all"] = diff.reshape(diff.shape[0], -1).norm(dim=1)

    metrics["log_z_all"] = compute_log_z(lw_all)
    # The per-instance absolute error in log Z
    metrics["logz_abserr_all"] = metrics["log_z_all"].abs()
    metrics["ess_all"] = compute_ess(lw_all)

    x_fin = result["final_samples"]
    lw_fin = result["final_log_w"]

    for i, fn in enumerate(eval_fns):
        est_fin = _estimate_expectation(x_fin, fn, log_w=lw_fin)
        diff = est_fin - gt_expectations[i]
        metrics[f"f{i}_err_final"] = diff.reshape(diff.shape[0], -1).norm(dim=1)

    metrics["ess_final"] = compute_ess(lw_fin)
    metrics["log_z_final"] = compute_log_z(lw_fin)
    metrics["logz_abserr_final"] = metrics["log_z_final"].abs()

    if compute_transport:
        x_src, lw_src = x_fin, lw_fin
        pool = x_fin.shape[1]

        # for reproducibility
        metrics["transport_pool"] = torch.full((B,), float(pool), device=device)

        # resample by weight, with replacement
        w = torch.softmax(lw_src, dim=-1)
        idx = torch.multinomial(w, num_samples=n_ref_samples, replacement=True)
        ar = torch.arange(B, device=device).unsqueeze(-1)
        x_resamp = x_src[ar, idx]

        metrics.update(_transport_metrics(x_resamp, gt_samples))

    # Divergences between the final proposal q and the target pi, estimated on samples drawn from q.
    if bool(result.get("proposal_is_target", False)):
        x_proposal = target.sample(n_ref_samples)
        target_log_prob = target.log_prob(x_proposal)
        proposal_log_prob = target_log_prob
    else:
        from amortised_adaptive_is.core.proposal import log_density_components, sample

        # Every sampler reports the proposal it ended on.
        prop = result["final_proposal"]
        x_proposal, _ = sample(prop, n_ref_samples, mode="categorical")
        target_log_prob = target.log_prob(x_proposal)
        proposal_log_prob, _ = log_density_components(
            prop, x_proposal, count_as_proposal=False
        )

    metrics["reverse_kl"] = (proposal_log_prob - target_log_prob).mean(dim=-1)
    metrics["chisq"] = ((target_log_prob - proposal_log_prob).mul(2).exp()).mean(dim=-1)

    if compute_transport:
        # Transport distance between the final proposal and the target.
        metrics.update({f"{k}_prop": v
                        for k, v in _transport_metrics(x_proposal, gt_samples).items()})

    return metrics
