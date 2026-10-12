"""Baseline adaptive importance sampling algorithms.
"""

import numpy as np
import torch
from omegaconf import DictConfig

from amortised_adaptive_is.core.counters import get_active_density_eval_counter
from amortised_adaptive_is.core.gmm import (
    batched_gaussian_mixture_log_density,
    draw_iid_from_mixture,
    draw_k_per_component,
    multinomial_resample,
    soft_em_update,
)
from amortised_adaptive_is.core.proposal import (
    Proposal,
    diagonal,
    log_density_components,
    sample,
)

EPS = 1e-8

def _per_step_trace(centres_hist, variances_hist, logits_hist, samples_list, log_w_list,
                    scale_tril_hist=None):
    """Per-block `(proposal, particles, log-weights)` trace, for the rollout figures.

    Args:
        centres_hist: Means before each block's update, `T + 1` entries.
        variances_hist: Variances alongside `centres_hist`.
        logits_hist: Mixture log-weights alongside `centres_hist`.
        samples_list: Particles per block.
        log_w_list: Log-weights per block.
        scale_tril_hist: Optional Cholesky factors of the proposals.

    Returns:
        List of per-block dicts.
    """
    lens = {len(centres_hist), len(variances_hist), len(logits_hist),
            len(samples_list), len(log_w_list)}
    if len(lens) != 1:
        raise ValueError(
            f"proposal history and sample blocks disagree: centres={len(centres_hist)}, "
            f"variances={len(variances_hist)}, logits={len(logits_hist)}, "
            f"samples={len(samples_list)}, log_w={len(log_w_list)}"
        )
    out = [
        {"centres": c, "variances": v, "log_component_logits": g, "samples": x, "log_w": w}
        for c, v, g, x, w in zip(centres_hist, variances_hist, logits_hist,
                                 samples_list, log_w_list)
    ]

    if scale_tril_hist is not None:
        for entry, L in zip(out, scale_tril_hist):
            entry["scale_tril"] = L
    return out


@torch.no_grad()
def run_m_pmc(target, init_centres, init_variances, n_steps, k_particles):
    """Run M-PMC. Diagonal covariance, Rao-Blackwellised EM update of the mixture parameters.

    Args:
        target: Target distribution exposing `log_prob`.
        init_centres: Initial proposal centres, shape (B, N, d).
        init_variances: Initial proposal variances, shape (B, N, d).
        n_steps: Number of adaptation steps.
        k_particles: Particles per component per step.

    Returns:
        Result dictionary compatible with evaluation/plotting.
    """
    B, N, _d = init_centres.shape
    device = init_centres.device
    dtype = init_centres.dtype

    centres = init_centres.clone()
    variances = init_variances.clone()
    log_component_logits = torch.log(torch.ones((B, N), device=device, dtype=dtype) / N)

    n_per_step = N * k_particles
    all_samples_list, all_log_pi_list, all_log_w_list = [], [], []
    centres_hist, variances_hist, logits_hist = [], [], []

    def draw_and_weight(centres, variances, logits):
        x = draw_iid_from_mixture(centres, variances, logits, n_per_step)
        log_pi = target.log_prob(x)
        log_w = log_pi - batched_gaussian_mixture_log_density(x, centres, variances, logits)
        return x, log_pi, log_w

    for _ in range(n_steps):
        centres_hist.append(centres.clone())
        variances_hist.append(variances.clone())
        logits_hist.append(log_component_logits.clone())
        x, log_pi, log_w = draw_and_weight(centres, variances, log_component_logits)
        all_samples_list.append(x)
        all_log_pi_list.append(log_pi)
        all_log_w_list.append(log_w)

        centres, variances, log_component_logits = soft_em_update(
            x, log_w, centres, variances, log_component_logits
        )

    # One extra final step, matching the other IS baselines.
    centres_hist.append(centres.clone())
    variances_hist.append(variances.clone())
    logits_hist.append(log_component_logits.clone())
    x, log_pi, log_w = draw_and_weight(centres, variances, log_component_logits)
    all_samples_list.append(x)
    all_log_pi_list.append(log_pi)
    all_log_w_list.append(log_w)

    return {
        "centres": centres,
        "variances": variances,
        "log_component_logits": log_component_logits,
        "final_proposal": diagonal(centres, variances, log_component_logits),
        "per_step": _per_step_trace(centres_hist, variances_hist, logits_hist, all_samples_list, all_log_w_list),
        "all_samples": torch.cat(all_samples_list, dim=1),
        "all_log_w": torch.cat(all_log_w_list, dim=1),
        "final_samples": all_samples_list[-1],
        "final_log_w": all_log_w_list[-1],
    }

@torch.no_grad()
def run_dm_pmc_resampling(target, init_centres, init_variances, n_steps, k_particles):
    """Run DM-PMC baseline using global resampling.

    Args:
        target: Target distribution exposing `log_prob`.
        init_centres: Initial proposal centres, shape (B, N, d).
        init_variances: Initial proposal variances, shape (B, N, d).
        n_steps: Number of adaptation steps.
        k_particles: Particles per component per step.

    Returns:
        Result dictionary compatible with evaluation/plotting.
    """
    B, N, _d = init_centres.shape
    device = init_centres.device
    dtype = init_centres.dtype

    centres = init_centres.clone()
    variances = init_variances.clone()
    log_component_logits = torch.log(torch.ones((B, N), device=device, dtype=dtype) / N)

    all_samples_list, all_log_pi_list, all_log_w_list = [], [], []
    centres_hist, variances_hist, logits_hist = [], [], []

    for _ in range(n_steps):
        centres_hist.append(centres.clone())
        variances_hist.append(variances.clone())
        logits_hist.append(log_component_logits.clone())

        x = draw_k_per_component(centres, variances, k_particles)
        log_pi = target.log_prob(x)
        log_w = log_pi - batched_gaussian_mixture_log_density(x, centres, variances, log_component_logits)

        all_samples_list.append(x)
        all_log_pi_list.append(log_pi)
        all_log_w_list.append(log_w)
        x_resamp = multinomial_resample(x, log_w, N) 

        centres = x_resamp

    # One extra final step: sample/weight from the final proposal parameters.
    centres_hist.append(centres.clone())
    variances_hist.append(variances.clone())
    logits_hist.append(log_component_logits.clone())

    x = draw_k_per_component(centres, variances, k_particles)
    log_pi = target.log_prob(x)
    log_w = log_pi - batched_gaussian_mixture_log_density(x, centres, variances, log_component_logits)

    all_samples_list.append(x)
    all_log_pi_list.append(log_pi)
    all_log_w_list.append(log_w)

    all_samples = torch.cat(all_samples_list, dim=1)
    all_log_pi = torch.cat(all_log_pi_list, dim=1)
    all_log_w = torch.cat(all_log_w_list, dim=1)


    final_samples = all_samples_list[-1]
    final_log_w = all_log_w_list[-1]


    return {
        "centres": centres,
        "variances": variances,
        "log_component_logits": log_component_logits,
        "final_proposal": diagonal(centres, variances, log_component_logits),
        "per_step": _per_step_trace(centres_hist, variances_hist, logits_hist, all_samples_list, all_log_w_list),
        "all_samples": all_samples,
        "all_log_w": all_log_w,
        "final_samples": final_samples,
        "final_log_w": final_log_w,
    }

def _log_target_derivatives(target, mu, need_hessian: bool):
    """Gradient and Hessian of `log pi` at each point of `mu`.

    Args:
        target: Target with `log_prob`, shape (B, S, d) -> (B, S).
        mu: Evaluation points, shape (B, N, d).
        need_hessian: Also return the Hessian.

    Returns:
        Tuple of (grad, hess) with shapes (B, N, d) and (B, N, d, d).
    """
    d = mu.shape[-1]
    mu_g = mu.detach().requires_grad_(True)
    with torch.enable_grad():
        lp = target.log_prob(mu_g)
        _counter = get_active_density_eval_counter()
        if _counter is not None:
            # One gradient evaluation per point, plus d more per point when the Hessian is needed.
            _counter.add_target_grad(mu)
            if need_hessian:
                _counter.add_target_hessian(mu)
        grad = torch.autograd.grad(lp.sum(), mu_g, create_graph=need_hessian)[0]
        if not need_hessian:
            return grad.detach(), None
        rows = [
            torch.autograd.grad(grad[..., j].sum(), mu_g, retain_graph=(j < d - 1))[0]
            for j in range(d)
        ]
        hess = torch.stack(rows, dim=-2)
    return grad.detach(), hess.detach()


@torch.no_grad()
def _invertible_hessian(H: torch.Tensor, eye_d: torch.Tensor):
    """Which components' Hessian can be inverted as a covariance, returns identity if not invertible.

    Args:
        H: Symmetrised Hessian of `-log pi`, shape (B, N, d, d).
        eye_d: Identity broadcast to `H`'s shape.

    Returns:
        `(ok, H_safe)`. `H_safe` carries the identity wherever the screen failed.
    """
    finite = torch.isfinite(H).all(dim=-1).all(dim=-1)
    H = torch.where(finite[..., None, None], H, eye_d)
    _, info = torch.linalg.cholesky_ex(H)
    ok = (info == 0) & finite & (torch.linalg.matrix_rank(H) == H.shape[-1])
    return ok, torch.where(ok[..., None, None], H, eye_d)


def run_gapis(
    target,
    init_centres,
    init_variances,
    n_steps,
    k_particles,
    *,
    lam: float,
    repulse_nu: float = 0.05,
    repulse_xi: float = 1.04,
    adapt_covariance: bool = True,
    full_cov: bool = True,
    min_variance: float = 1e-8,
    max_variance: float = 1e8,
):
    """Run GAPIS.

    Args:
        target: Target distribution with `log_prob`.
        init_centres: Initial proposal means, shape (B, N, d).
        init_variances: Initial proposal variances, shape (B, N, d).
        n_steps: Number of adaptation steps `T`.
        k_particles: Particles per proposal per iteration `K`.
        lam: Gradient step size `lambda`.
        repulse_nu: `nu` of the periodic repulsion schedule.
        repulse_xi: `xi` of the periodic repulsion schedule.
        adapt_covariance: If False, adapt means only..
        full_cov: Use the full covariance.
        min_variance: Floor on the adapted variances.
        max_variance: Ceiling on the adapted variances.

    Returns:
        Result dictionary compatible with evaluation/plotting.
    """
    B, N, d = init_centres.shape
    device = init_centres.device
    dtype = init_centres.dtype

    centres = init_centres.clone()
    variances = init_variances.clone()
    
    scale_tril = torch.diag_embed(variances.clamp(min=1e-30).sqrt())
    
    log_component_logits = torch.log(torch.ones((B, N), device=device, dtype=dtype) / N)

    # Gradient carried from the previous iteration
    carried_grad = None
    eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
    all_samples_list, all_log_pi_list, all_log_w_list = [], [], []
    centres_hist, variances_hist, logits_hist, scale_tril_hist = [], [], [], []

    def as_proposal(centres, L):
        """The current mixture as a `Proposal`."""
        diag = torch.diagonal(L, dim1=-2, dim2=-1).clamp(min=1e-30)
        return Proposal(
            mu=centres,
            log_diag=diag.log(),
            low=L - torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1)),
            logits=log_component_logits,
        )

    def draw_and_weight(centres, variances, L):
        centres_hist.append(centres.clone())
        variances_hist.append(variances.clone())
        logits_hist.append(log_component_logits.clone())
        scale_tril_hist.append(L.clone())
        if full_cov:
            # stratified
            prop = as_proposal(centres, L)
            x, _ = sample(prop, N * k_particles, mode="stratified")
            log_pi = target.log_prob(x)
            log_q, _ = log_density_components(prop, x)
            return x, log_pi, log_pi - log_q
        x = draw_k_per_component(centres, variances, k_particles)
        log_pi = target.log_prob(x)
        log_w = log_pi - batched_gaussian_mixture_log_density(
            x, centres, variances, log_component_logits
        )
        return x, log_pi, log_w

    for t in range(1, n_steps + 1):
        # mean adaptation
        if carried_grad is not None:
            grad = carried_grad
        else:
            grad, _ = _log_target_derivatives(target, centres, need_hessian=False)

        diff = centres.unsqueeze(2) - centres.unsqueeze(1) # d_ij, (B, N, N, d)
        dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-12) # ||d_ij||
        g_t = float(np.sin(2.0 * np.pi * repulse_nu * t) + repulse_xi)
        repulse = g_t * diff / dist.pow(3) # m_i = m_j = 1
        repulse = repulse.masked_fill(eye.unsqueeze(-1), 0.0).sum(dim=2)

        centres = torch.nan_to_num(centres + lam * grad + repulse, nan=0.0)

        # covariance adaptation
        if adapt_covariance:
            # Hessian of -log pi at the new means
            carried_grad, hess_new = _log_target_derivatives(target, centres, need_hessian=True)
            H = -0.5 * (hess_new + hess_new.transpose(-1, -2)) # symmetrise
            eye_d = torch.eye(d, device=device, dtype=dtype).expand_as(H)

            ok, H_safe = _invertible_hessian(H, eye_d)
            cov = torch.linalg.inv(H_safe)
            new_var = torch.diagonal(cov, dim1=-2, dim2=-1)
            new_var = torch.nan_to_num(new_var, nan=1.0).clamp(min=min_variance, max=max_variance)
            if full_cov:
                # Rescale rows and columns so the diagonal lands inside the variance limits.
                dg = torch.diagonal(cov, dim1=-2, dim2=-1).clamp(min=1e-30)
                scale = (new_var / dg).sqrt()
                cov_c = cov * scale.unsqueeze(-1) * scale.unsqueeze(-2)
                L_new, chol_info = torch.linalg.cholesky_ex(cov_c)
                good = ok & (chol_info == 0) & torch.isfinite(L_new).all(dim=-1).all(dim=-1)
                # from GRAMIS, keep the previous covariance wherever the inverse-Hessian is undefined.
                cand = torch.where(good[..., None, None], L_new, scale_tril)

                clamped = Proposal(
                    mu=centres,
                    log_diag=torch.diagonal(cand, dim1=-2, dim2=-1).clamp(min=1e-30).log(),
                    low=cand - torch.diag_embed(torch.diagonal(cand, dim1=-2, dim2=-1)),
                    logits=log_component_logits,
                ).clamp_scale(min_variance, max_variance)
                scale_tril = clamped.scale_tril()
                variances = torch.diagonal(
                    scale_tril @ scale_tril.transpose(-1, -2), dim1=-2, dim2=-1
                )
            else:
                # from GRAMIS, keep the previous covariance wherever the inverse-Hessian is undefined.
                variances = torch.where(ok.unsqueeze(-1), new_var, variances)

        if not adapt_covariance:
            carried_grad = None # no Step 3 ran, so nothing was computed at the new means

        # sample and weight
        x, log_pi, log_w = draw_and_weight(centres, variances, scale_tril)
        all_samples_list.append(x)
        all_log_pi_list.append(log_pi)
        all_log_w_list.append(log_w)

    # One extra terminal block, matching the other IS baselines.
    x, log_pi, log_w = draw_and_weight(centres, variances, scale_tril)
    all_samples_list.append(x)
    all_log_pi_list.append(log_pi)
    all_log_w_list.append(log_w)

    return {
        "centres": centres,
        "variances": variances,
        "log_component_logits": log_component_logits,
        "final_proposal": (as_proposal(centres, scale_tril) if full_cov else diagonal(centres, variances, log_component_logits)),
        "per_step": _per_step_trace(centres_hist, variances_hist, logits_hist, all_samples_list, all_log_w_list, scale_tril_hist if full_cov else None),
        "all_samples": torch.cat(all_samples_list, dim=1),
        "all_log_w": torch.cat(all_log_w_list, dim=1),
        "final_samples": all_samples_list[-1],
        "final_log_w": all_log_w_list[-1],
    }


class _BaseIS:
    """Base class for IS baselines."""

    name: str

    @classmethod
    def from_cfg(cls, cfg: DictConfig, device: torch.device, dtype: torch.dtype):
        return cls(cfg)

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

class M_PMC(_BaseIS):
    name = "m_pmc"

    def run(self, target, init_centres, init_variances, n_steps, k_particles):
        return run_m_pmc(target, init_centres, init_variances, n_steps, k_particles)


class DM_PMC_resampling(_BaseIS):
    name = "dm_pmc"

    def run(self, target, init_centres, init_variances, n_steps, k_particles):
        return run_dm_pmc_resampling(target, init_centres, init_variances, n_steps, k_particles)


class GAPIS(_BaseIS):
    name = "gapis"

    def run(self, target, init_centres, init_variances, n_steps, k_particles):
        return run_gapis(
            target,
            init_centres,
            init_variances,
            n_steps,
            k_particles,
            lam=float(self.cfg.lam),
            repulse_nu=float(self.cfg.get("repulse_nu", 0.05)),
            repulse_xi=float(self.cfg.get("repulse_xi", 1.04)),
            adapt_covariance=bool(self.cfg.get("adapt_covariance", True)),
            full_cov=bool(self.cfg.get("full_cov", True)),
        )


