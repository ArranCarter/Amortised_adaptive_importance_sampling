"""Gaussian-mixture proposal utilities.
"""

import numpy as np
import torch

from amortised_adaptive_is.core.counters import get_active_density_eval_counter

EPS = 1e-8


def batched_gaussian_mixture_log_density(
    x: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    component_logits: torch.Tensor,
    count_as_proposal: bool = True,
) -> torch.Tensor:
    """Evaluate log-density of a batched diagonal-covariance GMM.

    Args:
        x: Samples, shape (B, S, d).
        means: Component means, shape (B, N, d).
        variances: Component variances, shape (B, N, d).
        component_logits: Component logits (unnormalised), shape (B, N).

    Returns:
        Log density for each sample.
    """
    if count_as_proposal:
        counter = get_active_density_eval_counter()
        if counter is not None:
            counter.add_proposal(x)

    _B, _S, d = x.shape

    x_e = x.unsqueeze(1) # shape (B, 1, S, d)
    means_e = means.unsqueeze(2) # shape (B, N, 1, d)
    vars_e = variances.unsqueeze(2) # shape (B, N, 1, d)

    log_probs = (
        -0.5 * (((x_e - means_e) ** 2) / vars_e + vars_e.log()).sum(dim=-1)
        + component_logits.unsqueeze(2)
    ) # shape (B, N, S)

    log_density = torch.logsumexp(log_probs, dim=1) - 0.5 * d * float(np.log(2.0 * np.pi))
    return log_density


def draw_initial_centres(
    batch_size: int,
    n_components: int,
    spatial_dim: int,
    centre_scale: float,
    centre_sample_type: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Draw an initial set of proposal component means.

    Args:
        batch_size: Number of independent target instances.
        n_components: Mixture component count `N`.
        spatial_dim: Dimensionality `d`.
        centre_scale: Scale of the draw.
        centre_sample_type: `"gaussian"` for `randn * centre_scale`, or `"uniform"` for `U(-centre_scale, centre_scale)`.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        Initial centres, shape (batch_size, n_components, spatial_dim).
    """
    shape = (batch_size, n_components, spatial_dim)
    if centre_sample_type == "uniform":
        # Uniform in [-centre_scale, centre_scale]
        return (torch.rand(shape, device=device, dtype=dtype) * 2 - 1) * centre_scale
    if centre_sample_type == "gaussian":
        return torch.randn(shape, device=device, dtype=dtype) * centre_scale
    raise ValueError(f"Unsupported centre_sample_type={centre_sample_type!r}, expected 'gaussian' or 'uniform'")


def draw_k_per_component(centres: torch.Tensor, variances: torch.Tensor, k_particles: int) -> torch.Tensor:
    """Draw K particles per component from a diagonal Gaussian mixture.

    Args:
        centres: Component means, shape (B, N, d).
        variances: Component variances, shape (B, N, d).
        k_particles: Number of particles per component.

    Returns:
        Batched samples with stratified component allocation.
    """
    B, N, d = centres.shape
    device = centres.device
    idx = torch.arange(N, device=device).repeat_interleave(k_particles)
    eps = torch.randn((B, N * k_particles, d), device=device, dtype=centres.dtype)
    return centres[:, idx] + variances[:, idx].sqrt() * eps

def draw_iid_from_mixture(
    centres: torch.Tensor,
    variances: torch.Tensor,
    log_component_logits: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    """Draw i.i.d. samples from the mixture, selecting components by their weights.

    Args:
        centres: Component means, shape (B, N, d).
        variances: Component variances, shape (B, N, d).
        log_component_logits: Component logits, shape (B, N). Normalised internally.
        n_samples: Number of samples per batch element.

    Returns:
        Samples, shape (B, n_samples, d).
    """
    B = centres.shape[0]
    probs = torch.softmax(log_component_logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=n_samples, replacement=True)
    ar = torch.arange(B, device=centres.device).unsqueeze(-1)
    means = centres[ar, idx]
    stds = variances[ar, idx].sqrt()
    return means + stds * torch.randn_like(means)


def multinomial_resample(x: torch.Tensor, log_w: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Multinomial resampling with replacement.

    Args:
        x: Samples, shape (B, S, d).
        log_w: Unnormalised log weights, shape (B, S).
        n_samples: Number of resampled points.

    Returns:
        Resampled tensor.
    """
    B = x.shape[0]
    w = torch.softmax(log_w, dim=-1)
    idx = torch.multinomial(w, num_samples=n_samples, replacement=True)
    ar = torch.arange(B, device=x.device).unsqueeze(-1)
    return x[ar, idx]


def soft_em_update(
    x: torch.Tensor,
    log_w: torch.Tensor,
    centres: torch.Tensor,
    variances: torch.Tensor,
    log_component_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform a responsibility-weighted soft EM update.

    Args:
        x: Samples, shape (B, S, d).
        log_w: Unnormalised log weights, shape (B, S).
        centres: Current component means, shape (B, N, d).
        variances: Current component variances, shape (B, N, d).
        log_component_logits: Current component logits, shape (B, N).

    Returns:
        Tuple of (centres_new, variances_new, log_logits_new).
    """

    w = torch.softmax(log_w, dim=-1) # shape (B, S)

    x_exp = x.unsqueeze(2) # shape (B, S, 1, d)
    mu_exp = centres.unsqueeze(1) # shape (B, 1, N, d)
    var_exp = variances.unsqueeze(1) # shape (B, 1, N, d)

    # calculate posterior probabilities (component responsibilities)
    log_g = -0.5 * (((x_exp - mu_exp) ** 2) / var_exp + var_exp.log()).sum(dim=-1)
    log_pi = torch.log_softmax(log_component_logits, dim=-1)
    resp = torch.softmax(log_g + log_pi.unsqueeze(1), dim=-1) # shape (B, S, N)

    a = w.unsqueeze(-1) * resp
    a_n = a.sum(dim=1).clamp(min=EPS)

    centres_new = (a.unsqueeze(-1) * x.unsqueeze(2)).sum(dim=1) / a_n.unsqueeze(-1)
    diff = x.unsqueeze(2) - centres_new.unsqueeze(1)
    variances_new = (a.unsqueeze(-1) * diff**2).sum(dim=1) / a_n.unsqueeze(-1)
    variances_new = variances_new.clamp(min=1e-4)

    log_logits_new = torch.log(a_n / a_n.sum(dim=-1, keepdim=True).clamp(min=EPS))
    return centres_new, variances_new, log_logits_new
