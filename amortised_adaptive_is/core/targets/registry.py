"""Target registry and factory.
"""

from collections.abc import Callable

import torch
from omegaconf import DictConfig

from .banana_nd import sample_banana_nd_targets
from .blr import sample_blr_targets
from .funnel import sample_funnel_nd_targets
from .gmm import sample_gmm_targets, sample_gmm_targets_4_corners
from .hierarchical import sample_hierarchical_targets

TargetFactory = Callable[..., object]


TARGET_REGISTRY: dict[str, TargetFactory] = {
    "gmm": sample_gmm_targets,
    "gmm_4_corners": sample_gmm_targets_4_corners,
    "funnel_nd": sample_funnel_nd_targets,
    "banana_nd": sample_banana_nd_targets,
    "blr": sample_blr_targets,
    "hierarchical": sample_hierarchical_targets,
}


def build_target(
    cfg: DictConfig,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    n_components: int,
    spatial_dim: int,
):
    """Construct a batched target.

    Args:
        cfg: Target config containing a `name` field.
        batch_size: Number of target instances.
        device: Torch device.
        dtype: Torch dtype.
        n_components: Default mixture component count.
        spatial_dim: Default spatial dimension.

    Returns:
        Batched target object.
    """
    name = str(cfg.name)
    if name not in TARGET_REGISTRY:
        raise KeyError(f"Unknown target {name!r}. Available: {sorted(TARGET_REGISTRY)}")

    if name in ["gmm", "gmm_4_corners"]:
        return TARGET_REGISTRY[name](
            batch_size=batch_size,
            n_components=int(cfg.get("n_components", n_components)),
            centre_range=float(cfg.centre_range),
            target_variance=float(cfg.target_variance),
            min_sep=float(cfg.min_sep),
            device=device,
            dtype=dtype,
            spatial_dim=int(cfg.get("spatial_dim", spatial_dim)),
        )

    if name == "blr":
        return TARGET_REGISTRY[name](
            batch_size=batch_size,
            spatial_dim=int(cfg.get("spatial_dim", spatial_dim)),
            device=device,
            dtype=dtype,
            n_range=tuple(cfg.n_range),
            cond_log10_range=tuple(cfg.cond_log10_range),
            beta_scale=float(cfg.beta_scale),
            sigma_range=tuple(cfg.sigma_range),
            prior_precision=float(cfg.prior_precision),
            prior_a=float(cfg.prior_a),
            prior_b=float(cfg.prior_b),
            pin_seed=(None if cfg.get("pin_seed", None) is None else int(cfg.pin_seed)),
        )

    if name == "hierarchical":
        return TARGET_REGISTRY[name](
            batch_size=batch_size,
            spatial_dim=int(cfg.get("spatial_dim", spatial_dim)),
            device=device,
            dtype=dtype,
            sigma_range=tuple(cfg.sigma_range),
            tau_true_range=tuple(cfg.tau_true_range),
            mu_true_scale=float(cfg.mu_true_scale),
            s_mu=float(cfg.s_mu),
            s_tau=float(cfg.s_tau),
            pin_seed=(None if cfg.get("pin_seed", None) is None else int(cfg.pin_seed)),
        )

    if name == "banana_nd":
        return TARGET_REGISTRY[name](
            batch_size=batch_size,
            spatial_dim=int(cfg.get("spatial_dim", spatial_dim)),
            device=device,
            dtype=dtype,
            b_range=tuple(cfg.b_range),
            sigma1_range=tuple(cfg.sigma1_range),
            sigma2_range=tuple(cfg.sigma2_range),
            pin_seed=(None if cfg.get("pin_seed", None) is None else int(cfg.pin_seed)),
        )

    if name == "funnel_nd":
        return TARGET_REGISTRY[name](
            batch_size=batch_size,
            spatial_dim=int(cfg.get("spatial_dim", spatial_dim)),
            device=device,
            dtype=dtype,
            sigma_v_range=tuple(cfg.sigma_v_range),
        )

