"""The factory for the learnt sampler.
"""

from typing import Any

import torch


def build_runner(cfg: Any, device=None, dtype: torch.dtype = torch.float32, **kwargs):
    """Load a checkpoint named by the config and turn it into a runner.

    Args:
        cfg: sampler config, must carry `checkpoint`.
        device: Torch device.
        dtype: Torch dtype.
        **kwargs: Ignored.

    Returns:
        A callable `(target, init_centres, init_variances, n_steps, k_particles) -> dict`.
    """
    from amortised_adaptive_is.learnt.train import load, make_learnt_runner

    path = cfg.get("checkpoint") if hasattr(cfg, "get") else getattr(cfg, "checkpoint", None)
    if not path:
        raise ValueError(
            "the learnt sampler needs `sampler.checkpoint`"
        )
    net, train_cfg, _ = load(path, device=device or torch.device("cpu"), dtype=dtype)
    return make_learnt_runner(net, train_cfg)
