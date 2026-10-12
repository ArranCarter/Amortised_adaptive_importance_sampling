"""From the sampler name, get the module that implements it.
"""

import importlib
from typing import Any

from omegaconf import OmegaConf

SAMPLERS: dict[str, tuple[str, str]] = {
    "learnt": ("amortised_adaptive_is.learnt.registry", "build_runner"),
    "dm_pmc_resampling": ("amortised_adaptive_is.baselines.registry", "build_runner"),
    "m_pmc": ("amortised_adaptive_is.baselines.registry", "build_runner"),
    "gapis": ("amortised_adaptive_is.baselines.registry", "build_runner"),
    "true_sampler": ("amortised_adaptive_is.baselines.registry", "build_runner"),
}

def require_complete(cfg: Any, name: str) -> None:
    """Check for configs that have a missing value.

    Args:
        cfg: sampler config.
        name: Sampler name, for the message.
    """
    missing = sorted(k for k in cfg if OmegaConf.is_missing(cfg, k))
    if missing:
        raise ValueError(
            f"sampler {name!r} has no value for {', '.join(missing)}. provide (sampler.{missing[0]}=<value>)."
        )

def get_factory(name: str):
    """The factory for one sampler.

    Args:
        name: Sampler name, as it appears in `configs/sampler/`.

    Returns:
        A callable `(cfg, **kwargs) -> runner`.
    """
    if name not in SAMPLERS:
        raise KeyError(f"unknown sampler {name!r}, choices are: {sorted(SAMPLERS)}")
    module_path, attr = SAMPLERS[name]
    return getattr(importlib.import_module(module_path), attr)


def build(name: str, cfg: Any, **kwargs):
    """Get the factory for a sampler and build its runner."""
    require_complete(cfg, name)
    return get_factory(name)(cfg, **kwargs)
