"""Building the classical baselines as runners.
"""

from collections.abc import Callable, Iterable
from typing import Any

import torch
from omegaconf import OmegaConf

from amortised_adaptive_is.baselines import build_algorithm
from amortised_adaptive_is.core.paths import sampler_config
from amortised_adaptive_is.core.registry import require_complete
from amortised_adaptive_is.core.runner import score_runner
from amortised_adaptive_is.core.specs import TaskSpec

BASELINES: tuple[str, ...] = (
    "dm_pmc_resampling",
    "m_pmc",
    "gapis",
)


def _resolved(name: str, overrides: dict[str, dict] | None):
    """One sampler's config, with any caller overrides applied and mandatory values checked.

    Args:
        name: Sampler name.
        overrides: Per-sampler values to set, e.g. `{"gapis": {"lam": 0.001}}`.

    Returns:
        The resolved config.
    """
    cfg = OmegaConf.load(sampler_config(name))
    cfg.name = name
    for key, value in ((overrides or {}).get(name) or {}).items():
        cfg[key] = value
    require_complete(cfg, name)
    return cfg


def baseline_runners(
    names: Iterable[str] = BASELINES,
    include_oracle: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    spec: TaskSpec | None = None,
    sampler_overrides: dict[str, dict] | None = None,
) -> dict[str, Callable[..., dict]]:
    """Build runners for the registered baselines.

    Args:
        names: Registry keys to include.
        include_oracle: Whether to add `true_sampler`.
        device: Torch device.
        dtype: Torch dtype.
        spec: Unused.
        sampler_overrides: Per-sampler config values, e.g. `{"gapis": {"lam": 0.001}}`.

    Returns:
        Mapping from algorithm name to its `run` method.
    """
    device = device or torch.device("cpu")
    keys = list(names) + (["true_sampler"] if include_oracle else [])
    return {name: build_algorithm(_resolved(name, sampler_overrides),
                                  device=device, dtype=dtype).run
            for name in keys}


def score_baseline(
    name: str,
    spec: TaskSpec,
    seed: int = 0,
    **kwargs: Any,
) -> dict[str, float]:
    """Run and score one registered baseline algorithm.

    Args:
        name: Registry key, e.g. `dm_pmc`.
        spec: Task specification.
        seed: RNG seed.
        **kwargs: containing sampler_overrides and other arguments forwarded to `score_runner`.

    Returns:
        Flat dict of scalar metrics.
    """
    device = kwargs.get("device") or torch.device("cpu")
    dtype = kwargs.get("dtype", torch.float32)
    overrides = kwargs.pop("sampler_overrides", None)
    algo = build_algorithm(_resolved(name, overrides), device=device, dtype=dtype)
    return score_runner(algo.run, spec, seed=seed, **kwargs)
