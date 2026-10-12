"""Algorithm registry and construction helpers."""

import torch
from omegaconf import DictConfig

from amortised_adaptive_is.core.counters import (
    CountingTarget,
    DensityEvalCounter,
    density_eval_counter,
    get_active_density_eval_counter,
)

from .is_methods import GAPIS, M_PMC, DM_PMC_resampling
from .oracle import TrueSampler

ALGORITHM_REGISTRY = {
    "m_pmc": M_PMC,
    "dm_pmc_resampling": DM_PMC_resampling,
    "gapis": GAPIS,
    "true_sampler": TrueSampler,
}


class _CountingAlgorithmWrapper:
    """Wrap an algorithm to ensure density counting is always active.
    """

    def __init__(self, inner):
        self._inner = inner
        self.last_infer_density_evals: dict[str, int] | None = None

    def run(self, target, *args, **kwargs):
        active = get_active_density_eval_counter()

        # Choose a counter that keeps target/proposal counting aligned.
        if active is not None:
            counter = active
        elif isinstance(target, CountingTarget):
            counter = target.counter
        else:
            counter = DensityEvalCounter()

        # Ensure the target contributes to the same counter.
        if not isinstance(target, CountingTarget):
            target = CountingTarget(target, counter)

        with density_eval_counter(counter):
            result = self._inner.run(target=target, *args, **kwargs)

        self.last_infer_density_evals = counter.to_dict()
        return result

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def build_algorithm(cfg: DictConfig, device: torch.device, dtype: torch.dtype):
    """Instantiate an algorithm from a Hydra config.

    Args:
        cfg: Algorithm configuration containing a `name` field.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        Instantiated algorithm object implementing `run(...)`.
    """

    name = str(cfg.name)
    if name not in ALGORITHM_REGISTRY:
        raise KeyError(f"Unknown algorithm {name!r}. Available: {sorted(ALGORITHM_REGISTRY)}")

    cls = ALGORITHM_REGISTRY[name]
    algo = cls.from_cfg(cfg, device=device, dtype=dtype)
    return _CountingAlgorithmWrapper(algo)


def build_runner(cfg, device=None, dtype=torch.float32, **kwargs):
    """Builds a runner from a hydra config.

    Args:
        cfg: sampler config
        device: Torch device.
        dtype: Torch dtype.
        **kwargs: Ignored, to pass shared options.

    Returns:
        A callable `(target, init_centres, init_variances, n_steps, k_particles) -> dict`.
    """
    return build_algorithm(cfg, device=device or torch.device("cpu"), dtype=dtype).run
