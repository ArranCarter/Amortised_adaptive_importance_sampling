"""Seeding utils, from https://github.com/mmacosha/offpolicy-discrete-diffusion-samplers-and-bridges
"""

import contextlib
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set RNG seeds across torch/numpy/python for reproducibility.

    Args:
        seed: Random seed.

    Returns:
        None.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def temp_seed(seed: int | None):
    """Temporarily set RNG seeds within a context.

    Args:
        seed: Random seed. If None, acts as a no-op context manager.

    Returns:
        Context manager.
    """
    if seed is None:
        return contextlib.nullcontext()
    return _temp_seed(seed)


@contextlib.contextmanager
def _temp_seed(seed: int):
    """Context manager implementing `temp_seed` when seed is not None."""
    random_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    torch_cuda_states = torch.cuda.get_rng_state_all()
    set_seed(seed)

    try:
        yield
    finally:
        # Restore the previous RNG states.
        random.setstate(random_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        torch.cuda.set_rng_state_all(torch_cuda_states)
