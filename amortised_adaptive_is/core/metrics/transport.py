"""Optimal-transport distances.
"""

import numpy as np
import torch

# Projections for the sliced distance.
SLICED_SEED = 0
SLICED_PROJECTIONS = 512


def wasserstein(x: torch.Tensor, y: torch.Tensor, p: int = 1) -> torch.Tensor:
    """Exact Wasserstein-p, solved by the network simplex.

    Args:
        x: Samples, shape (B, S, d).
        y: Samples, shape (B, S', d).
        p: 1 or 2.

    Returns:
        Per-instance distance, shape (B,).
    """
    import ot

    if p not in (1, 2):
        raise ValueError(f"p must be 1 or 2; got {p}")
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(f"expected (B, S, d); got x={tuple(x.shape)} y={tuple(y.shape)}")
    if x.shape[0] != y.shape[0] or x.shape[2] != y.shape[2]:
        raise ValueError(f"batch and dimension must agree; got x={tuple(x.shape)} y={tuple(y.shape)}")
    # POT uses numpy on the CPU.
    xn = x.detach().double().cpu().numpy()
    yn = y.detach().double().cpu().numpy()
    a = np.full(xn.shape[1], 1.0 / xn.shape[1])
    b = np.full(yn.shape[1], 1.0 / yn.shape[1])
    metric = "euclidean" if p == 1 else "sqeuclidean"
    vals = []
    for i in range(xn.shape[0]):
        c = float(ot.emd2(a, b, ot.dist(xn[i], yn[i], metric=metric)))
        vals.append(c if p == 1 else np.sqrt(max(c, 0.0)))
    return torch.tensor(vals, dtype=x.dtype, device=x.device)


def sliced_wasserstein(x: torch.Tensor, y: torch.Tensor, p: int = 1,
                       n_projections: int = SLICED_PROJECTIONS,
                       seed: int = SLICED_SEED) -> torch.Tensor:
    """Sliced Wasserstein-p, batched.

    Args:
        x: Samples, shape (B, S, d).
        y: Samples, shape (B, S, d). Must match `x` in sample count.
        p: Order.
        n_projections: Directions averaged over.
        seed: Fixed by default, so every sampler is compared through the same directions.

    Returns:
        Per-instance distance, shape (B,).
    """
    import ot

    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"sliced_wasserstein needs equal sample counts, got {x.shape[1]} and {y.shape[1]}")
    xd, yd = x.detach().double(), y.detach().double()
    proj = ot.sliced.get_random_projections(x.shape[2], n_projections, seed) # (d, n_proj)
    pt = torch.as_tensor(np.asarray(proj), dtype=torch.float64, device=xd.device)
    xs = torch.sort(xd @ pt, dim=1).values # (B, S, n_proj)
    ys = torch.sort(yd @ pt, dim=1).values
    gaps = (xs - ys).abs()
    # SW_p = (mean over projections of W_p^p)^(1/p) -- the p-th powers are averaged, then the root is taken
    per_proj_pow = gaps.pow(p).mean(dim=1) # W_p^p (B, n_proj)
    return per_proj_pow.mean(dim=1).pow(1.0 / p).to(dtype=x.dtype, device=x.device)
