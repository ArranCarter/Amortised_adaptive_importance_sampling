"""Importance sampling evaluation metrics."""

import numpy as np
import torch

EPS = 1e-8


def compute_ess(log_w: torch.Tensor) -> torch.Tensor:
    """Compute normalised importance-sampling ESS.

    Args:
        log_w: Unnormalised log weights, shape (B, S).

    Returns:
        ESS normalised by the number of samples.
    """
    S = log_w.shape[1]
    w = torch.softmax(log_w, dim=-1)
    return 1.0 / ((w**2).sum(dim=-1) * S + EPS)


def compute_log_z(log_w: torch.Tensor) -> torch.Tensor:
    """Compute the log normaliser estimate from log weights.

    Args:
        log_w: Unnormalised log weights, shape (B, S).

    Returns:
        Estimate of log Z.
    """
    return torch.logsumexp(log_w, dim=-1) - float(np.log(log_w.shape[1]))
