"""Training objectives.
"""

import math

import torch

from amortised_adaptive_is.core.proposal import log_density_components
from amortised_adaptive_is.learnt.rollout import EMStepRecord

LOSSES = ("rkl", "fkl", "xent", "chisq")

# Objectives needing draws from the target rather than from the proposal.
NEEDS_TRUE_SAMPLES = ("xent",)

MAX_LOG_RATIO = 30.0


# Largest magnitude a per-particle log-ratio may carry into the compression.
MAX_RATIO_INPUT = 1e4


def _finite_ratio(u: torch.Tensor) -> torch.Tensor:
    """A log-ratio with infinities bounded.
    """
    return torch.clamp(u, min=-MAX_RATIO_INPUT, max=MAX_RATIO_INPUT)


def _soft_compress(u: torch.Tensor) -> torch.Tensor:
    """Compression on log-ratio
    """
    return MAX_LOG_RATIO * torch.asinh(u / MAX_LOG_RATIO)


def _rkl(rec: EMStepRecord) -> torch.Tensor:
    """Reverse KL `E_q[log q - log pi]` per instance, shape (B,).
    """
    r = _finite_ratio(rec.log_q - rec.log_pi)
    return _soft_compress(r).mean(dim=1)


def _fkl(rec: EMStepRecord) -> torch.Tensor:
    """Self-normalised IS estimate of the forward KL `E_pi[log pi - log q]`, shape (B,).
    """
    log_q = log_density_components(rec.prop, rec.samples.detach())[0]
    log_w = _finite_ratio(rec.log_pi.detach() - log_q)
    w_bar = torch.softmax(_finite_ratio(rec.log_w.detach()), dim=1)
    return (w_bar * _soft_compress(log_w)).sum(dim=1)


def _xent(rec: EMStepRecord, true_samples: torch.Tensor,
          true_log_pi: torch.Tensor) -> torch.Tensor:
    """Forward KL at oracle draws, `E_pi[log pi - log q]`, shape (B,).

    Args:
        rec: The block, for its proposal.
        true_samples: Draws from the target, shape (B, M, d).
        true_log_pi: `log pi` at `true_samples`, shape (B, M).

    Returns:
        Per-instance loss, shape (B,).
    """
    log_q = log_density_components(rec.prop, true_samples)[0]
    return _soft_compress(_finite_ratio(true_log_pi - log_q)).mean(dim=1)


def _chisq(rec: EMStepRecord) -> torch.Tensor:
    """Negative log ESS, shape (B,).
    """
    lw = _finite_ratio(rec.log_w)
    log_ess = 2 * torch.logsumexp(lw, dim=1) - torch.logsumexp(2 * lw, dim=1)
    return math.log(lw.shape[1]) - log_ess


def step_weights(n_blocks: int, mode: str, device, dtype) -> torch.Tensor:
    """Normalised weights over the trajectory's blocks, shape (n_blocks,).

    Args:
        n_blocks: Number of sample blocks, `T + 1`.
        mode: `uniform`, `final` (terminal block only), or `linear` (rising with the step index).
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        Weights summing to 1.
    """
    if mode == "uniform":
        w = torch.ones(n_blocks, device=device, dtype=dtype)
    elif mode == "final":
        w = torch.zeros(n_blocks, device=device, dtype=dtype)
        w[-1] = 1.0
    elif mode == "linear":
        w = torch.arange(n_blocks, device=device, dtype=dtype) + 1.0
    else:
        raise ValueError(f"Unknown step weighting {mode!r}")
    return w / w.sum()


def block_loss(rec: EMStepRecord, kind: str, true_samples: torch.Tensor | None = None,
               true_log_pi: torch.Tensor | None = None) -> torch.Tensor:
    """Evaluate one objective on one sample block.

    Args:
        rec: The block, as produced by the rollout.
        kind: One of `LOSSES`.
        true_samples: Draws from the target, shape (B, M, d).

    Returns:
        Per-instance loss, shape (B,).
    """
    if kind in NEEDS_TRUE_SAMPLES and true_samples is None:
        raise ValueError(f"loss {kind!r} needs true_samples")
    if kind == "rkl":
        return _rkl(rec)
    if kind == "fkl":
        return _fkl(rec)
    if kind == "xent":
        if true_log_pi is None:
            true_log_pi = rec.target.log_prob(true_samples).detach()
        return _xent(rec, true_samples, true_log_pi)
    if kind == "chisq":
        return _chisq(rec)
    raise ValueError(f"Unknown loss {kind!r}. Available: {LOSSES}")


def trajectory_loss(
    records: list[EMStepRecord],
    kind: str,
    weighting: str = "uniform",
    true_samples: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine per-block losses into a scalar training objective.

    Args:
        records: Blocks from the rollout.
        kind: Objective name, one of `LOSSES`.
        weighting: Step weighting mode.
        true_samples: Target draws, if required.

    Returns:
        Tuple of (scalar loss, final-block loss per instance).
    """

    true_log_pi = None
    if kind == "xent":
        true_log_pi = records[0].target.log_prob(true_samples).detach()
    per_block = torch.stack([block_loss(r, kind, true_samples, true_log_pi) for r in records],
                            dim=0) # (T+1, B)
    w = step_weights(len(records), weighting, per_block.device, per_block.dtype)
    return (w.unsqueeze(-1) * per_block).sum(dim=0).mean(), per_block[-1].detach()
