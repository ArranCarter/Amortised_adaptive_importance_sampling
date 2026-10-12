import contextlib
import contextvars
from dataclasses import dataclass

import torch


def _count_points(x: torch.Tensor) -> int:
    """Count number of points in a tensor interpreted as (..., d).
    """

    if not torch.is_tensor(x):
        return 0
    if x.ndim <= 1:
        return 1
    n = 1
    for s in x.shape[:-1]:
        n *= int(s)
    return int(n)


@dataclass
class DensityEvalCounter:
    """Accumulate target eval counts, separating evals, gradients and Hessians.
    """

    target_log_prob_calls: int = 0
    target_log_prob_evals: int = 0

    target_grad_calls: int = 0
    target_grad_evals: int = 0

    target_hess_calls: int = 0
    target_hess_evals: int = 0

    proposal_log_prob_calls: int = 0
    proposal_log_prob_evals: int = 0

    def add_target(self, x: torch.Tensor) -> None:
        self.target_log_prob_calls += 1
        self.target_log_prob_evals += _count_points(x)

    def add_target_grad(self, x: torch.Tensor) -> None:
        """Charge one gradient evaluation per point in `x`."""
        self.target_grad_calls += 1
        self.target_grad_evals += _count_points(x)

    def add_target_hessian(self, x: torch.Tensor) -> None:
        """Charge `d` gradient-equivalents per point."""
        d = int(x.shape[-1]) if torch.is_tensor(x) and x.ndim >= 1 else 1
        self.target_hess_calls += 1
        self.target_hess_evals += _count_points(x) * d

    @property
    def target_cost(self) -> int:
        """Total cost in density-evaluation units."""
        return int(
            self.target_log_prob_evals + self.target_grad_evals + self.target_hess_evals
        )

    def add_proposal(self, x: torch.Tensor) -> None:
        self.proposal_log_prob_calls += 1
        self.proposal_log_prob_evals += _count_points(x)

    def to_dict(self) -> dict[str, int]:
        return {
            "target_log_prob_calls": int(self.target_log_prob_calls),
            "target_log_prob_evals": int(self.target_log_prob_evals),
            "target_grad_calls": int(self.target_grad_calls),
            "target_grad_evals": int(self.target_grad_evals),
            "target_hess_calls": int(self.target_hess_calls),
            "target_hess_evals": int(self.target_hess_evals),
            "target_cost": int(self.target_cost),
            "proposal_log_prob_calls": int(self.proposal_log_prob_calls),
            "proposal_log_prob_evals": int(self.proposal_log_prob_evals),
        }


_ACTIVE_COUNTER: contextvars.ContextVar[DensityEvalCounter | None] = contextvars.ContextVar(
    "ACTIVE_DENSITY_EVAL_COUNTER",
    default=None,
)


def get_active_density_eval_counter() -> DensityEvalCounter | None:
    return _ACTIVE_COUNTER.get()


@contextlib.contextmanager
def ensure_density_eval_counter(counter: DensityEvalCounter | None = None):
    """Ensure a density-eval counter is active within the context.
    """
    active = get_active_density_eval_counter()
    if active is not None:
        yield active
        return

    if counter is None:
        counter = DensityEvalCounter()

    with density_eval_counter(counter):
        yield counter


@contextlib.contextmanager
def density_eval_counter(counter: DensityEvalCounter | None):
    """Context manager that activates a counter for proposal log-density calls."""

    token = _ACTIVE_COUNTER.set(counter)
    try:
        yield counter
    finally:
        _ACTIVE_COUNTER.reset(token)


class CountingTarget:
    """Target wrapper that counts `log_prob` evaluations."""

    def __init__(self, target, counter: DensityEvalCounter):
        self._target = target
        self._counter = counter

    @property
    def counter(self) -> DensityEvalCounter:
        return self._counter

    def log_prob(self, x: torch.Tensor):
        self._counter.add_target(x)
        return self._target.log_prob(x)

    def sample(self, *args, **kwargs):
        return self._target.sample(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._target, name)
