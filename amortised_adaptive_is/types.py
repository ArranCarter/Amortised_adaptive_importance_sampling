"""Core types and result containers.
"""

from dataclasses import dataclass
from typing import Protocol

import torch


class Target(Protocol):
    name: str
    d: int
    B: int

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute unnormalised log density.

        Args:
            x: (B, S, d)

        Returns:
            (B, S)
        """

    def sample(self, n_samples: int) -> torch.Tensor:
        """Draw reference samples.

        Returns:
            (B, n_samples, d)
        """

    def quick_scatter(self, result: dict, max_points: int = 2000):
        """Optional diagnostic plot for logging."""


@dataclass(frozen=True)
class SamplerResult:
    centres: torch.Tensor # shape (B, N, d)
    variances: torch.Tensor # shape (B, N, d)
    log_component_logits: torch.Tensor # shape (B, N)

    all_samples: torch.Tensor # shape (B, S_total, d)
    all_log_w: torch.Tensor | None # shape (B, S_total)

    final_samples: torch.Tensor # shape (B, S_final, d)
    final_log_w: torch.Tensor | None # shape (B, S_final)

    accept_rate: float | None = None


class Algorithm(Protocol):
    name: str

    def run(
        self,
        target: Target,
        init_centres: torch.Tensor,
        init_variances: torch.Tensor,
        n_steps: int,
        k_particles: int,
    ) -> dict:
        ...
