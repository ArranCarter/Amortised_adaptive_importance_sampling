"""Ground-truth sampler algorithm.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig


class TrueSampler:
    """Draw ground-truth samples from the target."""

    name = "true_sampler"

    @classmethod
    def from_cfg(cls, cfg: DictConfig, device: torch.device, dtype: torch.dtype):
        return cls(cfg=cfg, device=device, dtype=dtype)

    def __init__(self, cfg: DictConfig, device: torch.device, dtype: torch.dtype):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def run(
        self,
        target,
        init_centres: torch.Tensor,
        init_variances: torch.Tensor,
        n_steps: int,
        k_particles: int,
    ) -> dict:
        """Return a sampler-result dict compatible with evaluation/plotting.
        """

        B, n_components, d = init_centres.shape
        samples_per_step = int(n_components) * int(k_particles)
        n_steps = int(n_steps)
        samples_per_step = max(1, samples_per_step)
        n_steps = max(1, n_steps)

        # Draw i.i.d. samples from the target.
        all_samples_steps = [target.sample(samples_per_step) for _ in range(n_steps + 1)]

        # target.sample should return shape (B, S, d).
        all_samples = torch.cat(all_samples_steps, dim=1)
        final_samples = all_samples_steps[-1]

        all_log_w = torch.zeros((B, all_samples.shape[1]), device=all_samples.device, dtype=all_samples.dtype)
        final_log_w = torch.zeros((B, final_samples.shape[1]), device=final_samples.device, dtype=final_samples.dtype)

        # Keep fields expected by plotting/evaluation.
        log_component_logits = torch.log(
            torch.ones((B, n_components), device=init_centres.device, dtype=init_centres.dtype) / float(n_components)
        )

        return {
            # proposal parameters passed so that plotting/evaluation code works.
            "centres": init_centres,
            "variances": init_variances,
            "log_component_logits": log_component_logits,
            "all_samples": all_samples,
            "all_log_w": all_log_w,
            "final_samples": final_samples,
            "final_log_w": final_log_w,
            "proposal_is_target": True,
        }
