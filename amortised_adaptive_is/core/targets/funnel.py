"""Neal's funnel 2D target family.
"""

import numpy as np
import torch


# Clamp applied to the argument of exp to avoid overflow.
EXP_ARG_CLAMP = 60.0


def _clamped_exp(z: torch.Tensor) -> torch.Tensor:
    """`exp(z)` with clamped argument."""
    return torch.exp(z.clamp(min=-EXP_ARG_CLAMP, max=EXP_ARG_CLAMP))


class BatchedFunnelND:
    """Neal's funnel in `d` dimensions.

    Attributes:
        sigma_v: Scale of the first coordinate, shape (B,).
        d: Spatial dimension.
    """

    name = "FunnelND"

    def __init__(self, sigma_v: torch.Tensor, spatial_dim: int) -> None:
        if spatial_dim < 2:
            raise ValueError(f"FunnelND requires spatial_dim >= 2, got {spatial_dim}")
        self.sigma_v = sigma_v
        self.B = sigma_v.shape[0]
        self.d = spatial_dim

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute target log density.

        Args:
            x: Samples, shape (B, S, d).

        Returns:
            Log density, shape (B, S).
        """
        B, _S, d = x.shape
        if d != self.d:
            raise ValueError(f"Expected x.shape[-1]={self.d}, got {d}")

        sv = self.sigma_v.view(B, 1)
        v = x[..., 0]
        rest = x[..., 1:]

        log_pv = -0.5 * (v / sv) ** 2 - torch.log(sv) - 0.5 * float(np.log(2.0 * np.pi))
        
        log_rest = (
            -0.5 * rest.square().sum(dim=-1) * _clamped_exp(-v)
            - 0.5 * (self.d - 1) * v
            - 0.5 * (self.d - 1) * float(np.log(2.0 * np.pi))
        )
        return log_pv + log_rest

    def sample(self, n_samples: int) -> torch.Tensor:
        """Draw exact samples by ancestral sampling.

        Args:
            n_samples: Samples per batch element.

        Returns:
            Sample tensor, shape (B, n_samples, d).
        """
        device, dtype = self.sigma_v.device, self.sigma_v.dtype
        sv = self.sigma_v.view(self.B, 1)
        v = sv * torch.randn((self.B, n_samples), device=device, dtype=dtype)
        rest = torch.exp(0.5 * v).unsqueeze(-1) * torch.randn(
            (self.B, n_samples, self.d - 1), device=device, dtype=dtype
        )
        return torch.cat([v.unsqueeze(-1), rest], dim=-1)

    def quick_scatter(self, result: dict, max_points: int = 2000):
        """Scatter plot of the `(v, x_2)` marginal for logging.

        Args:
            result: Sampler output dictionary.
            max_points: Maximum points to plot.

        Returns:
            Matplotlib figure.
        """
        import matplotlib.pyplot as plt

        x = result["final_samples"][0].detach().cpu()
        if x.shape[0] > max_points:
            x = x[torch.randperm(x.shape[0])[:max_points]]
        gt = self.sample(min(max_points, 1000))[0].detach().cpu()

        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
        ax.scatter(gt[:, 0], gt[:, 1], s=6, alpha=0.12, c="#222222", label="target")
        ax.scatter(x[:, 0], x[:, 1], s=6, alpha=0.12, c="#4a90d9", label="samples")
        ax.set_title(f"{self.name} (v, x2)")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        return fig


def sample_funnel_nd_targets(
    batch_size: int,
    spatial_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    sigma_v_range: tuple = (1.5, 3.0),
) -> BatchedFunnelND:
    """Sample a batch of d-dimensional funnel targets.

    Args:
        batch_size: Number of target instances.
        spatial_dim: Dimensionality.
        device: Torch device.
        dtype: Torch dtype.
        sigma_v_range: Range for the funnel scale parameter.

    Returns:
        Batched d-dimensional funnel target.
    """
    sigma_v = (
        torch.rand(batch_size, device=device, dtype=dtype)
        * (sigma_v_range[1] - sigma_v_range[0])
        + sigma_v_range[0]
    )
    return BatchedFunnelND(sigma_v, spatial_dim=spatial_dim)
