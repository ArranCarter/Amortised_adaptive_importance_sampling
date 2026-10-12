"""A d-dimensional banana family, formed by products of two-dimensional bananas.
"""

import math

import torch


class BatchedBananaND:
    """Batched product of two-dimensional bananas, one per coordinate pair.

    Attributes:
        name: Display name.
        b: Curvature per block, shape (B, n_blocks).
        sigma1: First in-block scale, shape (B, n_blocks).
        sigma2: Second in-block scale, shape (B, n_blocks).
        theta: In-block rotation, shape (B, n_blocks).
    """

    name = "BananaND"

    def __init__(
        self,
        b: torch.Tensor,
        sigma1: torch.Tensor,
        sigma2: torch.Tensor,
        theta: torch.Tensor,
        spatial_dim: int,
    ) -> None:
        if spatial_dim < 2:
            raise ValueError(f"BananaND requires spatial_dim >= 2, got {spatial_dim}")
        self.b = b
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.theta = theta
        self.spatial_dim = spatial_dim
        self.B = b.shape[0]
        self.d = spatial_dim
        self.n_blocks = spatial_dim // 2
        self.has_tail = bool(spatial_dim % 2)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the target log density.

        Args:
            x: Samples, shape (B, S, d).

        Returns:
            Log density per sample, shape (B, S).
        """
        B, S, d = x.shape
        if d != self.d:
            raise ValueError(f"Expected x.shape[-1]={self.d}, got {d}")
        y = x

        # (B, S, n_blocks) views of the paired coordinates.
        pair = y[..., :2 * self.n_blocks].reshape(B, S, self.n_blocks, 2)
        u, v = pair[..., 0], pair[..., 1]

        cos_t = self.theta.cos().unsqueeze(1) # (B, 1, n_blocks)
        sin_t = self.theta.sin().unsqueeze(1)
        # Inverse within-block rotation.
        u_r = cos_t * u + sin_t * v
        v_r = -sin_t * u + cos_t * v

        s1 = self.sigma1.unsqueeze(1)
        s2 = self.sigma2.unsqueeze(1)
        bb = self.b.unsqueeze(1)
        z1 = u_r / s1
        z2 = (v_r - bb * u_r**2 + bb * s1**2) / s2

        # Per block: standard normal in (z1, z2) less the log Jacobian log(s1 s2).
        log_p = (
            -0.5 * (z1**2 + z2**2) - math.log(2.0 * math.pi) - torch.log(s1 * s2)
        ).sum(dim=-1) # (B, S)

        if self.has_tail:
            t = y[..., -1]
            log_p = log_p - 0.5 * t**2 - 0.5 * math.log(2.0 * math.pi)
        return log_p

    def sample(self, n_samples: int) -> torch.Tensor:
        """Draw exact samples.

        Args:
            n_samples: Samples per batch element.

        Returns:
            Sample tensor, shape (B, n_samples, d).
        """
        device, dtype = self.b.device, self.b.dtype
        z = torch.randn((self.B, n_samples, self.n_blocks, 2), device=device, dtype=dtype)
        s1 = self.sigma1.unsqueeze(1)
        s2 = self.sigma2.unsqueeze(1)
        bb = self.b.unsqueeze(1)

        u_r = s1 * z[..., 0]
        v_r = s2 * z[..., 1] + bb * u_r**2 - bb * s1**2

        cos_t = self.theta.cos().unsqueeze(1)
        sin_t = self.theta.sin().unsqueeze(1)
        # within-block rotation.
        u = cos_t * u_r - sin_t * v_r
        v = sin_t * u_r + cos_t * v_r

        y = torch.stack([u, v], dim=-1).reshape(self.B, n_samples, 2 * self.n_blocks)
        if self.has_tail:
            tail = torch.randn((self.B, n_samples, 1), device=device, dtype=dtype)
            y = torch.cat([y, tail], dim=-1)
        return y

    def quick_scatter(self, result: dict, max_points: int = 2000):
        """Scatter plot of the first coordinate pair against a ground-truth draw.

        Args:
            result: Sampler output dictionary.
            max_points: Maximum points to plot.

        Returns:
            Matplotlib figure.
        """
        import matplotlib.pyplot as plt

        x = result["final_samples"][0].detach().cpu()[..., :2]
        if x.shape[0] > max_points:
            x = x[torch.randperm(x.shape[0])[:max_points]]
        gt = self.sample(min(max_points, 1000))[0].detach().cpu()[..., :2]

        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
        ax.scatter(gt[:, 0], gt[:, 1], s=6, alpha=0.12, c="#222222", label="target")
        ax.scatter(x[:, 0], x[:, 1], s=6, alpha=0.12, c="#4a90d9", label="samples")
        ax.set_title(f"{self.name} (coords 0,1 of {self.d})")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        return fig


def sample_banana_nd_targets(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    b_range: tuple = (0.1, 0.5),
    sigma1_range: tuple = (1.0, 3.0),
    sigma2_range: tuple = (0.5, 1.5),
    spatial_dim: int = 2,
    pin_seed: int | None = None,
) -> BatchedBananaND:
    """Sample a batch of d-dimensional banana targets.

    Args:
        batch_size: Number of target instances.
        device: Torch device.
        dtype: Torch dtype.
        b_range: Range for the per-block curvature.
        sigma1_range: Range for the first per-block scale.
        sigma2_range: Range for the second per-block scale.
        spatial_dim: Dimensionality, at least 2.
        pin_seed: When set, draw one instance under this seed and replicate it across the batch.

    Returns:
        The batched target.
    """
    if pin_seed is not None:
        with torch.random.fork_rng(devices=[] if device.type == "cpu" else [device]):
            torch.manual_seed(int(pin_seed))
            one = sample_banana_nd_targets(
                1, device, dtype, b_range, sigma1_range, sigma2_range, spatial_dim,
                pin_seed=None,
            )
        return BatchedBananaND(
            one.b.expand(batch_size, -1).contiguous(),
            one.sigma1.expand(batch_size, -1).contiguous(),
            one.sigma2.expand(batch_size, -1).contiguous(),
            one.theta.expand(batch_size, -1).contiguous(),
            spatial_dim=spatial_dim,
        )

    n_blocks = max(1, spatial_dim // 2)

    def uniform(lo, hi):
        return torch.rand((batch_size, n_blocks), device=device, dtype=dtype) * (hi - lo) + lo

    b = uniform(*b_range)
    sigma1 = uniform(*sigma1_range)
    sigma2 = uniform(*sigma2_range)
    theta = uniform(0.0, 2.0 * math.pi)
    return BatchedBananaND(b, sigma1, sigma2, theta, spatial_dim=spatial_dim)
