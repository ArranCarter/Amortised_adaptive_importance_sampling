"""Gaussian mixture model target family.
"""

import torch

from amortised_adaptive_is.core.gmm import batched_gaussian_mixture_log_density


def _sample_centres(
    batch_size: int,
    n_components: int,
    spatial_dim: int,
    centre_range: float,
    min_sep: float,
    device: torch.device,
    dtype: torch.dtype,
    max_tries: int = 100,
) -> torch.Tensor:
    """Sample well-separated mixture centres for a batched GMM.

    Args:
        batch_size: Number of target instances.
        n_components: Mixture component count.
        spatial_dim: Dimensionality.
        centre_range: Range for uniform centre sampling.
        min_sep: Minimum pairwise separation between centres.
        device: Torch device.
        dtype: Torch dtype.
        max_tries: Maximum resampling attempts for violating targets.

    Returns:
        Sampled centre tensor.
    """
    centres = torch.rand((batch_size, n_components, spatial_dim), device=device, dtype=dtype)
    centres = centres * (2 * centre_range) - centre_range

    for _ in range(max_tries):
        dists = torch.cdist(centres, centres)
        eye = torch.eye(n_components, device=device, dtype=torch.bool).unsqueeze(0)
        dists_off = dists.masked_fill(eye, float("inf"))
        violation = (dists_off < min_sep).any(dim=-1).any(dim=-1)
        if not violation.any():
            break

        n_bad = int(violation.sum().item())
        new_centres = torch.rand((n_bad, n_components, spatial_dim), device=device, dtype=dtype)
        new_centres = new_centres * (2 * centre_range) - centre_range
        centres[violation] = new_centres

    perm = torch.argsort(torch.rand((batch_size, n_components), device=device), dim=1)
    centres = centres[torch.arange(batch_size, device=device).unsqueeze(1), perm]
    return centres

def _sample_centres_4_corners(
    batch_size: int,
    n_components: int,
    spatial_dim: int,
    centre_range: float,
    min_sep: float,
    device: torch.device,
    dtype: torch.dtype,
    max_tries: int = 100,
) -> torch.Tensor:
    """Sample GMM centres fixed at the corners of a square.

    Args:
        batch_size: Number of target instances.
        n_components: Mixture component count.
        spatial_dim: Dimensionality.
        centre_range: Range for uniform centre sampling.
        min_sep: Minimum pairwise separation between centres. (Ignored for corner sampling.)
        device: Torch device.
        dtype: Torch dtype.
        max_tries: Maximum resampling attempts for violating targets. (Ignored for corner sampling.)

    Returns:
        Sampled centre tensor.
    """
    centres = torch.randint(0, 2, (batch_size, n_components, spatial_dim), device=device, dtype=dtype)
    centres = centres * (2 * centre_range) - centre_range

    return centres


class BatchedGMM:
    """Batched diagonal-covariance Gaussian mixture model target."""

    name = "GMM"

    def __init__(self, centres: torch.Tensor, variances: torch.Tensor, component_logits: torch.Tensor) -> None:
        self.centres = centres
        self.variances = variances
        self.component_logits = component_logits
        self.B, self.N, self.d = centres.shape

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute target log density.

        Args:
            x: Samples, shape (B, S, d).

        Returns:
            Log density for each sample.
        """
        return batched_gaussian_mixture_log_density(
            x,
            self.centres,
            self.variances,
            self.component_logits,
            count_as_proposal=False,
        )

    def sample(self, n_samples: int) -> torch.Tensor:
        """Draw samples from the GMM.

        Args:
            n_samples: Number of samples per batch element.

        Returns:
                Sample tensor, shape (B, n_samples, d).
        """
        device = self.centres.device
        weights = torch.softmax(self.component_logits, dim=-1)
        idx = torch.multinomial(weights, num_samples=n_samples, replacement=True)
        ar = torch.arange(self.B, device=device).unsqueeze(-1)
        means = self.centres[ar, idx]
        stds = self.variances[ar, idx].sqrt()
        return means + stds * torch.randn_like(means)

    def quick_scatter(self, result: dict, max_points: int = 2000):
        """Create a simple 2D scatter plot for logging.

        Args:
            result: Sampler output dictionary.
            max_points: Maximum points to plot.

        Returns:
            Matplotlib figure.
        """
        import matplotlib.pyplot as plt

        # Pick first batch element for visualization.
        centres = result["centres"][0].detach().cpu()
        x = result["final_samples"][0].detach().cpu()
        if x.shape[0] > max_points:
            x = x[torch.randperm(x.shape[0])[:max_points]]

        gt = self.sample(min(max_points, 1000))[0].detach().cpu()

        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
        ax.scatter(gt[:, 0], gt[:, 1], s=6, alpha=0.12, c="#222222", label="target")
        ax.scatter(x[:, 0], x[:, 1], s=6, alpha=0.12, c="#4a90d9", label="samples")
        ax.scatter(centres[:, 0], centres[:, 1], s=120, marker="+", c="#00FFD5", label="centres")
        ax.set_title(self.name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        return fig


def sample_gmm_targets(
    batch_size: int,
    n_components: int,
    centre_range: float,
    target_variance: float,
    min_sep: float,
    device: torch.device,
    dtype: torch.dtype,
    spatial_dim: int = 2,
) -> BatchedGMM:
    """Sample a batch of GMM targets.

    Args:
        batch_size: Number of target instances.
        n_components: Mixture component count.
        centre_range: Range for uniform centre sampling.
        target_variance: Per-dimension component variance.
        min_sep: Minimum separation between component centres.
        device: Torch device.
        dtype: Torch dtype.
        spatial_dim: Dimensionality.

    Returns:
        Batched GMM target.
    """
    centres = _sample_centres(
        batch_size=batch_size,
        n_components=n_components,
        spatial_dim=spatial_dim,
        centre_range=centre_range,
        min_sep=min_sep,
        device=device,
        dtype=dtype,
    )
    variances = torch.ones((batch_size, n_components, spatial_dim), device=device, dtype=dtype) * float(target_variance)
    logits = torch.log(torch.ones((batch_size, n_components), device=device, dtype=dtype) / n_components)
    return BatchedGMM(centres, variances, logits)

def sample_gmm_targets_4_corners(
    batch_size: int,
    n_components: int,
    centre_range: float,
    target_variance: float,
    min_sep: float,
    device: torch.device,
    dtype: torch.dtype,
    spatial_dim: int = 2,
) -> BatchedGMM:
    """Sample a batch of GMM targets with centres fixed at the corners of a square.

    Args:
        batch_size: Number of target instances.
        n_components: Mixture component count.
        centre_range: Range for uniform centre sampling.
        target_variance: Per-dimension component variance.
        min_sep: Minimum separation between component centres. (Ignored for corner sampling.)
        device: Torch device.
        dtype: Torch dtype.
        spatial_dim: Dimensionality.

    Returns:
        Batched GMM target.
    """
    centres = _sample_centres_4_corners(
        batch_size=batch_size,
        n_components=n_components,
        spatial_dim=spatial_dim,
        centre_range=centre_range,
        min_sep=min_sep,
        device=device,
        dtype=dtype,
    )
    variances = torch.ones((batch_size, n_components, spatial_dim), device=device, dtype=dtype) * float(target_variance)
    logits = torch.log(torch.ones((batch_size, n_components), device=device, dtype=dtype) / n_components)
    return BatchedGMM(centres, variances, logits)