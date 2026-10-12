"""Hierarchical posterior.
"""

import math

import torch

# Clamp applied to the argument of exp to avoid overflow.
EXP_ARG_CLAMP = 60.0



def _clamped_exp(z: torch.Tensor) -> torch.Tensor:
    """`exp(z)` with clamped argument."""
    return torch.exp(z.clamp(min=-EXP_ARG_CLAMP, max=EXP_ARG_CLAMP))


class BatchedHierarchical:
    """Batched centred hierarchical normal-normal posteriors.

    Attributes:
        name: Display name.
        y: Observations, shape (B, J).
        sigma: Known per-group standard errors, shape (B, J).
        s_mu: Prior sd of `mu`.
        s_tau: Prior sd of `log tau`.
        d: Spatial dimension `J + 2`.
        log_z: Exact log normaliser per instance, shape (B,).
    """

    name = "Hierarchical"

    def __init__(
        self,
        y: torch.Tensor,
        sigma: torch.Tensor,
        s_mu: float = 5.0,
        s_tau: float = 1.0,
        n_grid: int = 401,
        mu_span: float = 6.0,
        lam_span: float = 5.0,
    ) -> None:
        self.y = y
        self.sigma = sigma
        self.s_mu = float(s_mu)
        self.s_tau = float(s_tau)
        self.B, self.J = y.shape
        self.d = self.J + 2

        device, dtype = y.device, y.dtype

        # --- 2-d quadrature grid for (mu, lambda) ---
        mu_g = torch.linspace(-mu_span * s_mu, mu_span * s_mu, n_grid, device=device, dtype=dtype)
        lam_g = torch.linspace(-lam_span * s_tau, lam_span * s_tau, n_grid, device=device, dtype=dtype)
        self.mu_grid, self.lam_grid = mu_g, lam_g
        M, L = torch.meshgrid(mu_g, lam_g, indexing="ij") # (n_grid, n_grid)
        lp = self._log_marginal(M.reshape(1, -1), L.reshape(1, -1)) # (B, n_grid*n_grid)
        lp = lp.reshape(self.B, n_grid, n_grid)

        mx = lp.amax(dim=(1, 2), keepdim=True)
        w = torch.exp(lp - mx)
        d_mu = (mu_g[1] - mu_g[0])
        d_lam = (lam_g[1] - lam_g[0])
        # Trapezoid in both directions.
        inner = torch.trapz(w, dx=1.0, dim=2) * d_lam # (B, n_grid)
        total = torch.trapz(inner, dx=1.0, dim=1) * d_mu # (B,)
        self.log_z = total.log() + mx.reshape(self.B)

        # Marginal of mu CDF and, for each mu, the conditional CDF of lambda, for sampling.
        self._mu_cdf = torch.cat(
            [torch.zeros((self.B, 1), device=device, dtype=dtype),
             torch.cumulative_trapezoid(inner, dx=1.0, dim=1) * d_mu], dim=1
        ) 
        self._mu_cdf = self._mu_cdf / self._mu_cdf[:, -1:]
        lam_cdf = torch.cat(
            [torch.zeros((self.B, n_grid, 1), device=device, dtype=dtype),
             torch.cumulative_trapezoid(w, dx=1.0, dim=2) * d_lam], dim=2
        )
        self._lam_cdf = lam_cdf / lam_cdf[..., -1:].clamp_min(1e-300) 

    def _log_marginal(self, mu: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """Unnormalised `log p(mu, lambda, y)` with `theta` integrated out, shape (B, S).

        Args:
            mu: Values of `mu`, shape (B, S) or (1, S).
            lam: Values of `log tau`, same shape.
        """
        tau2 = torch.exp(2.0 * lam)
        v = self.sigma.pow(2).unsqueeze(1) + tau2.unsqueeze(-1) # (B, S, J)
        resid = self.y.unsqueeze(1) - mu.unsqueeze(-1)
        log_lik = (-0.5 * (resid.pow(2) / v + v.log() + math.log(2.0 * math.pi))).sum(-1)
        log_prior = (
            -0.5 * (mu / self.s_mu) ** 2 - math.log(self.s_mu) - 0.5 * math.log(2.0 * math.pi)
            - 0.5 * (lam / self.s_tau) ** 2 - math.log(self.s_tau) - 0.5 * math.log(2.0 * math.pi)
        )
        return log_lik + log_prior

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised log density, shape (B, S).

        Args:
            x: Samples, shape (B, S, d) laid out as `(theta_1..J, mu, log tau)`.
        """
        theta = x[..., : self.J]
        mu = x[..., self.J]
        lam = x[..., self.J + 1]
        tau = _clamped_exp(lam)

        # p(y | theta) p(theta | mu, tau) p(mu) p(log tau)
        log_lik = (
            -0.5 * ((self.y.unsqueeze(1) - theta) / self.sigma.unsqueeze(1)) ** 2
            - self.sigma.unsqueeze(1).log()
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(-1)
        log_pop = (
            -0.5 * ((theta - mu.unsqueeze(-1)) / tau.unsqueeze(-1)) ** 2
            - lam.unsqueeze(-1)
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(-1)
        log_prior = (
            -0.5 * (mu / self.s_mu) ** 2 - math.log(self.s_mu) - 0.5 * math.log(2.0 * math.pi)
            - 0.5 * (lam / self.s_tau) ** 2 - math.log(self.s_tau) - 0.5 * math.log(2.0 * math.pi)
        )
        return log_lik + log_pop + log_prior - self.log_z.unsqueeze(1)

    def sample(self, n_samples: int) -> torch.Tensor:
        """i.i.d. draws, shape (B, n_samples, d).
        Uses the grid-based marginal CDF of (mu, lambda).
        """
        B, G = self.B, self.mu_grid.shape[0]
        device, dtype = self.y.device, self.y.dtype

        u = torch.rand((B, n_samples), device=device, dtype=dtype)
        idx = torch.searchsorted(self._mu_cdf, u.contiguous()).clamp(1, G - 1)
        c0, c1 = torch.gather(self._mu_cdf, 1, idx - 1), torch.gather(self._mu_cdf, 1, idx)
        g0, g1 = self.mu_grid[idx - 1], self.mu_grid[idx]
        mu = g0 + (u - c0) / (c1 - c0).clamp_min(1e-30) * (g1 - g0)

        # Conditional lambda, using the CDF column of the nearest mu.
        col = idx.clamp(0, G - 1) 
        lam_cdf = torch.gather(
            self._lam_cdf, 1, col.unsqueeze(-1).expand(B, n_samples, G)
        ) # (B, S, G)
        u2 = torch.rand((B, n_samples, 1), device=device, dtype=dtype)
        j = torch.searchsorted(lam_cdf.contiguous(), u2.contiguous()).clamp(1, G - 1)
        d0 = torch.gather(lam_cdf, 2, j - 1).squeeze(-1)
        d1 = torch.gather(lam_cdf, 2, j).squeeze(-1)
        l0, l1 = self.lam_grid[(j - 1).squeeze(-1)], self.lam_grid[j.squeeze(-1)]
        lam = l0 + (u2.squeeze(-1) - d0) / (d1 - d0).clamp_min(1e-30) * (l1 - l0)

        tau2 = torch.exp(2.0 * lam).unsqueeze(-1) # (B, S, 1)
        s2 = self.sigma.pow(2).unsqueeze(1) # (B, 1, J)
        prec = 1.0 / s2 + 1.0 / tau2
        mean = (self.y.unsqueeze(1) / s2 + mu.unsqueeze(-1) / tau2) / prec
        theta = mean + prec.rsqrt() * torch.randn((B, n_samples, self.J), device=device, dtype=dtype)
        return torch.cat([theta, mu.unsqueeze(-1), lam.unsqueeze(-1)], dim=-1)


def sample_hierarchical_targets(
    batch_size: int,
    spatial_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    sigma_range: tuple = (2.0, 18.0),
    tau_true_range: tuple = (0.5, 8.0),
    mu_true_scale: float = 5.0,
    s_mu: float = 5.0,
    s_tau: float = 1.0,
    pin_seed: int | None = None,
) -> BatchedHierarchical:
    """Sample a batch of hierarchical normal-normal posteriors.

    Args:
        batch_size: Number of independent posteriors `B`.
        spatial_dim: Total dimension `d = J + 2`; the model has `J = d - 2` groups.
        device: Torch device.
        dtype: Torch dtype.
        sigma_range: Range for the known per-group standard errors.
        tau_true_range: Range for the data-generating `tau*`
        mu_true_scale: Scale of the data-generating `mu*`.
        s_mu: Prior sd of `mu`.
        s_tau: Prior sd of `log tau`.
        pin_seed: Draw one dataset under this seed and replicate it across the batch.

    Returns:
        Batched posterior family.
    """
    if spatial_dim < 3:
        raise ValueError(f"hierarchical needs spatial_dim >= 3 (J >= 1), got {spatial_dim}")
    if pin_seed is not None:
        with torch.random.fork_rng(devices=[] if device.type == "cpu" else [device]):
            torch.manual_seed(int(pin_seed))
            one = sample_hierarchical_targets(
                1, spatial_dim, device, dtype, sigma_range, tau_true_range, mu_true_scale,
                s_mu, s_tau, pin_seed=None,
            )
        return BatchedHierarchical(
            y=one.y.expand(batch_size, -1).contiguous(),
            sigma=one.sigma.expand(batch_size, -1).contiguous(),
            s_mu=s_mu, s_tau=s_tau,
        )

    J = spatial_dim - 2
    sigma = (
        torch.rand((batch_size, J), device=device, dtype=dtype)
        * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
    )
    tau_star = (
        torch.rand((batch_size, 1), device=device, dtype=dtype)
        * (tau_true_range[1] - tau_true_range[0]) + tau_true_range[0]
    )
    mu_star = mu_true_scale * torch.randn((batch_size, 1), device=device, dtype=dtype)
    theta_star = mu_star + tau_star * torch.randn((batch_size, J), device=device, dtype=dtype)
    y = theta_star + sigma * torch.randn((batch_size, J), device=device, dtype=dtype)
    return BatchedHierarchical(y=y, sigma=sigma, s_mu=s_mu, s_tau=s_tau)
