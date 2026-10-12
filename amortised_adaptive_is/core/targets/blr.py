"""Bayesian linear regression posterior.
"""

import math

import torch

# Clamp applied to the argument of exp to avoid overflow.
EXP_ARG_CLAMP = 60.0


def _clamped_exp(z: torch.Tensor) -> torch.Tensor:
    """`exp(z)` with clamped argument."""
    return torch.exp(z.clamp(min=-EXP_ARG_CLAMP, max=EXP_ARG_CLAMP))


class BatchedBLR:
    """Batched normal-inverse-gamma posteriors in `(beta, tau = log sigma^2)` coordinates.

    Attributes:
        name: Display name.
        mu_n: Posterior mean of the coefficients, shape (B, p).
        L_n: Cholesky factor of the posterior precision `Lambda_n`, shape (B, p, p), lower-triangular.
        a_n: Posterior shape, shape (B,).
        b_n: Posterior rate, shape (B,).
        d: Total dimension `p + 1`.
    """

    name = "BLR"

    def __init__(
        self,
        mu_n: torch.Tensor,
        L_n: torch.Tensor,
        a_n: torch.Tensor,
        b_n: torch.Tensor,
    ) -> None:
        self.mu_n = mu_n
        self.L_n = L_n
        self.a_n = a_n
        self.b_n = b_n
        self.B, self.p = mu_n.shape
        self.d = self.p + 1
        # log|Lambda_n|^{1/2} = sum log diag(L_n), used by both log_prob and the exact log-normaliser.
        self.half_logdet = torch.log(torch.diagonal(L_n, dim1=-2, dim2=-1)).sum(-1) # (B,)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log density.

        Args:
            x: Samples, shape (B, S, d). Coordinates `0..p-1` are `beta`, coordinate `p` is `tau`.

        Returns:
            Log density, shape (B, S).
        """
        beta, tau = x[..., : self.p], x[..., self.p] # (B,S,p), (B,S)
        diff = beta - self.mu_n.unsqueeze(1) # (B,S,p)
        # quad = (beta - mu_n)^T Lambda_n (beta - mu_n), Lambda_n = L_n L_n^T.
        quad = (diff @ self.L_n).pow(2).sum(-1) # (B,S)

        a, b = self.a_n.unsqueeze(1), self.b_n.unsqueeze(1)
        p = self.p
        # N(beta; mu_n, e^tau Lambda_n^{-1})
        log_n = (
            -0.5 * p * math.log(2.0 * math.pi)
            + self.half_logdet.unsqueeze(1)
            - 0.5 * p * tau
            - 0.5 * quad * _clamped_exp(-tau)
        )
        # InvGamma(sigma^2 = e^tau; a, b) times the Jacobian e^tau.
        log_ig = (
            a * torch.log(b) - torch.lgamma(a) - a * tau - b * _clamped_exp(-tau)
        )
        return log_n + log_ig

    def sample(self, n_samples: int) -> torch.Tensor:
        """Draw exact iid samples, shape (B, n_samples, d).
        """
        B, p = self.B, self.p
        device, dtype = self.mu_n.device, self.mu_n.dtype
        gamma = torch._standard_gamma(
            self.a_n.unsqueeze(1).expand(B, n_samples).contiguous()
        )
        sigma2 = self.b_n.unsqueeze(1) / gamma # (B,S) InvGamma
        tau = torch.log(sigma2)

        eps = torch.randn((B, n_samples, p), device=device, dtype=dtype)
        z = torch.linalg.solve_triangular(
            self.L_n.transpose(-1, -2).unsqueeze(1).expand(B, n_samples, p, p),
            eps.unsqueeze(-1),
            upper=True,
        ).squeeze(-1) # (B,S,p), cov = Lambda_n^{-1}
        beta = self.mu_n.unsqueeze(1) + z * sigma2.sqrt().unsqueeze(-1)
        return torch.cat([beta, tau.unsqueeze(-1)], dim=-1)

    def mean(self) -> torch.Tensor:
        """Exact mean of `(beta, tau)`, shape (B, d).

        `E[beta] = mu_n` (defined for `a_n > 1/2`), `E[tau] = E[log sigma^2] = log b_n - digamma(a_n)`.
        """
        e_tau = torch.log(self.b_n) - torch.digamma(self.a_n)
        return torch.cat([self.mu_n, e_tau.unsqueeze(-1)], dim=-1)


def sample_blr_targets(
    batch_size: int,
    spatial_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    n_range: tuple = (3, 12),
    cond_log10_range: tuple = (0.0, 2.0),
    beta_scale: float = 2.0,
    sigma_range: tuple = (0.3, 1.5),
    prior_precision: float = 0.1,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
    pin_seed: int | None = None,
) -> BatchedBLR:
    """Sample a batch of Bayesian linear regression posteriors.

    Args:
        batch_size: Number of independent posteriors `B`.
        spatial_dim: Total dimension `d`, the regression has `p = d - 1` coefficients and one `log sigma^2`.
        device: Torch device.
        dtype: Torch dtype.
        n_range: Range of dataset sizes, drawn per instance.
        cond_log10_range: `log10` of the design's condition number, drawn per instance.
        beta_scale: Scale of the true coefficients.
        sigma_range: Range of the true noise scale.
        prior_precision: prior precision scalar value.
        prior_a: Inverse-gamma prior shape.
        prior_b: Inverse-gamma prior rate.
        pin_seed: When set, draw one dataset under this seed and replicate it across the batch.

    Returns:
        Batched posterior family.
    """
    if pin_seed is not None:
        # One instance under a fixed seed.
        with torch.random.fork_rng(devices=[] if device.type == "cpu" else [device]):
            torch.manual_seed(int(pin_seed))
            one = sample_blr_targets(
                1, spatial_dim, device, dtype, n_range, cond_log10_range, beta_scale,
                sigma_range, prior_precision, prior_a, prior_b, pin_seed=None,
            )
        return BatchedBLR(
            mu_n=one.mu_n.expand(batch_size, -1).contiguous(),
            L_n=one.L_n.expand(batch_size, -1, -1).contiguous(),
            a_n=one.a_n.expand(batch_size).contiguous(),
            b_n=one.b_n.expand(batch_size).contiguous(),
        )
    if spatial_dim < 2:
        raise ValueError(f"blr needs spatial_dim >= 2 (p >= 1), got {spatial_dim}")
    p = spatial_dim - 1
    B = batch_size
    n_max = int(n_range[1])

    # --- design matrices ---
    # Drawn at the maximum size and then masked to the per-instance n.
    n_i = torch.randint(int(n_range[0]), n_max + 1, (B,), device=device)
    rows = torch.arange(n_max, device=device).view(1, n_max)
    mask = (rows < n_i.view(B, 1)).to(dtype) # (B, n_max)

    # Condition number of X
    log10_cond = (
        torch.rand(B, device=device, dtype=dtype)
        * (cond_log10_range[1] - cond_log10_range[0])
        + cond_log10_range[0]
    )
    cond = torch.pow(torch.tensor(10.0, device=device, dtype=dtype), log10_cond) # (B,)
    if p > 1:
        # Singular values over [1, cond].
        t = torch.linspace(0.0, 1.0, p, device=device, dtype=dtype).view(1, p)
        sv = torch.exp(t * torch.log(cond).view(B, 1)) # (B, p)
    else:
        sv = torch.ones((B, 1), device=device, dtype=dtype)
    q, _ = torch.linalg.qr(torch.randn((B, p, p), device=device, dtype=dtype))
    z = torch.randn((B, n_max, p), device=device, dtype=dtype)
    X = (z * sv.unsqueeze(1)) @ q.transpose(-1, -2)
    X = X * mask.unsqueeze(-1) # zero the unused rows

    # --- data ---
    beta_star = beta_scale * torch.randn((B, p), device=device, dtype=dtype)
    sigma_star = (
        torch.rand(B, device=device, dtype=dtype) * (sigma_range[1] - sigma_range[0])
        + sigma_range[0]
    )
    noise = sigma_star.view(B, 1) * torch.randn((B, n_max), device=device, dtype=dtype)
    y = ((X @ beta_star.unsqueeze(-1)).squeeze(-1) + noise) * mask

    # --- conjugate posterior ---
    eye = torch.eye(p, device=device, dtype=dtype).expand(B, p, p)
    Lambda_0 = prior_precision * eye
    Lambda_n = X.transpose(-1, -2) @ X + Lambda_0 # (B,p,p)
    L_n = torch.linalg.cholesky(Lambda_n)
    xty = (X * y.unsqueeze(-1)).sum(dim=1) # (B,p) = X^T y
    mu_n = torch.cholesky_solve(xty.unsqueeze(-1), L_n).squeeze(-1) # mu_0 = 0

    a_n = prior_a + 0.5 * n_i.to(dtype)
    # b_n = b_0 + (y^T y - mu_n^T Lambda_n mu_n)/2, with mu_0 = 0.
    yty = (y * y).sum(dim=1)
    quad_mu = (mu_n.unsqueeze(1) @ Lambda_n @ mu_n.unsqueeze(-1)).reshape(B)
    b_n = (prior_b + 0.5 * (yty - quad_mu)).clamp_min(1e-6)
    return BatchedBLR(mu_n=mu_n, L_n=L_n, a_n=a_n, b_n=b_n)
