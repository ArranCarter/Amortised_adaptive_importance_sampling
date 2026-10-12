"""The Gaussian-mixture proposal, with isotropic, diagonal or full covariance.
"""

import math
from dataclasses import dataclass

import torch

from amortised_adaptive_is.core.counters import get_active_density_eval_counter

COV_MODES = ("iso", "diag", "full")
SAMPLING_MODES = ("stratified", "categorical")


@dataclass
class Proposal:
    """A batched diagonal or full Gaussian mixture.

    Attributes:
        mu: Component means, shape (B, N, d).
        log_diag: Log of the Cholesky diagonal, shape (B, N, d). For `iso` every coordinate
            holds the same value, for `diag` they differ, `full` adds `low`.
        low: Strictly-lower Cholesky entries, shape (B, N, d, d), zero on and above the
            diagonal. Kept at zero unless cov="full".
        logits: Unnormalised mixture logits, shape (B, N).
    """

    mu: torch.Tensor
    log_diag: torch.Tensor
    low: torch.Tensor
    logits: torch.Tensor

    @property
    def d(self) -> int:
        return self.mu.shape[-1]

    def scale_tril(self) -> torch.Tensor:
        """The Cholesky factor L, shape (B, N, d, d)."""
        return self.low + torch.diag_embed(self.log_diag.exp())

    def log_alpha(self) -> torch.Tensor:
        """Normalised log mixture weights, shape (B, N)."""
        return torch.log_softmax(self.logits, dim=-1)

    def detach(self) -> "Proposal":
        return Proposal(self.mu.detach(), self.log_diag.detach(), self.low.detach(), self.logits.detach())

    def clamp_scale(
        self, min_variance: float, max_variance: float, max_offdiag_ratio: float = 3.0
    ) -> "Proposal":
        """Keep the covariance in range.

        Args:
            min_variance: Floor on the Cholesky diagonal, squared.
            max_variance: Ceiling on the Cholesky diagonal, squared.
            max_offdiag_ratio: Bound on the off-diagonal row norm relative to the diagonal.

        Returns:
            The constrained proposal.
        """
        lo = 0.5 * math.log(min_variance)
        hi = 0.5 * math.log(max_variance) if max_variance != float("inf") else float("inf")
        log_diag = self.log_diag.clamp(min=lo, max=hi)

        low = self.low
        if low.abs().sum() > 0:
            off_norm = low.norm(dim=-1, keepdim=True) # (B, N, d, 1)
            cap = max_offdiag_ratio * log_diag.exp().unsqueeze(-1) # (B, N, d, 1)
            low = low * (cap / off_norm.clamp(min=1e-30)).clamp(max=1.0)
        return Proposal(self.mu, log_diag, low, self.logits)

    def is_diagonal(self) -> bool:
        """True when the strict lower triangle is empty.
        """
        return bool(torch.count_nonzero(self.low) == 0)

    def variances(self) -> torch.Tensor:
        """Variances, shape (B, N, d).
        """
        L = self.scale_tril()
        return (L * L).sum(dim=-1)


def isotropic(mu: torch.Tensor, variance: torch.Tensor | float) -> Proposal:
    """Build an isotropic proposal with uniform weights.

    Args:
        mu: Component means, shape (B, N, d).
        variance: Scalar or (B, N, d) tensor of initial variances.

    Returns:
        The proposal.
    """
    B, N, d = mu.shape
    if not torch.is_tensor(variance):
        variance = torch.full((B, N, d), float(variance), device=mu.device, dtype=mu.dtype)
    log_diag = 0.5 * variance.log()
    low = torch.zeros((B, N, d, d), device=mu.device, dtype=mu.dtype)
    logits = torch.zeros((B, N), device=mu.device, dtype=mu.dtype)
    return Proposal(mu=mu, log_diag=log_diag, low=low, logits=logits)


def diagonal(mu: torch.Tensor, variances: torch.Tensor, logits: torch.Tensor) -> Proposal:
    """Build a diagonal-covariance proposal.
    
    Args:
        mu: Component means, shape (B, N, d).
        variances: Per-coordinate variances, shape (B, N, d).
        logits: Unnormalised mixture weights, shape (B, N).

    Returns:
        The proposal.
    """
    return Proposal(
        mu=mu,
        log_diag=0.5 * variances.log(),
        low=torch.zeros(*variances.shape, variances.shape[-1],
                        device=variances.device, dtype=variances.dtype),
        logits=logits,
    )


def sample(
    prop: Proposal,
    n_particles: int,
    mode: str = "stratified",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a block of particles, reparameterised.

    Args:
        prop: The proposal.
        n_particles: Total particles `S` to draw. For `stratified` this must be a multiple of `N`.
        mode: `stratified` allocates exactly `S/N` particles to each component. `categorical` draws component indices i.i.d. according to `alpha`.
        generator: Optional RNG.

    Returns:
        Tuple of (x, comp_idx) with shapes (B, S, d) and (B, S). `comp_idx[b, i]` is the component that generated particle `i`.
    """
    B, N, d = prop.mu.shape
    device, dtype = prop.mu.device, prop.mu.dtype

    if mode == "stratified":
        if n_particles % N != 0:
            raise ValueError(f"stratified sampling needs S divisible by N; got {n_particles} and {N}")
        k = n_particles // N
        comp_idx = torch.arange(N, device=device).repeat_interleave(k).unsqueeze(0).expand(B, -1)
    elif mode == "categorical":
        probs = prop.log_alpha().exp()
        comp_idx = torch.multinomial(probs, num_samples=n_particles, replacement=True, generator=generator)
    else:
        raise ValueError(f"Unknown sampling mode {mode!r}, expected one of {SAMPLING_MODES}")

    eps = torch.randn((B, n_particles, d), device=device, dtype=dtype, generator=generator)
    ar = torch.arange(B, device=device).unsqueeze(-1)
    L_sel = prop.scale_tril()[ar, comp_idx] # (B, S, d, d)
    x = prop.mu[ar, comp_idx] + torch.einsum("bsij,bsj->bsi", L_sel, eps)
    return x, comp_idx


def log_density_components(
    prop: Proposal,
    x: torch.Tensor,
    count_as_proposal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the mixture, returning the per-component terms as well.

    Args:
        prop: The proposal.
        x: Points, shape (B, S, d).
        count_as_proposal: Report one proposal density evaluation to the active counter.

    Returns:
        Tuple of (log_q, log_comp), shapes (B, S) and (B, N, S).
    """
    if count_as_proposal:
        counter = get_active_density_eval_counter()
        if counter is not None:
            counter.add_proposal(x)

    _B, _N, d = prop.mu.shape
    L = prop.scale_tril() # (B, N, d, d)
    diff = x.unsqueeze(1) - prop.mu.unsqueeze(2) # (B, N, S, d)

    # z = L^{-1} diff
    if prop.is_diagonal():
        z = diff / prop.log_diag.exp().unsqueeze(2)
    else:
        z = torch.linalg.solve_triangular(
            L.unsqueeze(2), # (B, N, 1, d, d)
            diff.unsqueeze(-1), # (B, N, S, d, 1)
            upper=False,
        ).squeeze(-1) # (B, N, S, d)

    log_det = prop.log_diag.sum(dim=-1) # (B, N), = log|L|
    log_comp = (
        -0.5 * (z * z).sum(dim=-1)
        - log_det.unsqueeze(-1)
        - 0.5 * d * math.log(2.0 * math.pi)
        + prop.log_alpha().unsqueeze(-1)
    ) # (B, N, S)
    return torch.logsumexp(log_comp, dim=1), log_comp
