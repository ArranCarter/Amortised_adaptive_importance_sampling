"""Update network for the isotropic / diagonal / full-covariance proposal.
"""

import math

import torch
from torch import nn

LOG_FEAT_CLAMP = 20.0
LOG_FEAT_SCALE = 5.0

def soft_clamp(x: torch.Tensor, max_val: float) -> torch.Tensor:
    """Smoothly clamp values to [-max_val, max_val].

    Args:
        x: Input tensor.
        max_val: Symmetric bound.

    Returns:
        Tensor with values smoothly clamped.
    """
    return torch.tanh(x / max_val) * max_val    


from amortised_adaptive_is.core.proposal import Proposal
from amortised_adaptive_is.learnt.features import (
    apply_update,
    em_statistics_hard,
    n_scale_params,
    segment_logsumexp,
    segment_mean,
    whiten_em_statistics,
)


def _shape_features(M: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a batch of SPD matrices into log-scales and correlations.

    Args:
        M: Matrices, shape (..., d, d).
        eps: Floor on the diagonal.

    Returns:
        Tuple of (log_diag, corr) with shapes (..., d) and (..., d(d-1)/2).
    """
    d = M.shape[-1]
    diag = torch.diagonal(M, dim1=-2, dim2=-1).clamp(min=eps)
    r, c = torch.tril_indices(d, d, offset=-1, device=M.device)
    if r.numel() == 0:
        corr = M.new_zeros(M.shape[:-2] + (0,))
    else:
        denom = (diag[..., r] * diag[..., c]).sqrt().clamp(min=eps)
        corr = (M[..., r, c] / denom).clamp(-1.0, 1.0)
    return diag.log(), corr


class EMUpdateNet(nn.Module):
    """Equivariant update network.

    Args:
        d: Spatial dimension.
        hidden_dim: Width of every MLP.
        cov: `iso`, `diag` or `full`.
        learn_logits: Emit a mixture-logit increment as well.
        use_log_pi: Feed the globally-normalised target log density per particle.
        use_moments: Feed the EM statistics as component features.
        use_global_w: Feed the globally-normalised per-particle log weight.
        delta_scale: Length unit for the mean update -- `ensemble`, `sigma`, or `abs`.
        use_grad: Feed `L_n^T grad log pi(x)` per particle. 
    """

    def __init__(
        self,
        d: int,
        hidden_dim: int = 32,
        cov: str = "diag",
        learn_logits: bool = False,
        use_log_pi: bool = True,
        use_moments: bool = True,
        use_global_w: bool = True,
        delta_scale: str = "ensemble",
        use_grad: bool = False,
        feat_compress: str = "clip",
        feat_scale: float = LOG_FEAT_SCALE,
        scale_free: bool = True,
    ) -> None:
        super().__init__()
        self.scale_free = bool(scale_free)
        # How unbounded log-ratio features are clipped.
        self.feat_compress = str(feat_compress)
        self.feat_scale = float(feat_scale)

        self.d = d
        self.use_grad = use_grad
        self.cov = cov
        self.learn_logits = learn_logits
        self.use_log_pi = use_log_pi
        self.use_moments = use_moments
        self.use_global_w = use_global_w
        self.delta_scale = delta_scale

        n_corr = d * (d - 1) // 2 if cov == "full" else 0
        self.n_scale = n_scale_params(d, cov)

        # Per particle: offset (d) || within-component log weight (1)
        # || optional global log weight (1) || optional global log target density (1)
        # || optional target gradient (d).
        phi_in = d + 1 + int(use_global_w) + int(use_log_pi) + d * int(use_grad)
        self.phi = nn.Sequential(
            nn.Linear(phi_in, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

        # Per component: pooled code (hidden_dim) || mu_rel (d) || log(diag L / s) (d) || corr(Sigma)
        # || EM mass (1) || current log weight (1) || t (1) || optional EM stats.
        comp_in = hidden_dim + d + d + n_corr + 1 + 1 + 1
        if not self.scale_free:
            comp_in += d + d + 1 # absolute mean, absolute log-scale, log of the ensemble scale
        if use_moments:
            comp_in += d + d + n_corr # nu, log diag C, corr C
        if use_grad:
            comp_in += d # scaled grad log pi at the component mean
        self.rho = nn.Sequential(nn.Linear(comp_in, hidden_dim), nn.SiLU())
        self.cross = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU())

        self.head = nn.Linear(
            hidden_dim,
            d + self.n_scale + int(learn_logits)
            + (d if not self.scale_free else 0),
        )
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if not self.scale_free:
            # The absolute length the second mean update is measured in, learnt.
            self.log_abs_unit = nn.Parameter(torch.zeros(()))

    def _lf(self, x: torch.Tensor) -> torch.Tensor:
        """log-ratio feature transform."""
        if self.feat_compress == "clip":
                return x.clamp(-LOG_FEAT_CLAMP, LOG_FEAT_CLAMP) / self.feat_scale
        if self.feat_compress == "asinh":
            return torch.asinh(x / self.feat_scale)
        raise ValueError(f"feat_compress must be 'clip' or 'asinh', got {self.feat_compress!r}")

    
    def forward(
        self,
        prop: Proposal,
        x: torch.Tensor,
        comp_idx: torch.Tensor,
        log_w: torch.Tensor,
        log_pi: torch.Tensor,
        grad_log_pi: torch.Tensor | None = None,
        grad_mu: torch.Tensor | None = None,
        t: float = 0.0,
        max_delta_mu: float = 1.0,
        max_delta_log_sigma: float = 0.5,
        max_delta_logit: float = 1.0,
    ) -> Proposal:
        """Produce the next proposal.

        Args:
            prop: The current proposal.
            x: Particles drawn from it, shape (B, S, d).
            comp_idx: Generating component per particle, shape (B, S).
            log_w: Log importance weights, shape (B, S).
            log_pi: Target log-density at the particles, shape (B, S).
            grad_log_pi: Gradient of `log pi` at the particles, or None.
            grad_mu: Gradient of `log pi` at the component means, or None.
            t: Step index, fed as a feature.
            max_delta_mu: Bound on the mean update, in `delta_scale` units.
            max_delta_log_sigma: Bound on the change in log-variance.
            max_delta_logit: Bound on the change in a mixture logit.

        Returns:
            The updated proposal.
        """
        out = self.deltas(
            prop, x, comp_idx, log_w, log_pi, grad_log_pi, grad_mu,
            t=t, max_delta_mu=max_delta_mu, max_delta_log_sigma=max_delta_log_sigma,
            max_delta_logit=max_delta_logit,
        )
        delta_mu, delta_sc, delta_logit, unit = out[:4]
        return apply_update(prop, delta_mu, delta_sc, delta_logit, self.cov, unit)

    def deltas(
        self,
        prop: Proposal,
        x: torch.Tensor,
        comp_idx: torch.Tensor,
        log_w: torch.Tensor,
        log_pi: torch.Tensor,
        grad_log_pi: torch.Tensor | None = None,
        grad_mu: torch.Tensor | None = None,
        t: float = 0.0,
        max_delta_mu: float = 1.0,
        max_delta_log_sigma: float = 0.5,
        max_delta_logit: float = 1.0,
    ) -> tuple:
        """The bounded update components, before they are applied.
        
        Args:
            prop: Current proposal.
            x: Particles, shape (B, S, d).
            comp_idx: Generating component per particle, shape (B, S).
            log_w: Log importance weights, shape (B, S).
            log_pi: Target log density, shape (B, S).
            grad_log_pi: `grad log pi` at each particle, shape (B, S, d).
            grad_mu: `grad log pi` at each component mean, shape (B, N, d).
            t: Normalised step index in [0, 1].
            max_delta_mu: Bound on the mean update, in `delta_scale` units.
            max_delta_log_sigma: Bound on each log-scale update.
            max_delta_logit: Bound on the mixture-logit update.

        Returns:
            Tuple of (delta_mu, delta_scale, delta_logit, unit) in the form `apply_update`
            expects.
        """
        B, N, d = prop.mu.shape
        S = x.shape[1]
        L = prop.scale_tril()
        ar = torch.arange(B, device=x.device).unsqueeze(-1)

        # ensemble length scale (detached)
        mu_bar = prop.mu.mean(dim=1, keepdim=True)
        var_spread = (prop.mu - mu_bar).pow(2).mean(dim=(1, 2), keepdim=True)
        var_prop = prop.variances().mean(dim=(1, 2)).view(B, 1, 1)
        s = (var_spread + var_prop).sqrt().detach() # (B, 1, 1)

        # --- per-particle features ---
        # difference from the component that drew the particle, standardised.
        diff = x - prop.mu[ar, comp_idx] # (B, S, d)
        if prop.is_diagonal():
            z = diff / prop.log_diag.exp()[ar, comp_idx]
        else:
            z = torch.linalg.solve_triangular(
                L[ar, comp_idx], diff.unsqueeze(-1), upper=False
            ).squeeze(-1)

        counts = torch.zeros((B, N), device=x.device, dtype=x.dtype).scatter_add(
            1, comp_idx, torch.ones_like(comp_idx, dtype=x.dtype)
        )
        lw_rel = log_w - segment_logsumexp(log_w, comp_idx, N).gather(1, comp_idx)
        lw_rel = lw_rel + counts.clamp(min=1.0).log().gather(1, comp_idx)
        feats = [z, self._lf(lw_rel).unsqueeze(-1)]

        if self.use_global_w:
            lw_glob = torch.log_softmax(log_w, dim=1) + math.log(S)
            feats.append(self._lf(lw_glob).unsqueeze(-1))
        if self.use_log_pi:
            lp_glob = torch.log_softmax(log_pi, dim=1) + math.log(S)
            feats.append(self._lf(lp_glob).unsqueeze(-1))
        if self.use_grad:
            if grad_log_pi is None:
                raise ValueError("use_grad=True but grad_log_pi was not supplied")
            # L_n^T grad log pi
            gw = torch.einsum("bsij,bsi->bsj", L[ar, comp_idx], grad_log_pi)
            feats.append(self._lf(gw))

        h = segment_mean(self.phi(torch.cat(feats, dim=-1)), comp_idx, N) # (B, N, H)

        # --- per-component features ---
        mu_em, cov_em, log_mass = em_statistics_hard(x, log_w, comp_idx, N)
        sigma_log_diag, sigma_corr = _shape_features(torch.einsum("bnij,bnkj->bnik", L, L))

        comp = [
            h,
            (prop.mu - mu_bar) / s,
            0.5 * sigma_log_diag - s.log(), # log(scale_j / s)
            *( [sigma_corr] if self.cov == "full" else [] ),
            self._lf(log_mass + math.log(N)).unsqueeze(-1),
            self._lf(prop.log_alpha() + math.log(N)).unsqueeze(-1),
            torch.full((B, N, 1), float(t), device=x.device, dtype=x.dtype),
        ]
        if not self.scale_free:
            # Absolute position, absolute log-scale, and the ensemble scale itself.
            comp += [prop.mu, prop.log_diag, s.log().expand(B, N, 1)]
        if self.use_moments:
            nu, C = whiten_em_statistics(prop, mu_em, cov_em)
            c_log_diag, c_corr = _shape_features(C)
            comp += [nu, self._lf(c_log_diag)]
            if self.cov == "full":
                comp.append(c_corr)
        if self.use_grad:
            # The standardised gradient at the component mean
            gm = torch.einsum("bnij,bni->bnj", L, grad_mu)
            comp.append(self._lf(gm))

        # component level features
        zc = self.rho(torch.cat(comp, dim=-1))

        # component level features using the cross component information
        zc = self.cross(torch.cat([zc, zc.mean(dim=1, keepdim=True).expand(-1, N, -1)], dim=-1))

        delta = self.head(zc)

        # --- update ---
        delta_mu = soft_clamp(delta[..., :d], max_delta_mu)

        abs_step = None
        if not self.scale_free:
            # the combined update is  s * bounded(rel)  +  exp(rho) * bounded(abs).
            abs_step = self.log_abs_unit.exp() * soft_clamp(delta[..., -d:], max_delta_mu)


        # max_delta_log_sigma bounds the change in log-variance
        n_diag = 1 if self.cov == "iso" else d
        delta_sc = 0.5 * soft_clamp(delta[..., d : d + n_diag], max_delta_log_sigma)
        if self.cov == "full":
            off_diag = soft_clamp(delta[..., d + n_diag : d + self.n_scale], max_delta_log_sigma)
            delta_sc = torch.cat([delta_sc, off_diag], dim=-1)

        # max_delta_logit bounds the change in the mixture logits
        delta_logit = (
            soft_clamp(delta[..., d + self.n_scale], max_delta_logit)
            if self.learn_logits
            else None
        )

        if self.delta_scale == "ensemble":
            unit = s
        elif self.delta_scale == "sigma":
            unit = L
        else:
            unit = torch.ones_like(s)

        if abs_step is not None:
            # Combine the ensemble-relative and absolute mean steps into a single update
            if unit.dim() == 4:
                delta_mu = torch.einsum("bnij,bnj->bni", unit, delta_mu) + abs_step
            else:
                delta_mu = unit * delta_mu + abs_step
            unit = torch.ones_like(s)

        return delta_mu, delta_sc, delta_logit, unit


    def n_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())
