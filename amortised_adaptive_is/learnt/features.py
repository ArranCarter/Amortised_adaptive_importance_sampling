"""The EM-statistics features the update network reads.
"""


import torch

from amortised_adaptive_is.core.proposal import COV_MODES, Proposal


def n_scale_params(d: int, cov: str) -> int:
    """Number of network outputs needed to update the covariance under `cov`."""
    if cov == "iso":
        return 1
    if cov == "diag":
        return d
    if cov == "full":
        return d * (d + 1) // 2
    raise ValueError(f"Unknown cov mode {cov!r}; expected one of {COV_MODES}")


def em_statistics_hard(
    x: torch.Tensor,
    log_w: torch.Tensor,
    comp_idx: torch.Tensor,
    n_components: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """EM statistics with responsibilities replaced by the generating-component indicator.

    Args:
        x: Samples, shape (B, S, d).
        log_w: Unnormalised log importance weights, shape (B, S).
        comp_idx: Generating component per particle, shape (B, S).
        n_components: Number of components `N`.

    Returns:
        Tuple of (mu_em, cov_em, log_mass), shapes (B, N, d), (B, N, d, d), (B, N).
    """
    B, S, d = x.shape
    N = n_components
    group_lse = segment_logsumexp(log_w, comp_idx, N) # (B, N)
    a_hat = (log_w - group_lse.gather(1, comp_idx)).exp() # sums to 1 per group

    idx_d = comp_idx.unsqueeze(-1).expand(-1, -1, d)
    mu_em = torch.zeros((B, N, d), device=x.device, dtype=x.dtype).scatter_add(
        1, idx_d, a_hat.unsqueeze(-1) * x
    )
    outer = torch.einsum("bs,bsi,bsj->bsij", a_hat, x, x).reshape(B, S, d * d)
    second = torch.zeros((B, N, d * d), device=x.device, dtype=x.dtype).scatter_add(
        1, comp_idx.unsqueeze(-1).expand(-1, -1, d * d), outer
    ).reshape(B, N, d, d)
    cov_em = second - torch.einsum("bni,bnj->bnij", mu_em, mu_em)

    log_mass = group_lse - torch.logsumexp(log_w, dim=1, keepdim=True)
    return mu_em, cov_em, log_mass


def whiten_em_statistics(
    prop: Proposal,
    mu_em: torch.Tensor,
    cov_em: torch.Tensor,
    jitter: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale free statistics.

    Args:
        prop: The proposal supplying `L_n`.
        mu_em: EM mean update, shape (B, N, d).
        cov_em: EM covariance update, shape (B, N, d, d).
        jitter: Added to the diagonal of `cov_em` before.

    Returns:
        Tuple of (nu, C) with shapes (B, N, d) and (B, N, d, d).
    """
    d = prop.d
    L = prop.scale_tril()
    eye = torch.eye(d, device=L.device, dtype=L.dtype)

    nu = torch.linalg.solve_triangular(L, (mu_em - prop.mu).unsqueeze(-1), upper=False).squeeze(-1)
    # C = L^{-1} Sigma L^{-T}: solve on the left, then on the right.
    tmp = torch.linalg.solve_triangular(L, cov_em + jitter * eye, upper=False)
    C = torch.linalg.solve_triangular(L, tmp.transpose(-1, -2), upper=False).transpose(-1, -2)
    return nu, C


def apply_update(
    prop: Proposal,
    delta_mu: torch.Tensor,
    delta_scale: torch.Tensor,
    delta_logit: torch.Tensor | None,
    cov: str,
    unit: torch.Tensor,
) -> Proposal:
    """Apply the update to the proposal.

    Args:
        prop: Current proposal.
        delta_mu: Mean update in units, shape (B, N, d).
        delta_scale: Scale update, shape (B, N, n_scale_params(d, cov)).
        delta_logit: Mixture-logit update, shape (B, N), or None to leave weights alone.
        cov: One of `COV_MODES`.
        unit: Length unit for the mean update -- either (B, 1, 1) / (B, N, 1) for a scalar unit, or (B, N, d, d) to apply `L_n` itself.

    Returns:
        The updated proposal.
    """
    _B, _N, d = prop.mu.shape

    if unit.dim() == 4:
        mu_new = prop.mu + torch.einsum("bnij,bnj->bni", unit, delta_mu)
    else:
        mu_new = prop.mu + unit * delta_mu

    low_new = prop.low
    if cov == "iso":
        log_diag_new = prop.log_diag + delta_scale[..., :1]
    elif cov == "diag":
        log_diag_new = prop.log_diag + delta_scale
    elif cov == "full":
        # A_n = diag(exp(a)) + strict lower, L_new = L A. 
        r, c = torch.tril_indices(d, d, offset=-1, device=prop.mu.device)
        A = torch.diag_embed(delta_scale[..., :d].exp())
        A[..., r, c] = delta_scale[..., d:]
        L_new = prop.scale_tril() @ A

        diag_new = torch.diagonal(L_new, dim1=-2, dim2=-1).clamp(min=1e-12)
        log_diag_new = diag_new.log()
        low_new = torch.tril(L_new, diagonal=-1)
    else:
        raise ValueError(f"Unknown cov mode {cov!r}; expected one of {COV_MODES}")

    logits_new = prop.logits if delta_logit is None else prop.logits + delta_logit
    return Proposal(mu=mu_new, log_diag=log_diag_new, low=low_new, logits=logits_new)


def segment_logsumexp(vals: torch.Tensor, idx: torch.Tensor, n_segments: int) -> torch.Tensor:
    """`logsumexp` of `vals` within each segment named by `idx`.

    Args:
        vals: Values, shape (B, S).
        idx: Segment index per value, shape (B, S), entries in [0, n_segments).
        n_segments: Number of segments `N`.

    Returns:
        Segment log-sum-exp, shape (B, N). Empty segments yield `-inf`.
    """
    B = vals.shape[0]
    neg_inf = torch.finfo(vals.dtype).min
    m = torch.full((B, n_segments), neg_inf, device=vals.device, dtype=vals.dtype)
    m = m.scatter_reduce(1, idx, vals, reduce="amax", include_self=True)
    shifted = (vals - m.gather(1, idx)).exp()
    total = torch.zeros((B, n_segments), device=vals.device, dtype=vals.dtype)
    total = total.scatter_add(1, idx, shifted)
    return m + total.clamp(min=1e-30).log()


def segment_mean(vals: torch.Tensor, idx: torch.Tensor, n_segments: int) -> torch.Tensor:
    """Mean of `vals` within each segment.

    Args:
        vals: Values, shape (B, S, H).
        idx: Segment index per value, shape (B, S).
        n_segments: Number of segments `N`.

    Returns:
        Segment means, shape (B, N, H). Empty segments yield 0.
    """
    B, _S, H = vals.shape
    out = torch.zeros((B, n_segments, H), device=vals.device, dtype=vals.dtype)
    out = out.scatter_add(1, idx.unsqueeze(-1).expand(-1, -1, H), vals)
    counts = torch.zeros((B, n_segments), device=vals.device, dtype=vals.dtype)
    counts = counts.scatter_add(1, idx, torch.ones_like(idx, dtype=vals.dtype))
    return out / counts.clamp(min=1.0).unsqueeze(-1)
