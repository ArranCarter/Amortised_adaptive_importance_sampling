"""W&B logging for training runs.
"""

from __future__ import annotations

import math
import pathlib
from typing import Any

import torch


def _proposal_arrays(prop) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(means, widths, weights)`.
    """
    return prop.mu, prop.log_diag.exp(), torch.softmax(prop.logits, dim=-1)


class _TracedProposal:
    """The three attributes `_proposal_arrays` reads, built from a baseline's stored trace.
    """

    def __init__(self, centres: torch.Tensor, variances: torch.Tensor, logits: torch.Tensor,
                 scale_tril: torch.Tensor | None = None):
        self.mu = centres

        self.log_diag = (torch.diagonal(scale_tril, dim1=-2, dim2=-1).clamp(min=1e-30).log()
                         if scale_tril is not None
                         else 0.5 * variances.clamp(min=1e-30).log())
        self.logits = logits
        self._low = (scale_tril - torch.diag_embed(torch.diagonal(scale_tril, dim1=-2, dim2=-1))
                     if scale_tril is not None else None)

    def scale_tril(self) -> torch.Tensor:
        """The Cholesky factor `L`, shape (B, N, d, d)
        """
        L = torch.diag_embed(self.log_diag.exp())
        return L if self._low is None else L + self._low

    def log_alpha(self) -> torch.Tensor:
        """Normalised log mixture weights, shape (B, N)."""
        return torch.log_softmax(self.logits, dim=-1)


class _TracedRecord:
    """The three fields the rollout figures read from a record."""

    def __init__(self, prop, samples: torch.Tensor, log_w: torch.Tensor):
        self.prop, self.samples, self.log_w = prop, samples, log_w


def baseline_records(result: dict) -> list:
    """Adapt a baseline's `per_step` trace to what `rollout_figure`/`samples_figure` read.

    Args:
        result: A baseline `run` result carrying `per_step`.

    Returns:
        Records for `rollout_figure`, `samples_figure` and `step_metrics`.
    """
    if "per_step" not in result:
        raise KeyError("result has no 'per_step' trace")
    return [
        _TracedRecord(
            _TracedProposal(b["centres"], b["variances"], b["log_component_logits"],
                            b.get("scale_tril")),
            b["samples"], b["log_w"],
        )
        for b in result["per_step"]
    ]


def step_metrics(records, modes: torch.Tensor | None, mode_sd: float) -> list[dict[str, float]]:
    """Per-adaptation-step diagnostics for one rollout.

    Args:
        records: The `EMStepRecord` list from `unroll_for`.
        modes: Target mode locations `(B, M, d)`, or None for targets without them.
        mode_sd: Target standard deviation.

    Returns:
        One dict per step.
    """
    rows = []
    for t, rec in enumerate(records):
        mu, sd, mix = _proposal_arrays(rec.prop)
        w = torch.softmax(rec.log_w, dim=-1)
        row: dict[str, float] = {
            "step": float(t),
            "ess": float((1.0 / ((w**2).sum(-1) * w.shape[-1])).mean()),
            "max_w_share": float(w.max(dim=-1).values.mean()),
            "logw_std": float(rec.log_w.std(dim=-1).mean()),
            "width_sd": float(sd.clamp_min(1e-30).log().mean().exp() / mode_sd),
            "min_mix_weight": float(mix.min(dim=-1).values.mean()),
        }
        if modes is not None:
            d_all = torch.cdist(mu, modes) # (B, N, M)
            claimed = d_all.argmin(dim=-1)
            hit = torch.zeros_like(d_all[:, 0, :]).scatter_add(
                1, claimed, torch.ones_like(claimed, dtype=d_all.dtype))
            row["coverage"] = float((hit > 0).all(dim=-1).to(mu.dtype).mean())
            row["dist_sd"] = float(d_all.min(dim=-1).values.mean() / mode_sd)
            hits = (hit > 0).to(mu.dtype).sum(-1)
            row["modes_hit"] = float(hits.mean())

            row["modes_hit_frac"] = float((hits / min(mu.shape[1], modes.shape[1])).mean())
        rows.append(row)
    return rows


def _plane(modes: torch.Tensor) -> torch.Tensor:
    """An orthonormal 2-d basis `(2, d)` spanned by the first instance's first two modes."""
    d = modes.shape[-1]
    if d == 2:
        return torch.eye(2, dtype=modes.dtype, device=modes.device)
    a = modes[0, 0]
    b = modes[0, 1] if modes.shape[1] > 1 else torch.randn_like(a)
    e1 = a / a.norm().clamp_min(1e-12)
    r = b - (b @ e1) * e1
    if r.norm() < 1e-6 * max(1.0, float(b.norm())):
        axis = int(e1.abs().argmin())
        r = torch.zeros(d, dtype=modes.dtype, device=modes.device)
        r[axis] = 1.0
        r = r - (r @ e1) * e1
    e2 = r / r.norm().clamp_min(1e-12)
    return torch.stack([e1, e2])


def rollout_figure(records, modes: torch.Tensor | None, title: str, instance: int = 0):
    """One plot per adaptation step."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(records)
    cols = min(6, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.3 * rows), squeeze=False)
    basis = _plane(modes) if modes is not None else None
    proj_modes = _proj(modes[instance], basis) if modes is not None else None
    span = 0.0
    for rec in records:
        mu, _, _ = _proposal_arrays(rec.prop)
        pr = mu[instance] @ basis.T if basis is not None else mu[instance][:, :2]
        span = max(span, float(pr.abs().max()))
    if proj_modes is not None:
        span = max(span, 1.4 * float(proj_modes.abs().max()))
    span *= 1.05
    for t, rec in enumerate(records):
        ax = axes[t // cols][t % cols]
        mu, _sd, mix = _proposal_arrays(rec.prop)
        m = mu[instance]
        p = _proj(m, basis)

        area = 12.0 + 120.0 * mix[instance].detach().cpu()
        if proj_modes is not None:
            ax.scatter(proj_modes[:, 0], proj_modes[:, 1], marker="x", s=70, c=_MODE_C,
                       linewidths=1.8, label="target modes" if t == 0 else None, zorder=3)
        ax.scatter(p[:, 0], p[:, 1], s=area, c=_CENTRE_C, alpha=0.9,
                   label="components" if t == 0 else None, zorder=4)
        ax.set_title(f"t={t}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_aspect("equal")
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    if n:
        fig.legend(loc="lower right", fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return fig


_TARGET_C = "#333333"
_CENTRE_C = "#ff45cf"
_MODE_C = "#d40000"


def _basis_for(modes: torch.Tensor | None, instance: int, d: int, dtype, device):
    """The 2-d projection basis `(2, d)` used by every figure, or None to take coordinates 0 and 1."""
    if d == 2 or modes is None:
        return None
    return _plane(modes[instance : instance + 1])


def _proj(x: torch.Tensor, basis) -> torch.Tensor:
    """Project points.
    """
    out = x @ basis.T if basis is not None else x[..., :2]
    return out.detach().cpu()


def _frame(pts_list, pad_frac: float = 0.05):
    """Box covering every `(M, 2)` point set given."""
    lo = torch.stack([p.min(0).values for p in pts_list]).min(0).values
    hi = torch.stack([p.max(0).values for p in pts_list]).max(0).values
    pad = pad_frac * (hi - lo).clamp_min(1e-6)
    return lo - pad, hi + pad


def _gmm_contour(ax, m2, cov, w, lo, hi, grid, cmap="Blues", levels: int = 50):
    """Filled contours of a 2-d Gaussian mixture on a grid.
    """
    gx = torch.linspace(float(lo[0]), float(hi[0]), grid)
    gy = torch.linspace(float(lo[1]), float(hi[1]), grid)
    G = torch.stack(torch.meshgrid(gx, gy, indexing="xy"), dim=-1).reshape(-1, 2)
    Lc = torch.linalg.cholesky(cov + 1e-9 * torch.eye(2, dtype=cov.dtype))
    diff = G.unsqueeze(0) - m2.unsqueeze(1) # (N, P, 2)
    sol = torch.linalg.solve_triangular(Lc, diff.transpose(-1, -2), upper=False)
    quad = sol.pow(2).sum(1)
    logdet = 2.0 * torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)).sum(-1)
    comp = -0.5 * quad - 0.5 * logdet.unsqueeze(-1) - math.log(2.0 * math.pi)
    dens = torch.logsumexp(comp + w.clamp_min(1e-30).log().unsqueeze(-1), dim=0)


    ax.contourf(gx.numpy(), gy.numpy(), dens.reshape(grid, grid).exp().numpy(),
                levels=levels, cmap=cmap)


def _square(lo, hi):
    c = 0.5 * (lo + hi)
    h = 0.5 * (hi - lo).max().clamp_min(1e-6)
    return c - h, c + h


def common_view(records, target_pts=None, instance: int = 0, q: float = 0.005):
    """One shared view box for every plot of a rollout.
    """
    prop_pts = torch.cat(
        [r.samples[instance][:, :2].detach().cpu() for r in records]
        + [r.prop.mu[instance][:, :2].detach().cpu() for r in records], dim=0)
    lo = torch.quantile(prop_pts.double(), q, dim=0).to(prop_pts.dtype)
    hi = torch.quantile(prop_pts.double(), 1.0 - q, dim=0).to(prop_pts.dtype)
    if target_pts is not None and len(target_pts):
        lo = torch.minimum(lo, target_pts.min(0).values)
        hi = torch.maximum(hi, target_pts.max(0).values)
    pad = 0.05 * (hi - lo).clamp_min(1e-6)
    return _square(lo - pad, hi + pad)


def _bare(ax, lo, hi):
    ax.set_xlim(float(lo[0]), float(hi[0]))
    ax.set_ylim(float(lo[1]), float(hi[1]))
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")


def density_figure(rec, instance: int = 0, title: str = "", grid: int = 220,
                   view=None):
    """One rollout step: the proposal in blue contours, samples in red.

    Args:
        rec: One `EMStepRecord` -- its `.prop` and `.samples`.
        instance: Which target instance in the batch to draw.
        title: unused.
        grid: Points per side of the contour grid, ignored unless `d == 2`.

    Returns:
        The matplotlib figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prop = rec.prop
    mu = prop.mu[instance].detach()
    x = rec.samples[instance].detach()
    d = mu.shape[-1]
    m2, p2 = mu[:, :2].cpu(), x[:, :2].cpu()

    if d == 2:
        L = prop.scale_tril()[instance].detach()
        cov = (L @ L.transpose(-1, -2)).cpu()
        w = prop.log_alpha()[instance].detach().exp().cpu()
        sd = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-12))
        lo, hi = view if view is not None else _square(*_frame([m2 - 2.5 * sd, m2 + 2.5 * sd, p2]))
    else:
        cov = None
        lo, hi = view if view is not None else _square(*_frame([m2, p2]))

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    if cov is not None:
        _gmm_contour(ax, m2, cov, w, lo, hi, grid, cmap="Blues")
    ax.scatter(p2[:, 0].numpy(), p2[:, 1].numpy(), s=10, c="tab:red", alpha=1.0, linewidths=0)
    # Centres plotting
    # ax.scatter(m2[:, 0].numpy(), m2[:, 1].numpy(), s=70, c="deeppink", marker="+", linewidths=1.4)
    _bare(ax, lo, hi)
    fig.tight_layout(pad=0)
    return fig


def target_figure(target, instance: int = 0, title: str = "target", grid: int = 220,
                  n_samples: int = 600, seed: int = 12345, view=None, pts=None):
    """The target plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from amortised_adaptive_is.core.seeding import temp_seed

    if pts is None:
        with temp_seed(seed):
            try:
                pts = target.sample(n_samples).detach()
            except Exception:
                return None

    batch = pts.shape[0]
    d = pts.shape[-1]
    p2 = pts[instance][:, :2].cpu()
    lo, hi = view if view is not None else _square(*_frame([p2]))

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    if d == 2:
        base = getattr(target, "_target", target)
        gx = torch.linspace(float(lo[0]), float(hi[0]), grid)
        gy = torch.linspace(float(lo[1]), float(hi[1]), grid)
        G = torch.stack(torch.meshgrid(gx, gy, indexing="xy"), dim=-1).reshape(-1, 2)
        with torch.no_grad():
            lp = base.log_prob(G.unsqueeze(0).expand(batch, -1, -1).to(pts.device))
        ax.contourf(gx.numpy(), gy.numpy(),
                    lp[instance].cpu().reshape(grid, grid).exp().numpy(),
                    levels=50, cmap="Greys")
    # ax.scatter(p2[:, 0].numpy(), p2[:, 1].numpy(), s=4, c="black", alpha=0.75, linewidths=0)
    _bare(ax, lo, hi)
    fig.tight_layout(pad=0)
    return fig


def save_density_pdfs(records, out_dir, stem: str, instance: int = 0, target=None,
                      n_target: int = 600, seed: int = 12345, view=None) -> list[str]:
    """Create and save density plots.
    """
    import matplotlib.pyplot as plt

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    tgt = target if target is not None else getattr(records[0], "target", None)
    tpts = None
    if tgt is not None:
        from amortised_adaptive_is.core.seeding import temp_seed

        with temp_seed(seed):
            try:
                tpts = tgt.sample(n_target).detach()          # (B, n, d), full batch
            except Exception:
                tpts = None

    if view is None:
        view = common_view(records, None if tpts is None else tpts[instance][:, :2].cpu(),
                           instance)
    elif len(view) == 4:
        v = [float(x) for x in view]
        view = (torch.tensor([v[0], v[2]]), torch.tensor([v[1], v[3]]))

    if tpts is not None:
        fig = target_figure(tgt, instance, grid=220, view=view, pts=tpts)
        if fig is not None:
            f = out / f"{stem}_target.pdf"
            fig.savefig(f, format="pdf", bbox_inches="tight")
            plt.close(fig)
            written.append(str(f))
    for t, rec in enumerate(records):
        fig = density_figure(rec, instance, title=f"step {t}", view=view)
        f = out / f"{stem}_t{t:02d}.pdf"
        fig.savefig(f, format="pdf", bbox_inches="tight")
        plt.close(fig)
        written.append(str(f))
    return written


def _draw_from_proposal(prop, instance: int, n: int) -> torch.Tensor:
    """`n` draws from one instance's mixture proposal
    """
    mu = prop.mu[instance].detach() # (N, d)
    L = prop.scale_tril()[instance].detach() # (N, d, d)
    a = prop.log_alpha()[instance].detach().exp() # (N,)
    a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if float(a.sum()) <= 0.0: # if every weight was NaN, use uniform
        a = torch.ones_like(a)
    idx = torch.multinomial(a, num_samples=n, replacement=True)
    z = torch.randn(n, mu.shape[-1], device=mu.device, dtype=mu.dtype)
    return mu[idx] + torch.einsum("nij,nj->ni", L[idx], z)


def _clean_points(rec, instance: int, resample: bool, n_show: int, seed: int):
    """`n_show` points from the terminal step.
    """
    if not resample:
        return _draw_from_proposal(rec.prop, instance, n_show).detach()[:, :2].cpu()
    w = torch.softmax(rec.log_w[instance], dim=-1)
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    if float(w.sum()) <= 0.0:
        w = torch.ones_like(w)
    idx = torch.multinomial(w, num_samples=n_show, replacement=True)
    return rec.samples[instance, idx].detach()[:, :2].cpu()


def _clean_frame(point_sets, q: float = 0.005):
    """One quantile box covering every point set given, padded by 5%."""
    both = torch.cat([p for p in point_sets if p is not None and p.numel()], dim=0)
    finite = both[torch.isfinite(both).all(dim=-1)]
    if finite.numel() == 0:
        finite = torch.zeros(1, 2)
    lo = torch.quantile(finite.double(), q, dim=0).to(finite.dtype)
    hi = torch.quantile(finite.double(), 1.0 - q, dim=0).to(finite.dtype)
    pad = 0.05 * (hi - lo).clamp_min(1e-6)
    return lo - pad, hi + pad


KINDS = ("final", "proposal", "target")


def clean_samples_figure(records, target, instance: int = 0, n_show: int = 2000,
                         view=None, seed: int = 12345, q: float = 0.005,
                         kind: str = "final"):
    """scatter plot of terminal samples.

    Args:
        records: `EMStepRecord` list; only the terminal block is drawn.
        target: The target. Required for `kind="target"`.
        instance: Which target instance in the batch.
        n_show: Points on the panel.
        view: `[xmin, xmax, ymin, ymax]`, or None to do it automatically.
        seed: seed for draws.
        q: quantile padding for the frame
        kind: One of `KINDS`.

    Returns:
        The matplotlib figure, or None when `kind="target"` and the target cannot be sampled.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from amortised_adaptive_is.core.seeding import temp_seed

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")

    with temp_seed(seed):
        if kind == "target":
            if target is None or not hasattr(target, "sample"):
                return None
            try:
                pts = target.sample(n_show)[instance].detach()[:, :2].cpu()
            except Exception:
                return None
        else:
            pts = _clean_points(records[-1], instance, kind == "final", n_show, seed)

    if view is not None and len(view) == 4:
        v = [float(x) for x in view]
        lo, hi = torch.tensor([v[0], v[2]]), torch.tensor([v[1], v[3]])
    elif view is not None:
        lo, hi = view
    else:
        lo, hi = _clean_frame([pts], q)

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.scatter(pts[:, 0].numpy(), pts[:, 1].numpy(), s=5, alpha=0.25,
               c="black" if kind == "target" else "tab:red", linewidths=0)
    _bare(ax, lo, hi)
    fig.tight_layout(pad=0)
    return fig


def save_clean_samples(records, out_dir, stem: str, instance: int = 0, target=None,
                       view=None, seed: int = 12345, n_show: int = 2000,
                       formats: tuple[str, ...] = ("pdf", "png"), dpi: int = 300) -> list[str]:
    """Write the three plots: `_samples_final`, `_samples_proposal`, `_samples_target`.
    """
    import matplotlib.pyplot as plt

    from amortised_adaptive_is.core.seeding import temp_seed

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if view is None:
        sets = []
        if target is not None and hasattr(target, "sample"):
            with temp_seed(seed):
                try:
                    sets.append(target.sample(n_show)[instance].detach()[:, :2].cpu())
                except Exception:
                    pass
        if not sets:
            with temp_seed(seed):
                sets = [_clean_points(records[-1], instance, r, n_show, seed)
                        for r in (True, False)]
        view = _clean_frame(sets)

    written = []
    for kind in KINDS:
        fig = clean_samples_figure(records, target, instance=instance, view=view, seed=seed,
                                   n_show=n_show, kind=kind)
        if fig is None:
            continue
        for ext in formats:
            f = out / f"{stem}_samples_{kind}.{ext}"
            fig.savefig(f, format=ext, bbox_inches="tight",
                        **({"dpi": dpi} if ext != "pdf" else {}))
            written.append(str(f))
        plt.close(fig)
    return written


def _panel_grid(n_panels: int):
    import matplotlib.pyplot as plt

    cols = min(3, n_panels)
    rows = math.ceil(n_panels / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)
    return fig, [axes[i // cols][i % cols] for i in range(rows * cols)]


def samples_figure(records, target, modes, title: str, n_panels: int = 6,
                   max_points: int = 4000, n_true: int = 800, final_only: bool = False):
    """Particles across the rollout, coloured by adaptation step, over exact target draws.

    Args:
        records: `EMStepRecord` list, `records[t].samples` is the block drawn at step `t`.
        target: The target, for the grey reference cloud.
        modes: Mode locations `(B, M, d)` or None, fixes the projection plane.
        title: Figure title.
        n_panels: Target instances to show.
        max_points: Cap on plotted particles per panel, subsampled uniformly across steps.
        n_true: Exact target draws per panel.
        final_only: Plot only the terminal block, resampled by its importance weights, instead of every step.

    Returns:
        A matplotlib figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = records[0].samples.shape[-1]
    B = records[0].samples.shape[0]
    n_panels = min(n_panels, B)
    with torch.no_grad():
        gt = target.sample(n_true) if hasattr(target, "sample") else None
    fig, axes = _panel_grid(n_panels)
    T = len(records) - 1
    sc = None
    for i in range(n_panels):
        ax = axes[i]
        basis = _basis_for(modes, i, d, records[0].samples.dtype, records[0].samples.device)
        if gt is not None:
            g = _proj(gt[i].detach(), basis)
            ax.scatter(g[:, 0], g[:, 1], s=5, alpha=0.16, c=_TARGET_C,
                       label="target" if i == 0 else None, zorder=1)
        if final_only:
            rec = records[-1]
            w = torch.softmax(rec.log_w[i], dim=-1)
            idx = torch.multinomial(w, num_samples=min(max_points, 2000), replacement=True)
            pts = _proj(rec.samples[i, idx].detach(), basis)
            ax.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.25, c="#3b7dd8",
                       label="resampled" if i == 0 else None, zorder=2)
        else:
            # One colour per adaptation step.
            per_step = max(1, max_points // len(records))
            for t, rec in enumerate(records):
                x = rec.samples[i].detach()
                if x.shape[0] > per_step:
                    x = x[torch.randperm(x.shape[0])[:per_step]]
                pts = _proj(x, basis)
                sc = ax.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.30,
                                c=[t / max(1, T)] * pts.shape[0], cmap="viridis",
                                vmin=0.0, vmax=1.0, zorder=2)
        # One dotted track per component, with a square at its mean after every adaptation step.
        track = torch.stack([_proposal_arrays(rec.prop)[0][i].detach() for rec in records])
        track = _proj(track, basis) # (T+1, N, 2)

        # NaN breaks a matplotlib line into segments.
        track = torch.where(torch.isfinite(track), track, torch.full_like(track, float("nan")))
        for n in range(track.shape[1]):
            ax.plot(track[:, n, 0], track[:, n, 1], ls=":", lw=0.9, c="black",
                    marker="s", ms=2.6, mfc="black", mec="black", zorder=5,
                    label="component track" if (i == 0 and n == 0) else None)
            # Open circle at the start, so the direction of travel is clear.
            ax.plot(track[0, n, 0], track[0, n, 1], marker="o", ms=6.5, mfc="white",
                    mec="black", mew=1.1, ls="none", zorder=5,
                    label="start" if (i == 0 and n == 0) else None)
        mu, _, _ = _proposal_arrays(records[-1].prop)
        c = _proj(mu[i].detach(), basis)
        ax.scatter(c[:, 0], c[:, 1], s=110, marker="+", c=_CENTRE_C, linewidths=1.8,
                   label="final centres" if i == 0 else None, zorder=9)
        if modes is not None:
            m = _proj(modes[i].detach(), basis)
            ax.scatter(m[:, 0], m[:, 1], s=115, marker="x", c=_MODE_C, linewidths=2.1,
                       label="target modes" if i == 0 else None, zorder=8)
        w = torch.softmax(records[-1].log_w[i], dim=-1)
        ess = float(1.0 / ((w**2).sum() * w.shape[0]))
        ax.set_title(f"b={i}  ESS {ess:.2f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        # Square limits, symmetric about the origin.
        def _finite_max(*ts) -> float:
            best = 0.0
            for t_ in ts:
                if t_ is None:
                    continue
                f = t_[torch.isfinite(t_)]
                if f.numel():
                    best = max(best, float(f.abs().max()))
            return best

        lim = 1.45 * max(_finite_max(m if modes is not None else None, c, track), 1e-6)
        if not math.isfinite(lim) or lim <= 0.0:
            lim = 1.0
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
    for k in range(n_panels, len(axes)):
        axes[k].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, fontsize=7, loc="lower center",
                   ncol=len(handles), bbox_to_anchor=(0.45, -0.02))
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes[:n_panels], fraction=0.03, pad=0.01)
        cb.set_label("adaptation step (0 to T)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=10)
    if sc is None:
        fig.tight_layout()
    return fig


class BaselineWandbLogger:
    """Logs one classical-baseline run, on the rollout step rather than an epoch.
    """

    def __init__(self, project: str, name: str, config: dict[str, Any],
                 mode: str = "online", group: str | None = None,
                 tags: tuple[str, ...] = ()) -> None:
        self.enabled = mode != "disabled"
        self.run = None
        if not self.enabled:
            return
        import wandb
        self.wandb = wandb
        self.run = wandb.init(project=project, name=name, group=group, config=config,
                              mode=mode, tags=list(tags) or None, reinit=True)

        wandb.define_metric("rollout_step")
        wandb.define_metric("*", step_metric="rollout_step")

    def log_rollout(self, records, target, modes, mode_sd: float, title: str) -> None:
        """Log the per-step diagnostics as a curve, and the figures on the terminal step.

        Args:
            records: Per-block records, from `baseline_records` for a baseline result.
            target: The target, for the true-sample overlay in the samples figure.
            modes: Target mode locations, or None for a family without them.
            mode_sd: Target width, used to normalise the proposal-width series.
            title: Figure title.
        """
        if not self.enabled:
            return
        import matplotlib.pyplot as plt

        per_step = step_metrics(records, modes, mode_sd)
        for t, row in enumerate(per_step):
            payload = {f"rollout/{k}": v for k, v in row.items() if k != "step"}
            payload["rollout_step"] = t
            if t == len(per_step) - 1:

                payload |= {f"final/{k}": v for k, v in row.items() if k != "step"}
                try:
                    figs = {
                        "figure/samples": samples_figure(records, target, modes, title),
                        "figure/samples_final": samples_figure(records, target, modes, title,
                                                               final_only=True),
                        "figure/rollout": rollout_figure(records, modes, title),
                    }
                    payload |= {k: self.wandb.Image(v) for k, v in figs.items()}
                    for v in figs.values():
                        plt.close(v)
                except Exception as exc:
                    print(f"  [wandb] figure failed: {type(exc).__name__}: {exc}", flush=True)
            self.wandb.log(payload, step=t)

    def summary(self, data: dict[str, Any]) -> None:
        """Attach the run's scored metrics."""
        if self.enabled:
            self.run.summary.update(data)

    def finish(self) -> None:
        if self.enabled:
            self.run.finish()


class WandbLogger:
    """Logs one training run."""

    def __init__(self, project: str, name: str, config: dict[str, Any],
                 mode: str = "online", group: str | None = None,
                 tags: tuple[str, ...] = (), figure_every: int = 0) -> None:
        self.enabled = mode != "disabled"
        self.figure_every = int(figure_every)
        self.run = None
        if not self.enabled:
            return
        import wandb
        self.wandb = wandb
        self.run = wandb.init(project=project, name=name, group=group, config=config,
                              mode=mode, tags=list(tags) or None, reinit=True)

        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    def on_validate(self, epoch: int, row: dict[str, float], records, modes, mode_sd: float,
                    beta: float, n_epochs: int, target=None) -> None:
        """Log the epoch's curve point, the per-step rollout series, and periodically a figure."""
        if not self.enabled:
            return
        per_step = step_metrics(records, modes, mode_sd)
        payload = {**{f"train/{k}": v for k, v in row.items()}, "train/beta": beta, "epoch": epoch}
        if per_step:
            T = len(per_step) - 1
            payload |= {f"final/{k}": v for k, v in per_step[-1].items() if k != "step"}
            # Five slices through the rollout, on the epoch axis.
            for t in sorted({0, T // 4, T // 2, (3 * T) // 4, T}):
                payload |= {f"rollout_t{t:02d}/{k}": v for k, v in per_step[t].items()
                            if k != "step"}
        want_fig = self.figure_every and (epoch % self.figure_every == 0 or epoch >= n_epochs - 1)
        if want_fig:
            import matplotlib.pyplot as plt

            try:
                figs = {
                    "figure/samples": samples_figure(
                        records, target, modes, f"epoch {epoch}"),
                    "figure/samples_final": samples_figure(
                        records, target, modes, f"epoch {epoch}", final_only=True),
                    "figure/components": rollout_figure(records, modes, f"epoch {epoch}"),
                }
            except Exception as e:
                print(f"  [wandb] figure skipped at epoch {epoch}: {type(e).__name__}: {e}",
                      flush=True)
                figs = {}
            payload |= {k: self.wandb.Image(f) for k, f in figs.items()}
            for f in figs.values():
                plt.close(f)
        self.wandb.log(payload, step=epoch)

    def summary(self, data: dict[str, Any]) -> None:
        if self.enabled:
            self.run.summary.update(data)

    def finish(self) -> None:
        if self.enabled:
            self.run.finish()
