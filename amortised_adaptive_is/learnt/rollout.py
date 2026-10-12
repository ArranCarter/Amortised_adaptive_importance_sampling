"""Adaptation loop over the proposal.
"""

from dataclasses import dataclass

import torch

from amortised_adaptive_is.core.counters import get_active_density_eval_counter
from amortised_adaptive_is.core.proposal import Proposal, log_density_components, sample


def log_pi_grad(target, x: torch.Tensor) -> torch.Tensor:
    """Gradient of `log pi` at each particle, shape (B, S, d).

    Args:
        target: Target with `log_prob`.
        x: Particles, shape (B, S, d).

    Returns:
        The gradient, detached, shape (B, S, d).
    """
    counter = get_active_density_eval_counter()
    if counter is not None:
        counter.add_target_grad(x)
    x_g = x.detach().requires_grad_(True)
    with torch.enable_grad():
        lp = target.log_prob(x_g)
        (g,) = torch.autograd.grad(lp.sum(), x_g)
    return g.detach()



def log_pi_and_grad(target, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(log pi(x), grad_x log pi(x))` computation at the same time to avoid double density evaluation counting.

    Args:
        target: Target with `log_prob`.
        x: Points, shape (B, S, d).

    Returns:
        `(log_pi, grad)`, shapes (B, S) and (B, S, d).
    """
    counter = get_active_density_eval_counter()
    if counter is not None:
        # The forward is counted by `CountingTarget` inside `log_prob`.
        # This counts the backward.
        counter.add_target_grad(x)
    live = torch.is_grad_enabled() and x.requires_grad
    with torch.enable_grad():
        x_g = x if live else x.detach().requires_grad_(True)
        lp = target.log_prob(x_g)
        (g,) = torch.autograd.grad(lp.sum(), x_g, retain_graph=live)
    return (lp if live else lp.detach()), g.detach()




@dataclass
class EMStepRecord:
    """One sample block, drawn from the proposal before that step's update.

    Attributes:
        samples: Particles, shape (B, S, d).
        comp_idx: Generating component per particle, shape (B, S).
        log_pi: Target log density, shape (B, S).
        log_q: Proposal log density, shape (B, S).
        log_w: `log_pi - log_q`, shape (B, S).
        prop: The generating proposal.
        target: The target this block was drawn against.
        grad_log_pi: Gradient of `log pi` at each particle, shape (B, S, d), or None.
        grad_mu: Gradient of `log pi` at each component mean, shape (B, N, d), or None.
    """

    samples: torch.Tensor
    comp_idx: torch.Tensor
    log_pi: torch.Tensor
    log_q: torch.Tensor
    log_w: torch.Tensor
    prop: Proposal
    target: object | None = None
    grad_log_pi: torch.Tensor | None = None
    grad_mu: torch.Tensor | None = None


def rollout_em(
    target,
    net,
    init_prop: Proposal,
    n_steps: int,
    n_particles: int,
    sampling: str = "stratified",
    use_grad_input: bool = False,
    max_delta_mu: float = 1.0,
    max_delta_log_sigma: float = 0.5,
    max_delta_logit: float = 1.0,
    min_variance: float = 1e-4,
    max_variance: float = float("inf"),
    detach_between_steps: bool = False,
) -> tuple[list[EMStepRecord], dict]:
    """Adapt the proposal for `n_steps`, emitting `n_steps + 1` blocks.

    Args:
        target: Batched target with `log_prob`.
        net: An `EMUpdateNet`.
        init_prop: Initial proposal.
        n_steps: Number of adaptation steps `T`.
        n_particles: Particles per block `S`.
        sampling: `stratified` or `categorical`.
        use_grad_input: Supply `grad log pi` at each particle to the network.
        max_delta_mu: Bound on the per-step mean update.
        max_delta_log_sigma: Bound on each per-step log-scale update.
        max_delta_logit: Bound on the per-step mixture-logit update.
        min_variance: Floor on the proposal scale.
        max_variance: Ceiling on the proposal scale.
        detach_between_steps: Truncate the gradient at each step boundary.

    Returns:
        Tuple of (records, result dict in the evaluation schema).
    """
    prop = init_prop
    records: list[EMStepRecord] = []

    def draw(prop: Proposal) -> EMStepRecord:
        x, comp_idx = sample(prop, n_particles, mode=sampling)

        grad_x = None
        if use_grad_input:
            log_pi, grad_x = log_pi_and_grad(target, x)
        else:
            log_pi = target.log_prob(x)

        log_q, _ = log_density_components(prop, x)

        return EMStepRecord(
            samples=x,
            comp_idx=comp_idx,
            log_pi=log_pi,
            log_q=log_q,
            log_w=log_pi - log_q,
            prop=prop,
            target=target,
            grad_log_pi=grad_x if use_grad_input else None,
            grad_mu=log_pi_grad(target, prop.mu) if use_grad_input else None,
        )

    for t in range(n_steps):
        rec = draw(prop)
        records.append(rec)
        kwargs = {
            "t":float(t) / max(1, n_steps - 1),
            "max_delta_mu":max_delta_mu,
            "max_delta_log_sigma":max_delta_log_sigma,
            "max_delta_logit":max_delta_logit,
        }
        
        cut = detach_between_steps
        out = net(
            prop.detach() if cut else prop,
            rec.samples.detach() if cut else rec.samples,
            rec.comp_idx,
            rec.log_w.detach() if cut else rec.log_w,
            rec.log_pi.detach() if cut else rec.log_pi,
            rec.grad_log_pi,
            rec.grad_mu,
            **kwargs,
        )
        prop = out.clamp_scale(min_variance, max_variance)

    final = draw(prop)
    records.append(final)

    result = {
        "centres": prop.mu,
        "variances": prop.variances(),
        "log_component_logits": prop.log_alpha(),
        "scale_tril": prop.scale_tril(),
        "all_samples": torch.cat([r.samples for r in records], dim=1),
        "all_log_w": torch.cat([r.log_w for r in records], dim=1),
        "all_log_pi": torch.cat([r.log_pi for r in records], dim=1),
        "all_log_q": torch.cat([r.log_q for r in records], dim=1),
        "final_samples": final.samples,
        "final_log_w": final.log_w,
        "final_log_pi": final.log_pi,
        "final_log_q": final.log_q,
        "final_proposal": prop,
    }
    return records, result


@torch.no_grad()
def run_learnt_em(
    target,
    net,
    init_centres: torch.Tensor,
    init_variances: torch.Tensor,
    n_steps: int,
    k_particles: int,
    **kwargs,
) -> dict:
    """
    Args:
        target: Batched target with `log_prob`.
        net: An `EMUpdateNet`.
        init_centres: Initial means, shape (B, N, d).
        init_variances: Initial variances, shape (B, N, d).
        n_steps: Adaptation steps.
        k_particles: Particles per component per step. Block size is `N * k_particles`.
        **kwargs: Passed to `rollout_em`.

    Returns:
        Result dict in the evaluation schema.
    """
    from amortised_adaptive_is.core.proposal import isotropic

    was_training = net.training
    net.eval()
    try:
        _records, result = rollout_em(
            target,
            net,
            isotropic(init_centres, init_variances),
            n_steps,
            int(init_centres.shape[1]) * int(k_particles),
            **kwargs,
        )
    finally:
        net.train(was_training)
        _ = was_training

    result["per_step"] = [
        {"centres": r.prop.mu.detach(),
         "variances": r.prop.log_diag.detach().exp().pow(2),
         "log_component_logits": r.prop.logits.detach(),
         "scale_tril": r.prop.scale_tril().detach(),
         "samples": r.samples.detach(),
         "log_w": r.log_w.detach()}
        for r in _records
    ]
    return result
