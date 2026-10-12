"""The target density evaluation costs: what one instance of each method costs.
"""



def cost(method: str, n: int, k: int, t: int, d: int) -> int:
    """Cost for one instance, in number of density evaluations.

    Args:
        method: One of `df`, `gapis`, `gapis_fixed`, `learnt`.
        n: Proposal components.
        k: Particles per component per step.
        t: Adaptation steps.
        d: Dimension.

    Returns:
        Total cost.
    """
    blocks = (t + 1) * n * k # every method draws T+1 blocks and weights them
    if method == "df":
        return blocks
    if method == "gapis":
        # One grad-only call at the old means, then one grad plus Hessian call at the new ones.
        return blocks + n * ((4 + d) + (t - 1) * (2 + d))
    if method == "gapis_fixed":
        # With covariance adaptation off there is no second derivative call, so a step costs one
        # gradient per component.
        return blocks + 2 * n * t
    if method == "learnt":
        # One forward and one backward at every point whose gradient the features need: all n * k
        # particles plus the n component means, at each of the T + 1 blocks.
        return 2 * (t + 1) * n * (k + 1)
    if method == "learnt_nograd":
        # for use_grad_input=False
        return blocks
    raise ValueError(f"unknown method {method!r}")
