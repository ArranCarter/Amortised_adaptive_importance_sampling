"""Everything that runs and evaluates a sampler.

A runner is any callable:
    runner(target, init_centres, init_variances, n_steps, k_particles) -> dict
"""
