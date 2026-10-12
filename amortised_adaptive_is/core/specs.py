"""Task specifications: one TaskSpec per task (family + proposal specifications).
"""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass
class TaskSpec:
    """Dataclass for a single task specification.

    Attributes:
        name: Task key.
        target: Target config dict.
        n_components: Proposal mixture components `N`.
        spatial_dim: `d`.
        n_steps: Adaptation steps `T`.
        k_particles: Particles per component per step `K`.
        init_variance: Initial proposal variance (isotropic).
        centre_scale: scale of the initial-centre draw.
        centre_sample_type: `uniform` or `gaussian`.
        plot_lims: Axis limits for diagnostic scatter plots.
    """

    name: str
    target: dict[str, Any]
    n_components: int
    spatial_dim: int = 2
    n_steps: int = 15
    k_particles: int = 15
    init_variance: float = 1.0
    centre_scale: float = 4.0
    centre_sample_type: str = "uniform"
    plot_lims: list = field(default_factory=lambda: [[-8.0, 8.0], [-8.0, 8.0]])

    @property
    def target_cfg(self) -> DictConfig:
        """The target config"""
        return OmegaConf.create(dict(self.target))

    def replace(self, **kwargs: Any) -> "TaskSpec":
        """Return a copy with fields overridden."""
        base = dict(self.__dict__)
        base.update(kwargs)
        return TaskSpec(**base)


TASKS: dict[str, TaskSpec] = {
    # gmm5 with every length multiplied by 10.
    "gmm5_x10": TaskSpec(
        name="gmm5_x10",
        target={
            "name": "gmm",
            "n_components": 5,
            "spatial_dim": 2,
            "centre_range": 40.0,
            "target_variance": 50.0,
            "min_sep": 0.0,
        },
        n_components=5,
        init_variance=100.0,
        centre_scale=40.0,
        centre_sample_type="uniform",
        plot_lims=[[-80.0, 80.0], [-80.0, 80.0]],
    ),
}


# -----------------------------------
_DIMS = (2, 5, 25)

for _d in _DIMS:
    TASKS[f"gmm5_d{_d}"] = TaskSpec(
        name=f"gmm5_d{_d}",
        target={
            "name": "gmm",
            "n_components": 5,
            "spatial_dim": _d,
            "centre_range": 4.0,
            "target_variance": 0.5,
            "min_sep": 0.0,
        },
        n_components=5,
        spatial_dim=_d,
        centre_scale=4.0,
        centre_sample_type="uniform",
        plot_lims=[[-8.0, 8.0], [-8.0, 8.0]],
    )
    TASKS[f"corners_d{_d}"] = TaskSpec(
        name=f"corners_d{_d}",
        target={
            "name": "gmm_4_corners",
            "n_components": 2,
            "spatial_dim": _d,
            "centre_range": 4.0,
            "target_variance": 0.2,
            "min_sep": 0.0,
        },
        n_components=2,
        spatial_dim=_d,
        centre_scale=1.0,
        centre_sample_type="gaussian",
        plot_lims=[[-6.0, 6.0], [-6.0, 6.0]],
    )



# -----------------------------
for _d in _DIMS:
    TASKS[f"funnel_d{_d}"] = TaskSpec(
        name=f"funnel_d{_d}",
        target={
            "name": "funnel_nd",
            "spatial_dim": _d,
            "sigma_v_range": [1.5, 3.0],
        },
        n_components=5,
        spatial_dim=_d,
        centre_scale=3.0,
        centre_sample_type="uniform",
        plot_lims=[[-8.0, 8.0], [-8.0, 8.0]],
    )

    # funnel family of one target
    _sv = 2.25
    TASKS[f"funnelfix_d{_d}"] = TASKS[f"funnel_d{_d}"].replace(
        name=f"funnelfix_d{_d}",
        target={**TASKS[f"funnel_d{_d}"].target, "sigma_v_range": [_sv, _sv]},
    )


# ---------------------------------
# bananahard, blr, hierarchical
for _d in _DIMS:
    TASKS[f"bananahard_d{_d}"] = TaskSpec(
        name=f"bananahard_d{_d}",
        target={
            "name": "banana_nd",
            "spatial_dim": _d,
            "b_range": [0.1, 0.5],
            "sigma1_range": [1.0, 3.0],
            "sigma2_range": [0.5, 1.5],
        },
        n_components=5,
        spatial_dim=_d,
        centre_scale=4.0,
        centre_sample_type="uniform",
        plot_lims=[[-8.0, 8.0], [-8.0, 8.0]],
    )

    # single target banana family
    TASKS[f"bananahardfix_d{_d}"] = TASKS[f"bananahard_d{_d}"].replace(
        name=f"bananahardfix_d{_d}",
        target={**TASKS[f"bananahard_d{_d}"].target, "pin_seed": 2000,
                "b_range": [0.3, 0.3], "sigma1_range": [2.0, 2.0],
                "sigma2_range": [1.0, 1.0]},
    )



    TASKS[f"blr_d{_d}"] = TaskSpec(
        name=f"blr_d{_d}",
        target={
            "name": "blr",
            "spatial_dim": _d,
            "n_range": [3 * (_d - 1), 12 * (_d - 1)],
            "cond_log10_range": [0.0, 2.0],
            "beta_scale": 2.0,
            "sigma_range": [0.3, 1.5],
            "prior_precision": 0.1,
            "prior_a": 1.0,
            "prior_b": 1.0,
        },
        n_components=5,
        spatial_dim=_d,
        centre_scale=6.0,
        centre_sample_type="uniform",
        plot_lims=[[-10.0, 10.0], [-10.0, 10.0]],
    )

    # hierarchical normal-normal model.
    TASKS[f"hierarchical_d{_d}"] = TaskSpec(
        name=f"hierarchical_d{_d}",
        target={
            "name": "hierarchical",
            "spatial_dim": _d,
            "sigma_range": [2.0, 18.0],
            "tau_true_range": [0.5, 8.0],
            "mu_true_scale": 5.0,
            "s_mu": 5.0,
            "s_tau": 1.0,
        },
        n_components=5,
        spatial_dim=_d,
        centre_scale=8.0,
        centre_sample_type="uniform",
        plot_lims=[[-12.0, 12.0], [-12.0, 12.0]],
    )

    # single target hierarchical family
    TASKS[f"hierarchicalfix_d{_d}"] = TASKS[f"hierarchical_d{_d}"].replace(
        name=f"hierarchicalfix_d{_d}",
        target={**TASKS[f"hierarchical_d{_d}"].target, "pin_seed": 3000,
                "sigma_range": [10.0, 10.0], "tau_true_range": [4.25, 4.25]},
    )

    # single target blr family
    TASKS[f"blrfix_d{_d}"] = TASKS[f"blr_d{_d}"].replace(
        name=f"blrfix_d{_d}",
        target={**TASKS[f"blr_d{_d}"].target, "pin_seed": 1000,
                "n_range": [15 * (_d - 1) // 2, 15 * (_d - 1) // 2],
                "cond_log10_range": [1.0, 1.0], "sigma_range": [0.9, 0.9]},
    )




# ------------------------------------
for _d in (50,):
    TASKS[f"bananahard_d{_d}"] = TASKS["bananahard_d25"].replace(
        name=f"bananahard_d{_d}",
        spatial_dim=_d,
        target={**TASKS["bananahard_d25"].target, "spatial_dim": _d},
    )
    TASKS[f"funnel_d{_d}"] = TASKS["funnel_d25"].replace(
        name=f"funnel_d{_d}",
        spatial_dim=_d,
        target={**TASKS["funnel_d25"].target, "spatial_dim": _d},
    )


def get_task(name: str, **overrides: Any) -> TaskSpec:
    """Look up a task by name, optionally overriding fields.

    Args:
        name: Key into `TASKS`.
        **overrides: Field overrides applied to a copy.

    Returns:
        The task spec.
    """
    if name not in TASKS:
        raise KeyError(f"Unknown task {name!r}. Available: {sorted(TASKS)}")
    spec = TASKS[name]
    return spec.replace(**overrides) if overrides else spec
