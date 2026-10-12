"""Training loop for the update network.
"""

import copy
import math
import time
from dataclasses import asdict, dataclass, field, fields, replace
from functools import partial
from typing import Any

import torch

from amortised_adaptive_is.core.counters import (
    CountingTarget,
    DensityEvalCounter,
    density_eval_counter,
)
from amortised_adaptive_is.core.gmm import draw_initial_centres
from amortised_adaptive_is.core.seeding import set_seed, temp_seed
from amortised_adaptive_is.core.specs import TaskSpec
from amortised_adaptive_is.core.targets import build_target
from amortised_adaptive_is.learnt.net import EMUpdateNet
from amortised_adaptive_is.learnt.objectives import (
    NEEDS_TRUE_SAMPLES,
    block_loss,
    trajectory_loss,
)


@dataclass
class TrainConfig:
    """Dataclass for a training run.

    Attributes:
        loss: Objective name, see `objectives.LOSSES`.
        step_weighting: How the per-step losses are combined, see `objectives.step_weights`.
        cov: `iso`, `diag` or `full` covariance.
        learn_logits: also adapt the mixture weights. Requires `sampling="categorical"`.
        sampling: `stratified` or `categorical`.
        use_grad_input: feed the target gradient at each particle to the network.
        anneal_beta0: Inverse temperature at the start of training for annealing.
        anneal_frac: Fraction of training over which `anneal_beta0` reaches 1.
        anneal_shape: `linear` or `sqrt`.
        checkpoint_every: checkpoint frequency.
        max_delta_logit: Bound on the per-step mixture-logit update.
        hidden_dim: Network width.
        use_log_pi: Feed the target log density to the network.
        use_moments: Feed the EM sufficient statistics to the network.
        use_global_w: Feed globally-normalised particle weights.
        delta_scale: Length unit for the mean update.
        feat_compress: How unbounded log-ratio features are squashed for the network, 'clip' or 'asinh'.
        feat_scale: The divisor in that transform.
        scale_free: Express every feature and the mean update relative to the component's own covariance or the ensemble scale.
        max_delta_mu: Bound on the per-step mean update, in `delta_scale` units.
        max_delta_log_sigma: Bound on the per-step log-variance update.
        min_variance: Floor on the proposal variance.
        max_variance: Ceiling on the proposal variance.
        detach_between_steps: detach the gradient at each step.
        n_true_samples: Target draws per epoch, for oracle fkl objective (xent).
        val_select: Which quantity picks the kept weights.
        n_epochs: Gradient steps.
        lr: Adam learning rate.
        lr_schedule: `none` or `cosine`.
        lr_min_factor: Floor of the cosine schedule, as a fraction of `lr`.
        clip_grad_norm: Gradient-norm clip.
        train_batch_size: Target instances per epoch.
        val_batch_size: Target instances in the held-out validation set.
        val_every: Validate every this many epochs.
        seed: Seed for model init and training.
        val_seed: Fixed seed for the validation draw.
        print_every: logging interval.
    """

    loss: str = "rkl"
    step_weighting: str = "uniform"

    cov: str = "diag"
    learn_logits: bool = False
    sampling: str = "stratified"
    use_grad_input: bool = False
    max_delta_logit: float = 1.0
    hidden_dim: int = 32
    use_log_pi: bool = True
    use_moments: bool = True
    use_global_w: bool = True
    delta_scale: str = "sigma"

    max_delta_mu: float = 1.0
    max_delta_log_sigma: float = 0.5
    min_variance: float = 1e-4
    max_variance: float = float("inf")
    detach_between_steps: bool = False

    n_true_samples: int = 256
    val_select: str = "loss"

    n_epochs: int = 1000
    lr: float = 3e-3
    lr_schedule: str = "none"
    lr_min_factor: float = 0.05
    feat_compress: str = "clip"
    feat_scale: float = 5.0
    scale_free: bool = True
    anneal_beta0: float = 1.0
    anneal_frac: float = 0.5
    anneal_shape: str = "linear"
    checkpoint_every: int = 0
    clip_grad_norm: float = 1.0
    train_batch_size: int = 32
    val_batch_size: int = 64
    val_every: int = 25
    seed: int = 100
    val_seed: int = 20000
    print_every: int = 100

    def rollout_kwargs(self) -> dict[str, Any]:
        """Arguments for `rollout_em`.
        """
        return {
            "sampling": self.sampling,
            "use_grad_input": self.use_grad_input,
            "max_delta_mu": self.max_delta_mu,
            "max_delta_log_sigma": self.max_delta_log_sigma,
            "max_delta_logit": self.max_delta_logit,
            "min_variance": self.min_variance,
            "max_variance": self.max_variance,
            "detach_between_steps": self.detach_between_steps,
        }


@dataclass
class TrainResult:
    """Outcome of a training run.

    Attributes:
        net: The best-by-validation network, in eval mode.
        history: Records of epoch, train loss, val loss, val ESS, and cumulative target evaluations.
        best_epoch: Epoch of the selected checkpoint.
        diverged: True when the run did not produce a usable result -- the weights went non-finite mid-run.
        nan_abort_epoch: Epoch at which training was cut short because the weights had gone non-finite, or -1 if that never happened.
        best_val_ess: Held-out ESS at the selected checkpoint.
        best_val_xent: Held-out cross-entropy at the selected checkpoint.
        target_evals: Target density evaluations used by training.
        seconds: Wall-clock training time.
        snapshots: Periodic `(epoch, target_evals, state_dict)` triples when `checkpoint_every > 0`.
    """

    net: EMUpdateNet
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1
    diverged: bool = False
    nan_abort_epoch: int = -1
    best_val_ess: float = 0.0
    best_val_xent: float = float("inf")
    target_evals: int = 0
    seconds: float = 0.0
    snapshots: list = field(default_factory=list)


def build_net(spec: TaskSpec, cfg: TrainConfig) -> torch.nn.Module:
    """Instantiate the update network.

    Args:
        spec: Task specification.
        cfg: Training config.

    Returns:
        An untrained `EMUpdateNet`.
    """
    if cfg.learn_logits and cfg.sampling != "categorical":
        raise ValueError(
            "learn_logits=True needs sampling='categorical'."
        )
    return EMUpdateNet(
        d=spec.spatial_dim,
        hidden_dim=cfg.hidden_dim,
        cov=cfg.cov,
        learn_logits=cfg.learn_logits,
        use_log_pi=cfg.use_log_pi,
        use_moments=cfg.use_moments,
        use_global_w=cfg.use_global_w,
        delta_scale=cfg.delta_scale,
        use_grad=cfg.use_grad_input,
        feat_compress=cfg.feat_compress,
        feat_scale=cfg.feat_scale,
        scale_free=cfg.scale_free,
    )


def make_learnt_runner(net, cfg: TrainConfig):
    """Bind a trained network into the `runner` signature for evaluation.

    Args:
        net: Trained update network.
        cfg: The config it was trained under.

    Returns:
        A callable `(target, init_centres, init_variances, n_steps, k_particles) -> dict`.
    """
    from amortised_adaptive_is.learnt.rollout import run_learnt_em

    return partial(run_learnt_em, net=net, **cfg.rollout_kwargs())


def unroll_for(cfg: TrainConfig, spec: TaskSpec, net, target, init, n_steps: int | None = None,
               k_particles: int | None = None):
    """Run one trajectory of the EM rollout.

    Args:
        cfg: Training config.
        spec: Task spec (supplies T and K).
        net: The update network.
        target: Batched target.
        init: Tuple of (init_centres, init_variances).
        n_steps: Override the horizon for this rollout.
        k_particles: Override the particles per component.

    Returns:
        Tuple of (records, result dict).
    """
    steps = spec.n_steps if n_steps is None else int(n_steps)
    kp = spec.k_particles if k_particles is None else int(k_particles)
    from amortised_adaptive_is.core.proposal import isotropic
    from amortised_adaptive_is.learnt.rollout import rollout_em

    return rollout_em(
        target,
        net,
        isotropic(init[0], init[1]),
        steps,
        spec.n_components * kp,
        **cfg.rollout_kwargs(),
    )


def _initial_proposal(spec: TaskSpec, n: int, device, dtype, target=None):
    """Draws the initial proposal for a batch of target instances."""
    centres = draw_initial_centres(
        batch_size=n,
        n_components=spec.n_components,
        spatial_dim=spec.spatial_dim,
        centre_scale=spec.centre_scale,
        centre_sample_type=spec.centre_sample_type,
        device=device,
        dtype=dtype,
    )
    return centres, torch.ones_like(centres) * float(spec.init_variance)


def _mean_ess(records) -> float:
    """Mean normalised ESS of the final block."""
    lw = records[-1].log_w
    S = lw.shape[1]
    w = torch.softmax(lw, dim=-1)
    return float((1.0 / ((w**2).sum(dim=-1) * S)).mean())


class TemperedTarget:
    """A tempered target.

    Attributes:
        target: The wrapped target.
        beta: Inverse temperature in (0, 1].
    """

    def __init__(self, target, beta: float) -> None:
        self.target = target
        self.beta = float(beta)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.beta * self.target.log_prob(x)

    def sample(self, n_samples: int) -> torch.Tensor:
        raise NotImplementedError(
            "Cannot sample from the tempered target."
        )


def anneal_beta(epoch: int, cfg: "TrainConfig") -> float:
    """Inverse temperature for this epoch, from `anneal_beta0` to 1 over `anneal_frac`.

    Args:
        epoch: Zero-based epoch index.
        cfg: The training config.

    Returns:
        Beta in (0, 1].
    """
    if cfg.anneal_beta0 >= 1.0:
        return 1.0
    span = max(1.0, cfg.anneal_frac * cfg.n_epochs)
    frac = min(1.0, epoch / span)
    if cfg.anneal_shape == "linear":
        return float(cfg.anneal_beta0 + (1.0 - cfg.anneal_beta0) * frac)
    if cfg.anneal_shape == "sqrt":
        r = math.sqrt(cfg.anneal_beta0)
        return float((r + (1.0 - r) * frac) ** 2)
    raise ValueError(f"anneal_shape must be 'linear' or 'sqrt', got {cfg.anneal_shape!r}")


def train(
    spec: TaskSpec,
    cfg: TrainConfig,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    verbose: bool = True,
    on_validate=None,
) -> TrainResult:
    """Train an update network on one task.

    Args:
        spec: Task specification.
        cfg: Training hyperparameters.
        device: Defaults to CPU.
        dtype: Defaults to float32.
        verbose: Print progress.
        on_validate: Optional `(epoch, row, records, val_target, beta) -> None`, called at every validation. Used for W&B logging.

    Returns:
        The training outcome, carrying the weights of the selected model.
    """
    device = device or torch.device("cpu")
    set_seed(cfg.seed)

    net = build_net(spec, cfg).to(device=device, dtype=dtype)

    optimiser = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    if cfg.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=max(1, cfg.n_epochs), eta_min=cfg.lr * cfg.lr_min_factor
        )
    elif cfg.lr_schedule == "none":
        scheduler = None
    else:
        raise ValueError(f"lr_schedule must be 'none' or 'cosine', got {cfg.lr_schedule!r}")
    counter = DensityEvalCounter()

    # Held-out target instances, fixed for the whole run. Drawn with val_seed.
    with temp_seed(cfg.val_seed):
        val_target = build_target(
            spec.target_cfg,
            batch_size=cfg.val_batch_size,
            device=device,
            dtype=dtype,
            n_components=spec.n_components,
            spatial_dim=spec.spatial_dim,
        )
        val_init = _initial_proposal(spec, cfg.val_batch_size, device, dtype, val_target)

        val_true = val_target.sample(cfg.n_true_samples)

    def validate() -> dict[str, float]:
        """Score the held-out set. Returns loss, ESS and cross-entropy to target draws.
        """
        net.eval()
        # use a throwaway counter so the validation cost is not counted in the training cost.
        throwaway = DensityEvalCounter()
        with temp_seed(cfg.val_seed + 1), torch.no_grad(), density_eval_counter(throwaway):
            records, _ = unroll_for(cfg, spec, net, val_target, val_init)
            loss, _ = trajectory_loss(
                records, cfg.loss, cfg.step_weighting, true_samples=val_true,
            )
            xent = (float("nan") if val_true is None
                    else float(block_loss(records[-1], "xent", val_true).mean()))
        net.train()

        return {"loss": float(loss), "ess": _mean_ess(records), "xent": xent}, records

    # Higher is better for ess, lower is better for loss and xent.
    sign = 1.0 if cfg.val_select == "ess" else -1.0

    result = TrainResult(net=net)
    best_state = copy.deepcopy(net.state_dict())
    best_score = -float("inf")
    # Epoch 0 is zero-initialised, so dont start selection immediately.
    selection_warmup = max(1, cfg.val_every)

    t0 = time.time()
    net.train()
    with density_eval_counter(counter):
        for epoch in range(cfg.n_epochs):
            target = build_target(
                spec.target_cfg,
                batch_size=cfg.train_batch_size,
                device=device,
                dtype=dtype,
                n_components=spec.n_components,
                spatial_dim=spec.spatial_dim,
            )
            # steps=None means the task's full T is used.
            steps = None
            epoch_spec = spec

            # draw initial proposal
            init = _initial_proposal(epoch_spec, cfg.train_batch_size, device, dtype, target)

            # Draw target samples for the loss if needed.
            needs_true = cfg.loss in NEEDS_TRUE_SAMPLES
            true_samples = target.sample(cfg.n_true_samples) if needs_true else None

            # Annealing
            beta = anneal_beta(epoch, cfg)
            train_target = CountingTarget(target, counter)
            if beta < 1.0:
                train_target = TemperedTarget(train_target, beta)

            # run the rollout and compute the loss
            records, _ = unroll_for(
                cfg, epoch_spec, net, train_target, init, n_steps=steps
            )
            loss, _ = trajectory_loss(
                records,
                cfg.loss,
                cfg.step_weighting,
                true_samples=true_samples,
            )

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=cfg.clip_grad_norm)
            optimiser.step()
            if scheduler is not None:
                scheduler.step()

            if cfg.checkpoint_every and (
                epoch % cfg.checkpoint_every == 0 or epoch == cfg.n_epochs - 1
            ):
                result.snapshots.append(
                    (epoch, int(counter.target_log_prob_evals), copy.deepcopy(net.state_dict()))
                )

            if epoch % cfg.val_every == 0 or epoch == cfg.n_epochs - 1:
                val, val_records = validate()
                result.history.append(
                    {
                        "epoch": float(epoch),
                        "train_loss": float(loss.detach()),
                        "val_loss": val["loss"],
                        "val_ess": val["ess"],
                        "val_xent": val["xent"],
                        "target_evals": float(counter.target_log_prob_evals),
                    }
                )
                if on_validate is not None:
                    on_validate(epoch, result.history[-1], val_records, val_target, beta)

                # perform selection of models based on validation score.
                score = sign * val[cfg.val_select]

                # keep track of the best so far
                if epoch >= selection_warmup and score > best_score:
                    best_score = score
                    result.best_epoch = epoch
                    result.best_val_ess = val["ess"]
                    result.best_val_xent = val["xent"]
                    best_state = copy.deepcopy(net.state_dict())

                # Print progress
                if verbose and cfg.print_every and epoch % cfg.print_every == 0:
                    print(
                        f"  epoch {epoch:5d}  train {float(loss.detach()):10.4f}  "
                        f"val {val['loss']:10.4f}  ess {val['ess']:.4f}  "
                        f"xent {val['xent']:8.3f}  "
                        f"evals {counter.target_log_prob_evals / 1e6:.1f}M",
                        flush=True,
                    )

                # Stop training if the network parameters are non-finite, and log it.
                if not math.isfinite(val["loss"]) and not all(
                    torch.isfinite(p).all() for p in net.parameters()
                ):
                    result.nan_abort_epoch = epoch
                    result.diverged = True
                    print(
                        f"ABORT: weights non-finite at epoch {epoch}/{cfg.n_epochs}. "
                        f"Selected checkpoint: epoch {result.best_epoch}.",
                        flush=True,
                    )
                    break

    if result.best_epoch < 0:
        result.diverged = True
        n_nan = sum(1 for h in result.history if h["val_loss"] != h["val_loss"])
        print(
            f"WARNING: no checkpoint selected (best_epoch=-1); returning the UNTRAINED network. "
            f"{n_nan}/{len(result.history)} validations were non-finite.",
            flush=True,
        )

    net.load_state_dict(best_state)
    net.eval()
    result.net = net
    result.target_evals = int(counter.target_log_prob_evals)
    result.seconds = time.time() - t0
    return result


def save(
    path,
    net: EMUpdateNet,
    cfg: TrainConfig,
    spec: TaskSpec,
    extra: dict | None = None,
    snapshots: list | None = None,
) -> None:
    """Write a checkpoint with everything needed to rebuild and run the model.

    Args:
        path: Destination file.
        net: Trained network.
        cfg: Training config used.
        spec: Task the model was trained for.
        extra: Optional extras (e.g. final scores).
        snapshots: Optional `(epoch, target_evals, state_dict)` triples from `TrainResult.snapshots`.
    """
    torch.save(
        {
            "state_dict": net.state_dict(),
            "train_cfg": asdict(cfg),
            "task": spec.name,
            "spec": {k: v for k, v in spec.__dict__.items()},
            "extra": extra or {},
            **({"snapshots": snapshots} if snapshots else {}),
        },
        str(path),
    )


# for loading old stored results that have fields no longer defined
_RETIRED_ANNOUNCED: set[tuple[str, str]] = set()

def from_stored(cls, stored, *, source: str = ""):
    """Rebuild a dataclass from a stored record, dropping fields the code no longer defines.
    """
    names = {f.name for f in fields(cls)}
    retired = sorted(set(stored) - names)
    if retired:
        key = (cls.__name__, ",".join(retired))
        if key not in _RETIRED_ANNOUNCED:
            _RETIRED_ANNOUNCED.add(key)
            where = f" in {source}" if source else ""
            print(f"[migrate] {cls.__name__}{where}: ignoring retired field(s) "
                  f"{', '.join(retired)}")
    return cls(**{k: v for k, v in stored.items() if k in names})


def load(path, device: torch.device | None = None, dtype: torch.dtype = torch.float32):
    """Rebuild a network and its config from a checkpoint written by `save`.

    Args:
        path: Checkpoint file.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        Tuple of (net in eval mode, TrainConfig, load dict).
    """
    load_dict = torch.load(str(path), map_location=device or "cpu", weights_only=False)

    cfg = from_stored(TrainConfig, load_dict["train_cfg"], source=str(path))
    spec = from_stored(TaskSpec, load_dict["spec"], source=str(path))
    net = build_net(spec, cfg).to(device=device or "cpu", dtype=dtype)
    net.load_state_dict(load_dict["state_dict"])
    net.eval()
    return net, cfg, load_dict
