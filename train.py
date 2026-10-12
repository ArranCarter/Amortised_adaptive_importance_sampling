"""Train the learnt sampler on one task and evaluate it, selected by config.

    .venv/bin/python train.py task=gmm5_d5 train.loss=rkl tag=my-run
    .venv/bin/python train.py task=hierarchical_d25 task.n_components=25 train.seed=100 \
        train.delta_scale=sigma wandb.project=aais wandb.figure_every=250
"""

import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from amortised_adaptive_is.core.paths import CONFIG_DIR
from amortised_adaptive_is.core.specs import TaskSpec
from amortised_adaptive_is.learnt.pipeline import train_and_score
from amortised_adaptive_is.learnt.train import TrainConfig


def _device(name: str) -> torch.device:
    """Get device
    """
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _dtype(name: str) -> torch.dtype:
    try:
        return {"float32": torch.float32, "float64": torch.float64}[name]
    except KeyError:
        raise ValueError(f"dtype must be float32 or float64, got {name!r}") from None

@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="train")
def main(cfg: DictConfig) -> None:
    if os.environ.get("AAIS_DETECT_ANOMALY") == "1":
        torch.autograd.set_detect_anomaly(True)
        print("  [anomaly] autograd anomaly detection ON", flush=True)
    """Train and score one cell."""
    if cfg.threads:
        torch.set_num_threads(int(cfg.threads))
    device, dtype = _device(str(cfg.device)), _dtype(str(cfg.dtype))

    spec = TaskSpec(**OmegaConf.to_container(cfg.task, resolve=True))
    train_cfg = TrainConfig(**OmegaConf.to_container(cfg.train, resolve=True))
    label = str(cfg.tag) or f"{spec.name}-{train_cfg.loss}-h{train_cfg.hidden_dim}"

    train_and_score(
        spec,
        train_cfg,
        label=label,
        eval_seeds=tuple(cfg.eval.seeds),
        eval_batch_size=int(cfg.eval.batch_size),
        wandb=OmegaConf.to_container(cfg.wandb, resolve=True),
        device=device,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()
