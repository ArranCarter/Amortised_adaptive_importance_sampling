"""Where to read configs from, and where to write results to.
"""

import json
import os
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("AAIS_RESULTS_DIR", "results"))

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
CONFIG_DIR = Path(os.environ.get("AAIS_CONFIG_DIR", _DEFAULT_CONFIG_DIR))


def sampler_config(name: str) -> Path:
    """The config file for one sampler.

    Args:
        name: Sampler name, matching a file under `configs/sampler/`.

    Returns:
        Path to the YAML.
    """
    path = CONFIG_DIR / "sampler" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"no config for sampler {name!r} at {path}. "
            + (f"The config tree at {CONFIG_DIR} does not exist, set AAIS_CONFIG_DIR to point at the repository's configs/ directory."
               if not CONFIG_DIR.is_dir()
               else f"Known samplers: "
                    f"{sorted(p.stem for p in (CONFIG_DIR / 'sampler').glob('*.yaml'))}.")
        )
    return path


def per_instance_dir(task: str) -> Path:
    """Directory holding one task's per-instance metric vectors."""
    return RESULTS_DIR / "per_instance" / task


def write_per_instance(task: str, key: str, vectors: dict) -> Path:
    """Record one run's per-instance metric vectors.

    Args:
        task: Task name, e.g. `blr_d10`.
        key: Run identity within the task, e.g. a method name or `learnt/<label>`.
        vectors: Metric name -> one value per target instance, in seed order.

    Returns:
        The file written.
    """
    d = per_instance_dir(task)
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{key.replace('/', '__')}.json"
    f.write_text(json.dumps(vectors))
    return f


def read_per_instance(task: str) -> dict:
    """Every recorded run for one task."""
    d = per_instance_dir(task)
    if not d.is_dir():
        return {}
    return {f.stem.replace("__", "/"): json.loads(f.read_text()) for f in sorted(d.glob("*.json"))}
