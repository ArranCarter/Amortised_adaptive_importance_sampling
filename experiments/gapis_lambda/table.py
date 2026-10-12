"""Read the swept step-size table.
"""

import json
import pathlib

TUNED_PATH = pathlib.Path(__file__).with_name("tuned_lam.json")


def load_tuned() -> dict[str, float]:
    """Task name -> swept `lam`.
    """
    if not TUNED_PATH.exists():
        raise FileNotFoundError(
            f"no gapis_lam table at {TUNED_PATH}."
        )
    return {k: float(v["lam"]) for k, v in json.loads(TUNED_PATH.read_text()).items()}


def overrides_for(task: str) -> dict[str, dict]:
    """`sampler_overrides` carrying the swept `lam`.
    """
    tuned = load_tuned()
    if task not in tuned:
        raise KeyError(
            f"no swept gapis_lam for {task!r}. The table at {TUNED_PATH} holds {len(tuned)} values."
        )
    return {"gapis": {"lam": tuned[task]}}
