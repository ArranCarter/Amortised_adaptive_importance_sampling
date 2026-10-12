"""Baselines.

    is_methods.py  IS methods
    oracle.py      i.i.d. draws from the target
    registry.py    name -> class
"""

from .registry import ALGORITHM_REGISTRY, build_algorithm

__all__ = ["ALGORITHM_REGISTRY", "build_algorithm"]
