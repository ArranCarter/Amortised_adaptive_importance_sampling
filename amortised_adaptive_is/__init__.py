"""Amortised adaptive importance sampling experiments.
"""

from importlib.metadata import version as _version

__all__ = ["__version__"]

try:
    __version__ = _version("amortised-adaptive-is")
except Exception:
    __version__ = "0.0.0"
