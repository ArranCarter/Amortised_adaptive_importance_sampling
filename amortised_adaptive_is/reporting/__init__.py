"""Reporting and logging utilities.
"""

from collections.abc import Iterable

# The metrics printed by the entry points, in column order.
# Metrics with a `_final` / `_all` are for the final step and all steps pooled respectively.
REPORT_KEYS: tuple[str, ...] = (
    "ess_final",
    "ess_all",
    "sliced_w1",
    "w1_prop",
    "w2_prop",
    "sliced_w1_prop",
    "f0_err_final",
    "f0_err_all",
    "logz_err",
    "logz_err_all",
    "reverse_kl",
    "chisq",
)


def format_table(rows: dict[str, dict[str, float]], keys: Iterable[str] = REPORT_KEYS) -> str:
    """Render metric table.

    Args:
        rows: Mapping from method name to its metrics.
        keys: Metric keys to show, in order.

    Returns:
        The rendered table.
    """
    keys = list(keys)
    name_w = max(len(n) for n in rows) if rows else 4
    header = f"{'method':<{name_w}}  " + "  ".join(f"{k:>12}" for k in keys)
    lines = [header, "-" * len(header)]
    for name, m in rows.items():
        cells = []
        for k in keys:
            v = m.get(k, float("nan"))
            cells.append(f"{v:>12.4g}")
        lines.append(f"{name:<{name_w}}  " + "  ".join(cells))
    return "\n".join(lines)
