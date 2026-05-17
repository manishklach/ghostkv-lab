"""Metrics utilities for GhostKV experiments."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


def percentile(values: Iterable[float], q: float) -> float:
    """Compute a percentile using NumPy."""
    return float(np.percentile(list(values), q))


def summarize_metrics(metrics: Mapping[str, Iterable[float] | float]) -> dict[str, float]:
    """Summarize scalar and vector metrics into a flat dictionary."""
    summary: dict[str, float] = {}
    for key, value in metrics.items():
        if np.isscalar(value):
            summary[key] = float(value)
            continue

        values = np.asarray(list(value), dtype=float)
        if values.size == 0:
            continue
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_p50"] = percentile(values, 50)
        summary[f"{key}_p95"] = percentile(values, 95)
    return summary


def pretty_print_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    """Render a simple aligned text table."""
    widths = {column: len(column) for column in columns}
    for row in rows:
        for column in columns:
            widths[column] = max(widths[column], len(str(row.get(column, ""))))

    header = " | ".join(column.ljust(widths[column]) for column in columns)
    divider = "-+-".join("-" * widths[column] for column in columns)
    lines = [header, divider]
    for row in rows:
        lines.append(
            " | ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns)
        )
    return "\n".join(lines)

