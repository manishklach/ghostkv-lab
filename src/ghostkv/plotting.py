"""Headless plotting helpers for GhostKV synthetic experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _rows_from_input(df_or_rows: Any) -> list[dict[str, Any]]:
    """Normalize either a dataframe-like object or row dictionaries."""
    if hasattr(df_or_rows, "to_dict"):
        try:
            return list(df_or_rows.to_dict(orient="records"))
        except TypeError:
            pass
    return [dict(row) for row in df_or_rows]


def _prepare_output(output_path: str | Path) -> Path:
    """Ensure the output directory exists before saving a figure."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def plot_sketch_dim_vs_topk_overlap(df_or_rows: Any, output_path: str | Path) -> None:
    """Plot sketch dimension against top-k overlap."""
    rows = _rows_from_input(df_or_rows)
    sketch_dims = np.array([float(row["sketch_dim"]) for row in rows], dtype=float)
    topk_overlap = np.array([float(row["top32_overlap"]) for row in rows], dtype=float)

    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sketch_dims, topk_overlap, marker="o", linewidth=2)
    ax.set_title("Sketch Dimension vs Top-32 Overlap")
    ax.set_xlabel("Sketch Dimension")
    ax.set_ylabel("Top-32 Overlap")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_theta_vs_elimination_rate(df_or_rows: Any, output_path: str | Path) -> None:
    """Plot elimination rate across theta values for each sketch dimension."""
    rows = _rows_from_input(df_or_rows)
    output = _prepare_output(output_path)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sketch_dims = sorted({int(row["sketch_dim"]) for row in rows})
    for sketch_dim in sketch_dims:
        dim_rows = sorted(
            (row for row in rows if int(row["sketch_dim"]) == sketch_dim),
            key=lambda row: float(row["theta"]),
        )
        theta = np.array([float(row["theta"]) for row in dim_rows], dtype=float)
        elimination_rate = np.array(
            [float(row["elimination_rate"]) for row in dim_rows],
            dtype=float,
        )
        ax.plot(theta, elimination_rate, marker="o", linewidth=1.8, label=f"sketch={sketch_dim}")

    ax.set_title("Elimination Rate vs Threshold")
    ax.set_xlabel("Elimination Threshold (theta)")
    ax.set_ylabel("Elimination Rate")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_resurrection_rate_vs_bandwidth(df_or_rows: Any, output_path: str | Path) -> None:
    """Plot modeled GhostKV bytes against resurrection rate."""
    rows = _rows_from_input(df_or_rows)
    ghost_rows = [
        row for row in rows if str(row.get("scheme", "")).lower() == "ghostkv"
    ]
    ghost_rows.sort(key=lambda row: float(row["resurrection_rate"]))

    resurrection_rate = np.array([float(row["resurrection_rate"]) * 100.0 for row in ghost_rows])
    ghost_bytes_mib = np.array(
        [float(row["bytes"]) / (1024.0 * 1024.0) for row in ghost_rows],
        dtype=float,
    )

    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(resurrection_rate, ghost_bytes_mib, marker="o", linewidth=2)
    ax.set_title("Resurrection Rate vs GhostKV Bytes Touched")
    ax.set_xlabel("Resurrection Rate (%)")
    ax.set_ylabel("GhostKV Bytes Touched (MiB)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_real_attention_topk_overlap(df_or_rows: Any, output_path: str | Path) -> None:
    """Plot top-k overlap against sketch dimension for each analyzed layer."""
    rows = _rows_from_input(df_or_rows)
    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    layer_ids = sorted({int(row["layer_idx"]) for row in rows})
    for layer_idx in layer_ids:
        layer_rows = sorted(
            (row for row in rows if int(row["layer_idx"]) == layer_idx),
            key=lambda row: float(row["sketch_dim"]),
        )
        sketch_dim = np.array([float(row["sketch_dim"]) for row in layer_rows], dtype=float)
        topk_overlap = np.array([float(row["topk_overlap_mean"]) for row in layer_rows], dtype=float)
        ax.plot(sketch_dim, topk_overlap, marker="o", linewidth=1.8, label=f"layer={layer_idx}")

    ax.set_title("Real Attention Top-k Overlap vs Sketch Dimension")
    ax.set_xlabel("Sketch Dimension")
    ax.set_ylabel("Mean Top-k Overlap")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_real_attention_false_elimination(df_or_rows: Any, output_path: str | Path) -> None:
    """Plot false elimination rate across theta for each analyzed layer."""
    rows = _rows_from_input(df_or_rows)
    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    layer_ids = sorted({int(row["layer_idx"]) for row in rows})
    for layer_idx in layer_ids:
        layer_rows = sorted(
            (row for row in rows if int(row["layer_idx"]) == layer_idx),
            key=lambda row: float(row["theta"]),
        )
        theta = np.array([float(row["theta"]) for row in layer_rows], dtype=float)
        false_elimination = np.array(
            [float(row["false_elimination_rate_mean"]) for row in layer_rows],
            dtype=float,
        )
        ax.plot(theta, false_elimination, marker="o", linewidth=1.8, label=f"layer={layer_idx}")

    ax.set_title("Real Attention False Elimination vs Threshold")
    ax.set_xlabel("Elimination Threshold (theta)")
    ax.set_ylabel("Mean False Elimination Rate")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_layerwise_overlap(df_or_rows: Any, output_path: str | Path) -> None:
    """Plot layer-wise overlap for a fixed sketch configuration."""
    rows = _rows_from_input(df_or_rows)
    rows = sorted(rows, key=lambda row: int(row["layer_idx"]))
    output = _prepare_output(output_path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    layer_idx = np.array([float(row["layer_idx"]) for row in rows], dtype=float)
    topk_overlap = np.array([float(row["topk_overlap_mean"]) for row in rows], dtype=float)
    ax.plot(layer_idx, topk_overlap, marker="o", linewidth=2)
    ax.set_title("Layer-wise Real Attention Top-k Overlap")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean Top-k Overlap")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_head_variance(df_or_rows: Any, output_path: str | Path) -> None:
    """Plot the variance in head-wise overlap for each layer."""
    rows = _rows_from_input(df_or_rows)
    output = _prepare_output(output_path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    layer_ids = sorted({int(row["layer_idx"]) for row in rows})
    variance = []
    for layer_idx in layer_ids:
        layer_rows = [row for row in rows if int(row["layer_idx"]) == layer_idx]
        variance.append(float(np.std([float(row["topk_overlap"]) for row in layer_rows])))
    ax.bar([str(layer_idx) for layer_idx in layer_ids], variance)
    ax.set_title("Head-wise Top-k Overlap Variance")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Std. Dev. Across Heads")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_hierarchical_vs_flat(df_or_rows: Any, output_path: str | Path) -> None:
    """Plot hierarchical versus flat false elimination behavior."""
    rows = _rows_from_input(df_or_rows)
    output = _prepare_output(output_path)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        method_rows = sorted(
            (row for row in rows if str(row["method"]) == method),
            key=lambda row: float(row["theta"]),
        )
        theta = np.array([float(row["theta"]) for row in method_rows], dtype=float)
        false_elimination = np.array(
            [float(row["false_elimination_rate_mean"]) for row in method_rows],
            dtype=float,
        )
        ax.plot(theta, false_elimination, marker="o", linewidth=1.8, label=method)

    ax.set_title("Hierarchical vs Flat False Elimination")
    ax.set_xlabel("Elimination Threshold (theta)")
    ax.set_ylabel("Mean False Elimination Rate")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
