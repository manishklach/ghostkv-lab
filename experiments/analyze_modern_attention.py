# GhostKV Lab — github.com/manishklach/ghostkv-lab
# Patent: IN 202641062451
"""Analyze modern real-attention metrics and update the public results summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = REPO_ROOT / "results" / "real_attention_modern" / "metrics.csv"
OUTPUT_DIR = REPO_ROOT / "results" / "modern"
RESULTS_MD = REPO_ROOT / "RESULTS.md"


def _prepare_output(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output = _prepare_output(output_path)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _layer_summary_table(dataframe: pd.DataFrame, sketch_dim: int = 64, theta: float = 0.2) -> pd.DataFrame:
    filtered = dataframe[
        (dataframe["sketch_dim"] == sketch_dim)
        & np.isclose(dataframe["theta"], theta)
    ]
    if filtered.empty:
        return pd.DataFrame()
    summary = (
        filtered.groupby("layer_idx", as_index=False)[
            ["false_elimination_rate", "elimination_rate", "top32_overlap", "rank_correlation"]
        ]
        .mean()
        .sort_values("layer_idx")
    )
    return summary


def plot_false_elim_vs_theta_by_layer(dataframe: pd.DataFrame, output_path: Path) -> None:
    filtered = dataframe[dataframe["sketch_dim"] == 64]
    summary = (
        filtered.groupby(["layer_idx", "theta"], as_index=False)["false_elimination_rate"]
        .mean()
        .sort_values(["layer_idx", "theta"])
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for layer_idx, layer_rows in summary.groupby("layer_idx"):
        ax.plot(
            layer_rows["theta"],
            layer_rows["false_elimination_rate"],
            marker="o",
            linewidth=1.8,
            label=f"layer={int(layer_idx)}",
        )
    model_id = str(dataframe["model_id"].iloc[0])
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1.0)
    ax.set_title(f"False Elimination Rate vs θ by Layer — {model_id}")
    ax.set_xlabel("Theta")
    ax.set_ylabel("False Elimination Rate")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    _save_figure(fig, output_path)


def plot_elimination_rate_vs_false_elim(dataframe: pd.DataFrame, output_path: Path) -> None:
    filtered = dataframe[np.isclose(dataframe["theta"], 0.2)]
    summary = (
        filtered.groupby(["sketch_dim", "layer_idx", "effective_context_length"], as_index=False)[
            ["elimination_rate", "false_elimination_rate"]
        ]
        .mean()
        .sort_values(["sketch_dim", "elimination_rate"])
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for sketch_dim, dim_rows in summary.groupby("sketch_dim"):
        ax.plot(
            dim_rows["elimination_rate"],
            dim_rows["false_elimination_rate"],
            marker="o",
            linewidth=1.8,
            label=f"sketch={int(sketch_dim)}",
        )
    ax.axhspan(0.0, 0.05, xmin=0.6, xmax=1.0, alpha=0.12, color="green")
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1.0)
    ax.axvline(0.6, color="green", linestyle=":", linewidth=1.0)
    ax.set_title("Elimination Rate vs False Elimination")
    ax.set_xlabel("Elimination Rate")
    ax.set_ylabel("False Elimination Rate")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    _save_figure(fig, output_path)


def plot_topk_overlap_heatmap(dataframe: pd.DataFrame, output_path: Path) -> None:
    filtered = dataframe[
        (dataframe["sketch_dim"] == 64)
        & np.isclose(dataframe["theta"], 0.2)
    ]
    summary = (
        filtered.groupby(["layer_idx", "head_idx"], as_index=False)["top32_overlap"]
        .mean()
        .sort_values(["layer_idx", "head_idx"])
    )
    layers = sorted(summary["layer_idx"].unique().tolist())
    heads = sorted(summary["head_idx"].unique().tolist())
    matrix = np.zeros((len(layers), len(heads)), dtype=float)
    for layer_pos, layer_idx in enumerate(layers):
        for head_pos, head_idx in enumerate(heads):
            matches = summary[(summary["layer_idx"] == layer_idx) & (summary["head_idx"] == head_idx)]
            matrix[layer_pos, head_pos] = float(matches["top32_overlap"].iloc[0]) if not matches.empty else 0.0

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_title("Top-k Overlap Heatmap")
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(len(heads)))
    ax.set_xticklabels([str(head) for head in heads])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(layer) for layer in layers])
    fig.colorbar(image, ax=ax, label="Top-32 Overlap")
    _save_figure(fig, output_path)


def plot_context_length_scaling(dataframe: pd.DataFrame, output_path: Path) -> None:
    summary = (
        dataframe.groupby(["sketch_dim", "effective_context_length"], as_index=False)[
            ["elimination_rate", "false_elimination_rate"]
        ]
        .mean()
        .sort_values(["sketch_dim", "effective_context_length"])
    )
    sketch_dims = sorted(summary["sketch_dim"].unique().tolist())
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), sharex=True, sharey=False)
    axes_flat = list(axes.flatten())
    for axis, sketch_dim in zip(axes_flat, sketch_dims):
        dim_rows = summary[summary["sketch_dim"] == sketch_dim]
        axis.plot(
            dim_rows["effective_context_length"],
            dim_rows["elimination_rate"],
            marker="o",
            linewidth=1.8,
            label="elimination_rate",
        )
        axis.plot(
            dim_rows["effective_context_length"],
            dim_rows["false_elimination_rate"],
            marker="s",
            linewidth=1.8,
            label="false_elim_rate",
        )
        axis.set_title(f"Sketch {int(sketch_dim)}")
        axis.set_xlabel("Context Length")
        axis.set_ylabel("Rate")
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend(frameon=False, fontsize=8)
    for axis in axes_flat[len(sketch_dims):]:
        axis.axis("off")
    _save_figure(fig, output_path)


def _assign_layer_quartiles(layers: list[int]) -> dict[int, str]:
    sorted_layers = sorted(layers)
    labels = ["early", "mid-early", "mid-late", "late"]
    assignments: dict[int, str] = {}
    for position, layer_idx in enumerate(sorted_layers):
        quartile = min(int(position * 4 / max(len(sorted_layers), 1)), 3)
        assignments[layer_idx] = labels[quartile]
    return assignments


def plot_sketch_dim_comparison(dataframe: pd.DataFrame, output_path: Path) -> None:
    filtered = dataframe[np.isclose(dataframe["theta"], 0.1)].copy()
    quartiles = _assign_layer_quartiles(sorted(filtered["layer_idx"].unique().tolist()))
    filtered["layer_quartile"] = filtered["layer_idx"].map(quartiles)
    summary = (
        filtered.groupby(["layer_quartile", "sketch_dim"], as_index=False)["false_elimination_rate"]
        .mean()
    )
    quartile_order = [label for label in ["early", "mid-early", "mid-late", "late"] if label in summary["layer_quartile"].unique()]
    sketch_dims = sorted(summary["sketch_dim"].unique().tolist())

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x_positions = np.arange(len(quartile_order))
    width = 0.18
    for offset_idx, sketch_dim in enumerate(sketch_dims):
        dim_rows = summary[summary["sketch_dim"] == sketch_dim].set_index("layer_quartile")
        heights = [float(dim_rows.loc[label, "false_elimination_rate"]) if label in dim_rows.index else 0.0 for label in quartile_order]
        ax.bar(x_positions + (offset_idx - (len(sketch_dims) - 1) / 2) * width, heights, width=width, label=f"sketch={int(sketch_dim)}")

    ax.set_title("Sketch Dimension Comparison at θ=0.1")
    ax.set_xlabel("Layer Quartile")
    ax.set_ylabel("Mean False Elimination Rate")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(quartile_order)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False, ncol=2)
    _save_figure(fig, output_path)


def _markdown_table(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "_No rows available for the requested slice._"
    header = "| layer_idx | false_elimination_rate | elimination_rate | top32_overlap | rank_correlation |\n| --- | --- | --- | --- | --- |"
    body = []
    for row in dataframe.itertuples(index=False):
        body.append(
            "| "
            + " | ".join(
                [
                    str(int(row.layer_idx)),
                    f"{row.false_elimination_rate:.4f}",
                    f"{row.elimination_rate:.4f}",
                    f"{row.top32_overlap:.4f}",
                    f"{row.rank_correlation:.4f}",
                ]
            )
            + " |"
        )
    return "\n".join([header, *body])


def _modern_section(dataframe: pd.DataFrame, summary_table: pd.DataFrame) -> str:
    model_id = str(dataframe["model_id"].iloc[0])
    requested_model_id = str(dataframe["requested_model_id"].iloc[0])
    contexts = sorted(int(value) for value in dataframe["effective_context_length"].unique().tolist())
    requested_contexts = sorted(int(value) for value in dataframe["requested_context_length"].unique().tolist())
    mean_false_elim = float(dataframe["false_elimination_rate"].mean())
    mean_elim = float(dataframe["elimination_rate"].mean())
    mean_rank = float(dataframe["rank_correlation"].mean())
    mean_top32 = float(dataframe["top32_overlap"].mean())
    max_elim = float(dataframe["elimination_rate"].max())
    max_false_elim = float(dataframe["false_elimination_rate"].max())

    findings = [
        f"- Current modern-capture runs resolved to `{model_id}` (requested `{requested_model_id}`).",
        f"- Mean rank correlation is `{mean_rank:.4f}`, while mean top-32 overlap is `{mean_top32:.4f}`.",
        f"- Mean elimination rate is only `{mean_elim:.6f}` under the current positive-threshold sweep, with a maximum observed elimination rate of `{max_elim:.6f}`.",
        f"- Mean false elimination rate is `{mean_false_elim:.4f}`, but the current run does not reach a meaningful elimination regime; the methodology is currently more informative than the operating point.",
        f"- Requested context lengths were `{requested_contexts}`, with effective lengths `{contexts}` after model-specific truncation.",
        f"- Maximum observed false elimination rate in the captured rows was `{max_false_elim:.4f}`.",
    ]

    return f"""
## Modern Architecture Validation

The modern real-attention pipeline uses architecture-aware hooks to capture raw query/key projections and evaluate sketch preservation on non-synthetic tensors. In this environment, the memory-safe loader fell back to GPT-2, so broader modern-model validation remains pending even though the hook pathway now supports larger decoder families.

### Summary table

{_markdown_table(summary_table)}

### Key findings

{chr(10).join(findings)}

### Plots

- [results/modern/false_elim_vs_theta_by_layer.png](results/modern/false_elim_vs_theta_by_layer.png)
- [results/modern/elimination_rate_vs_false_elim.png](results/modern/elimination_rate_vs_false_elim.png)
- [results/modern/topk_overlap_heatmap.png](results/modern/topk_overlap_heatmap.png)
- [results/modern/context_length_scaling.png](results/modern/context_length_scaling.png)
- [results/modern/sketch_dim_comparison.png](results/modern/sketch_dim_comparison.png)
- [results/real_attention_modern/metrics.csv](results/real_attention_modern/metrics.csv)
- [results/real_attention_modern/summary.json](results/real_attention_modern/summary.json)
""".strip()


def update_results_markdown(results_path: Path, modern_section: str) -> None:
    existing = results_path.read_text(encoding="utf-8") if results_path.exists() else "# GhostKV Results\n"
    header = "## Modern Architecture Validation"
    if header in existing:
        prefix, _, remainder = existing.partition(header)
        next_header_index = remainder.find("\n## ", 1)
        if next_header_index == -1:
            updated = prefix.rstrip() + "\n\n" + modern_section + "\n"
        else:
            suffix = remainder[next_header_index:]
            updated = prefix.rstrip() + "\n\n" + modern_section + suffix
    else:
        updated = existing.rstrip() + "\n\n" + modern_section + "\n"
    results_path.write_text(updated, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--results-md", type=Path, default=RESULTS_MD)
    return parser


def main() -> None:
    plt.style.use("seaborn-v0_8-paper")
    parser = build_parser()
    args = parser.parse_args()
    dataframe = pd.read_csv(args.metrics_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_false_elim_vs_theta_by_layer(dataframe, args.output_dir / "false_elim_vs_theta_by_layer.png")
    plot_elimination_rate_vs_false_elim(dataframe, args.output_dir / "elimination_rate_vs_false_elim.png")
    plot_topk_overlap_heatmap(dataframe, args.output_dir / "topk_overlap_heatmap.png")
    plot_context_length_scaling(dataframe, args.output_dir / "context_length_scaling.png")
    plot_sketch_dim_comparison(dataframe, args.output_dir / "sketch_dim_comparison.png")

    summary_table = _layer_summary_table(dataframe)
    update_results_markdown(args.results_md, _modern_section(dataframe, summary_table))

    print(f"Wrote {args.output_dir}")
    print(f"Updated {args.results_md}")


if __name__ == "__main__":
    main()
