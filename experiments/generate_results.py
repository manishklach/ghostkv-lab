"""Generate synthetic GhostKV CSV results, plots, and a summary markdown file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from ghostkv.bandwidth import (
    bandwidth_reduction_ratio,
    ghostkv_bytes,
    quantized_kv_bytes,
    standard_kv_bytes,
)
from ghostkv.bounds import topk_overlap
from ghostkv.plotting import (
    plot_resurrection_rate_vs_bandwidth,
    plot_sketch_dim_vs_topk_overlap,
    plot_theta_vs_elimination_rate,
)
from ghostkv.simulator import SyntheticGhostKVSimulator
from ghostkv.sketches import (
    cosine_rank_correlation,
    project_keys,
    project_query,
    random_projection_matrix,
    sketch_similarity,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_MD = REPO_ROOT / "RESULTS.md"


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    """Write rows to CSV with stable field ordering."""
    if not rows:
        raise ValueError(f"No rows provided for {path.name}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV rows back as dictionaries for markdown generation."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _markdown_table(rows: Iterable[dict[str, str]], columns: list[str]) -> str:
    """Render a simple markdown table from row dictionaries."""
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def run_sketch_quality_sweep(num_tokens: int, dim: int, seed: int) -> list[dict[str, float | int]]:
    """Generate sketch-quality rows across sketch dimensions."""
    rng = np.random.default_rng(seed)
    keys = rng.normal(size=(num_tokens, dim))
    query = rng.normal(size=(dim,))
    exact_scores = (keys @ query) / np.sqrt(dim)

    rows: list[dict[str, float | int]] = []
    for sketch_dim in (8, 16, 32, 64):
        projection = random_projection_matrix(dim, sketch_dim, seed=seed + sketch_dim)
        key_sketches = project_keys(keys, projection)
        query_sketch = project_query(query, projection)
        approx_scores = sketch_similarity(query_sketch, key_sketches)
        rows.append(
            {
                "sketch_dim": sketch_dim,
                "top8_overlap": round(topk_overlap(exact_scores, approx_scores, 8), 6),
                "top16_overlap": round(topk_overlap(exact_scores, approx_scores, 16), 6),
                "top32_overlap": round(topk_overlap(exact_scores, approx_scores, 32), 6),
                "top64_overlap": round(topk_overlap(exact_scores, approx_scores, 64), 6),
                "rank_correlation": round(cosine_rank_correlation(exact_scores, approx_scores), 6),
            }
        )
    return rows


def run_elimination_tradeoff_sweep(
    num_tokens: int,
    dim: int,
    hot_window: int,
    steps: int,
    epsilon: float,
    sigma: float,
    seed: int,
) -> list[dict[str, float | int]]:
    """Generate elimination tradeoff rows across theta and sketch dimension."""
    rows: list[dict[str, float | int]] = []
    for sketch_dim in (8, 16, 32, 64):
        for theta in np.linspace(0.1, 0.9, num=9):
            simulator = SyntheticGhostKVSimulator(
                num_tokens=num_tokens,
                dim=dim,
                sketch_dim=sketch_dim,
                hot_window=hot_window,
                theta_elim=float(theta),
                epsilon=epsilon,
                sigma=sigma,
                seed=seed + sketch_dim,
            )
            metrics = simulator.run_many_steps(steps)
            rows.append(
                {
                    "theta": round(float(theta), 3),
                    "sketch_dim": sketch_dim,
                    "elimination_rate": round(metrics["elimination_rate_mean"], 6),
                    "resurrection_rate": round(metrics["resurrection_rate_mean"], 6),
                    "false_elimination_rate": round(metrics["false_elimination_rate_mean"], 6),
                    "topk_overlap": round(metrics["topk_overlap_mean"], 6),
                    "bandwidth_reduction": round(metrics["bandwidth_reduction_mean"], 6),
                }
            )
    return rows


def run_bandwidth_sweep(
    context: int,
    dim: int,
    hot_window: int,
    ghost_record_bytes: float,
    quant_factor: float,
    kv_bytes_per_token: float | None,
) -> list[dict[str, float | int | str]]:
    """Generate illustrative bandwidth rows for comparison schemes."""
    per_token = kv_bytes_per_token or standard_kv_bytes(1, dim)
    full_kv = standard_kv_bytes(context, dim)
    int4_like = quantized_kv_bytes(context, dim, quant_factor=quant_factor)

    rows: list[dict[str, float | int | str]] = [
        {
            "scheme": "full_kv",
            "resurrection_rate": 0.0,
            "bytes": round(full_kv, 3),
            "reduction_vs_full": 0.0,
        },
        {
            "scheme": "int4_style",
            "resurrection_rate": 0.0,
            "bytes": round(int4_like, 3),
            "reduction_vs_full": round(bandwidth_reduction_ratio(full_kv, int4_like), 6),
        },
    ]

    num_ghost = context - hot_window
    for resurrection_rate in (0.005, 0.01, 0.02, 0.05):
        num_resurrected = int(num_ghost * resurrection_rate)
        ghost_bytes = ghostkv_bytes(
            num_hot=hot_window,
            num_resurrected=num_resurrected,
            kv_bytes_per_token=per_token,
            num_ghost=num_ghost,
            ghost_record_bytes=ghost_record_bytes,
        )
        rows.append(
            {
                "scheme": "ghostkv",
                "resurrection_rate": resurrection_rate,
                "bytes": round(ghost_bytes, 3),
                "reduction_vs_full": round(bandwidth_reduction_ratio(full_kv, ghost_bytes), 6),
            }
        )
    return rows


def generate_results_markdown(
    sketch_rows: list[dict[str, str]],
    elimination_rows: list[dict[str, str]],
    bandwidth_rows: list[dict[str, str]],
) -> str:
    """Build the RESULTS.md content from generated CSV data."""
    representative_elimination = [
        row
        for row in elimination_rows
        if row["theta"] in {"0.1", "0.5", "0.9"} and row["sketch_dim"] in {"16", "32", "64"}
    ]
    representative_elimination.sort(key=lambda row: (float(row["theta"]), int(row["sketch_dim"])))

    sketch_table = _markdown_table(
        sketch_rows,
        ["sketch_dim", "top8_overlap", "top16_overlap", "top32_overlap", "top64_overlap"],
    )
    elimination_table = _markdown_table(
        representative_elimination,
        [
            "theta",
            "sketch_dim",
            "elimination_rate",
            "resurrection_rate",
            "false_elimination_rate",
            "topk_overlap",
        ],
    )
    bandwidth_table = _markdown_table(
        bandwidth_rows,
        ["scheme", "resurrection_rate", "bytes", "reduction_vs_full"],
    )

    return f"""# GhostKV Synthetic Results

## Status

Synthetic simulator working. Real-model validation pending.

## Important disclaimer

These are synthetic simulation results, not real-model results.

These results use synthetic key/query tensors. They test whether the GhostKV evaluation pipeline works and whether sketch-based elimination behaves plausibly. They do not prove production speedups or model-quality preservation.

## Experiment 1: Sketch quality audit

This sweep varies `sketch_dim` across `8, 16, 32, 64` and measures how well sketch-space ranking preserves top-k exact score membership. Higher sketch dimensions should usually improve overlap, though synthetic noise and random projections can still introduce variance.

{sketch_table}

Plot: [results/sketch_dim_vs_topk_overlap.png](results/sketch_dim_vs_topk_overlap.png)

## Experiment 2: Elimination tradeoff

This sweep varies `theta` from `0.1` to `0.9` and reports elimination rate, resurrection rate, false elimination rate, and top-k overlap. The table below shows representative rows; the full grid is stored in `results/elimination_tradeoff.csv`.

{elimination_table}

Plot: [results/theta_vs_elimination_rate.png](results/theta_vs_elimination_rate.png)

## Experiment 3: Bandwidth model

This illustrative model compares full KV traffic, an INT4-style compressed baseline, and GhostKV under several resurrection rates. The purpose is to show why query-time movement reduction can matter even when exact attention still runs on survivors.

{bandwidth_table}

Plot: [results/resurrection_rate_vs_bandwidth.png](results/resurrection_rate_vs_bandwidth.png)

## Interpretation

- Synthetic results validate the harness, not the algorithm.
- Higher sketch dimensions are a positive sign when they preserve top-k overlap more reliably.
- Elimination is encouraging only when false elimination remains controlled while useful pruning still occurs.
- The bandwidth model is illustrative, but it helps motivate why reducing movement may matter more than only compressing bytes at rest.

## Limitations

- Random tensors are not transformer tensors.
- Real attention distributions may be more structured or more adversarial.
- Softmax denominator handling is not fully modeled.
- Resurrection latency is estimated, not benchmarked.
- No HuggingFace or real LLM validation exists in this repository yet.

## Next milestone

Real K/Q tensor capture from a small transformer:

- GPT-2 small or TinyLlama
- capture attention Q/K tensors
- compare exact QK ranking vs sketch ranking
- evaluate false elimination rate on real attention
"""


def generate_all_results(
    results_dir: Path,
    results_md_path: Path,
    sketch_num_tokens: int,
    tradeoff_num_tokens: int,
    bandwidth_context: int,
    dim: int,
    hot_window: int,
    bandwidth_hot_window: int,
    tradeoff_steps: int,
    epsilon: float,
    sigma: float,
    seed: int,
    ghost_record_bytes: float,
    quant_factor: float,
) -> dict[str, Path]:
    """Generate all result artifacts and return their paths."""
    sketch_rows = run_sketch_quality_sweep(
        num_tokens=sketch_num_tokens,
        dim=dim,
        seed=seed,
    )
    elimination_rows = run_elimination_tradeoff_sweep(
        num_tokens=tradeoff_num_tokens,
        dim=dim,
        hot_window=hot_window,
        steps=tradeoff_steps,
        epsilon=epsilon,
        sigma=sigma,
        seed=seed + 100,
    )
    bandwidth_rows = run_bandwidth_sweep(
        context=bandwidth_context,
        dim=dim,
        hot_window=bandwidth_hot_window,
        ghost_record_bytes=ghost_record_bytes,
        quant_factor=quant_factor,
        kv_bytes_per_token=None,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    sketch_csv = results_dir / "sketch_quality.csv"
    elimination_csv = results_dir / "elimination_tradeoff.csv"
    bandwidth_csv = results_dir / "bandwidth_sweep.csv"
    sketch_plot = results_dir / "sketch_dim_vs_topk_overlap.png"
    theta_plot = results_dir / "theta_vs_elimination_rate.png"
    bandwidth_plot = results_dir / "resurrection_rate_vs_bandwidth.png"

    _write_csv(sketch_csv, sketch_rows)
    _write_csv(elimination_csv, elimination_rows)
    _write_csv(bandwidth_csv, bandwidth_rows)

    plot_sketch_dim_vs_topk_overlap(sketch_rows, sketch_plot)
    plot_theta_vs_elimination_rate(elimination_rows, theta_plot)
    plot_resurrection_rate_vs_bandwidth(bandwidth_rows, bandwidth_plot)

    results_markdown = generate_results_markdown(
        sketch_rows=_read_csv(sketch_csv),
        elimination_rows=_read_csv(elimination_csv),
        bandwidth_rows=_read_csv(bandwidth_csv),
    )
    results_md_path.write_text(results_markdown, encoding="utf-8")

    return {
        "sketch_csv": sketch_csv,
        "elimination_csv": elimination_csv,
        "bandwidth_csv": bandwidth_csv,
        "sketch_plot": sketch_plot,
        "theta_plot": theta_plot,
        "bandwidth_plot": bandwidth_plot,
        "results_md": results_md_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sketch-num-tokens", type=int, default=8192)
    parser.add_argument("--tradeoff-num-tokens", type=int, default=16384)
    parser.add_argument("--bandwidth-context", type=int, default=128 * 1024)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--hot-window", type=int, default=2048)
    parser.add_argument("--bandwidth-hot-window", type=int, default=4096)
    parser.add_argument("--tradeoff-steps", type=int, default=25)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--ghost-record-bytes", type=float, default=64.0)
    parser.add_argument("--quant-factor", type=float, default=4.0)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--results-md", type=Path, default=RESULTS_MD)
    args = parser.parse_args()

    artifacts = generate_all_results(
        results_dir=args.results_dir,
        results_md_path=args.results_md,
        sketch_num_tokens=args.sketch_num_tokens,
        tradeoff_num_tokens=args.tradeoff_num_tokens,
        bandwidth_context=args.bandwidth_context,
        dim=args.dim,
        hot_window=args.hot_window,
        bandwidth_hot_window=args.bandwidth_hot_window,
        tradeoff_steps=args.tradeoff_steps,
        epsilon=args.epsilon,
        sigma=args.sigma,
        seed=args.seed,
        ghost_record_bytes=args.ghost_record_bytes,
        quant_factor=args.quant_factor,
    )
    for artifact in artifacts.values():
        print(f"Wrote {artifact}")


if __name__ == "__main__":
    main()
