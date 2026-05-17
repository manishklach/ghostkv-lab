"""Characterize the false-elimination frontier on real transformer attention tensors."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from ghostkv.frontier import (
    compute_false_elimination_frontier,
    ensure_output_dir,
    find_safe_operating_points,
    summarize_frontier,
)
from ghostkv.hf_capture import capture_qk_tensors, flatten_attention_heads, load_model_and_tokenizer
from ghostkv.plotting import (
    plot_false_elim_vs_elim_by_layer,
    plot_headwise_false_elim_heatmap,
    plot_sketch_dim_frontier,
    plot_theta_frontier_by_layer,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "frontier"
PROMPTS_PATH = REPO_ROOT / "data" / "prompts.txt"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV rows with stable field ordering."""
    if not rows:
        raise ValueError(f"No rows provided for {path.name}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_prompts(path: Path, max_prompts: int | None = None) -> list[tuple[str, str]]:
    """Load prompt category and text pairs from the prompt file."""
    prompts: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        prompt_type, prompt_text = stripped.split("\t", maxsplit=1)
        prompts.append((prompt_type, prompt_text))
    if max_prompts is not None:
        return prompts[:max_prompts]
    return prompts


def _write_frontier_report(
    output_path: Path,
    model_name: str,
    relevance_mode: str,
    topk: int,
    percentile: float,
    num_prompt_slices: int,
    skipped_short_topk_slices: int,
    safe_rows: list[dict[str, Any]],
    layer_summary: list[dict[str, Any]],
    head_summary: list[dict[str, Any]],
) -> None:
    """Write the frontier markdown summary."""
    safe_table_lines = [
        "| layer | sketch_dim | theta_elim | elimination_rate_mean | false_elimination_rate_mean | top32_overlap_mean |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in safe_rows[:12]:
        safe_table_lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("layer_idx", "")),
                    str(row.get("sketch_dim", "")),
                    str(row.get("theta_elim", "")),
                    str(row.get("elimination_rate_mean", "")),
                    str(row.get("false_elimination_rate_mean", "")),
                    str(row.get("top32_overlap_mean", "")),
                ]
            )
            + " |"
        )

    best_layer_rows = sorted(
        layer_summary,
        key=lambda row: (float(row["false_elimination_rate_mean"]), -float(row["elimination_rate_mean"])),
    )[:6]
    layer_lines = [
        f"- Layer {row['layer_idx']}, sketch {row['sketch_dim']}, theta {row['theta_elim']}: "
        f"false elim {row['false_elimination_rate_mean']}, elimination {row['elimination_rate_mean']}"
        for row in best_layer_rows
    ]

    head_lines = []
    best_head_rows = sorted(
        head_summary,
        key=lambda row: (float(row["false_elimination_rate_mean"]), -float(row["elimination_rate_mean"])),
    )[:6]
    for row in best_head_rows:
        head_lines.append(
            f"- Layer {row['layer_idx']} head {row['head_idx']} at sketch {row['sketch_dim']} and theta {row['theta_elim']}: "
            f"false elim {row['false_elimination_rate_mean']}, elimination {row['elimination_rate_mean']}"
        )

    if not safe_rows:
        safe_table_lines.append("| _none found_ |  |  |  |  |  |")

    summary = f"""# False Elimination Frontier

## Status

Real-attention frontier analysis generated for `{model_name}`.

## Methodology

This experiment captures real query/key tensors from selected transformer layers, applies GhostKV-style sketch projections, sweeps `theta_elim`, and records the tradeoff between elimination and false elimination at layer/head granularity.

## Relevance definition

- Relevance mode: `{relevance_mode}`
- Top-k: `{topk}`
- Percentile: `{percentile}`
- Informative prompt-layer slices analyzed: `{num_prompt_slices}`
- Prompt-layer slices skipped because `seq_len <= topk`: `{skipped_short_topk_slices}`

By default, relevance is defined by exact top-k membership, not by the approximate sketch score.

## Key findings

- This analysis does not prove GhostKV correctness.
- The goal is to locate operating regions where elimination is meaningful while false elimination remains controlled.
- False elimination remains the primary technical risk.
- High rank correlation alone is not enough to establish acceptable elimination behavior.

## Safe-ish operating points

{chr(10).join(safe_table_lines)}

## Layer-wise observations

{chr(10).join(layer_lines) if layer_lines else "- No especially conservative layer-level operating points were identified under the current thresholds."}

## Head-wise observations

{chr(10).join(head_lines) if head_lines else "- Head-wise variability remains substantial, and no clearly conservative head-specific region was isolated."}

## Limitations

- This frontier is measured on GPT-2 attention tensors, not on larger long-context models.
- The analysis does not measure runtime or memory movement directly.
- Resurrection is still simulated.
- Extreme-rank preservation is sensitive to prompt, layer, and head behavior.

## Next steps

- Expand the frontier analysis to additional decoder architectures.
- Compare top-k relevance with percentile-based relevance.
- Study whether hierarchical or learned sketches can lower false elimination without collapsing elimination rate.
- Use the frontier to decide whether deeper systems integration is warranted.
"""
    output_path.write_text(summary, encoding="utf-8")


def run_false_elimination_frontier(
    model_name: str,
    layers: list[int],
    sketch_dims: list[int],
    theta_values: list[float],
    topk: int,
    relevance_mode: str,
    percentile: float,
    max_prompts: int | None,
    output_dir: Path,
    seed: int,
    epsilon: float = 0.05,
    sigma: float = 0.05,
    prompts_path: Path = PROMPTS_PATH,
) -> dict[str, Path]:
    """Run the frontier experiment and write CSV, plots, and markdown outputs."""
    output_dir = ensure_output_dir(output_dir)
    model, tokenizer = load_model_and_tokenizer(model_name)
    prompts = _load_prompts(prompts_path, max_prompts=max_prompts)

    frontier_rows: list[dict[str, Any]] = []
    num_prompt_slices = 0
    skipped_short_topk_slices = 0
    for prompt_idx, (prompt_type, prompt_text) in enumerate(prompts):
        for layer_idx in layers:
            capture = capture_qk_tensors(model, tokenizer, prompt_text, layer_idx=layer_idx)
            if relevance_mode == "topk" and int(capture.input_ids.shape[-1]) <= topk:
                skipped_short_topk_slices += 1
                continue

            num_prompt_slices += 1
            query_heads, key_heads = flatten_attention_heads(capture.query_states, capture.key_states)
            for sketch_dim in sketch_dims:
                head_rows = compute_false_elimination_frontier(
                    query_heads=query_heads,
                    key_heads=key_heads,
                    sketch_dim=sketch_dim,
                    theta_values=theta_values,
                    epsilon=epsilon,
                    sigma=sigma,
                    relevance_mode=relevance_mode,
                    topk=topk,
                    percentile=percentile,
                    seed=seed + prompt_idx + layer_idx + sketch_dim,
                )
                for row in head_rows:
                    frontier_rows.append(
                        {
                            "model_name": str(getattr(model.config, "ghostkv_resolved_model_name", model_name)),
                            "prompt_idx": prompt_idx,
                            "prompt_type": prompt_type,
                            "layer_idx": layer_idx,
                            **row,
                        }
                    )

    layer_summary = summarize_frontier(frontier_rows, ["layer_idx", "sketch_dim", "theta_elim"])
    head_summary = summarize_frontier(frontier_rows, ["layer_idx", "head_idx", "sketch_dim", "theta_elim"])
    safe_rows = find_safe_operating_points(layer_summary, target_false_elim=0.05, min_elimination_rate=0.30)

    frontier_csv = output_dir / "false_elimination_frontier.csv"
    safe_csv = output_dir / "safe_operating_points.csv"
    layer_csv = output_dir / "layer_summary.csv"
    head_csv = output_dir / "head_summary.csv"
    frontier_md = output_dir / "FRONTIER.md"
    false_vs_elim_png = output_dir / "false_elim_vs_elim_by_layer.png"
    theta_png = output_dir / "theta_frontier_by_layer.png"
    sketch_png = output_dir / "sketch_dim_frontier.png"
    heatmap_png = output_dir / "headwise_false_elim_heatmap.png"

    _write_csv(frontier_csv, frontier_rows)
    _write_csv(layer_csv, layer_summary)
    _write_csv(head_csv, head_summary)
    if safe_rows:
        _write_csv(safe_csv, safe_rows)
    else:
        _write_csv(
            safe_csv,
            [
                {
                    "note": "No safe-ish operating points found under the current thresholds.",
                    "target_false_elim": 0.05,
                    "min_elimination_rate": 0.30,
                }
            ],
        )

    plot_false_elim_vs_elim_by_layer(layer_summary, false_vs_elim_png)
    theta_focus_rows = [
        row for row in layer_summary if int(row["sketch_dim"]) == sketch_dims[min(2, len(sketch_dims) - 1)]
    ]
    plot_theta_frontier_by_layer(theta_focus_rows, theta_png)
    plot_sketch_dim_frontier(layer_summary, sketch_png)
    heatmap_rows = [
        row for row in head_summary if int(row["sketch_dim"]) == sketch_dims[min(2, len(sketch_dims) - 1)]
    ]
    plot_headwise_false_elim_heatmap(heatmap_rows, heatmap_png)

    _write_frontier_report(
        frontier_md,
        model_name=str(getattr(model.config, "ghostkv_resolved_model_name", model_name)),
        relevance_mode=relevance_mode,
        topk=topk,
        percentile=percentile,
        num_prompt_slices=num_prompt_slices,
        skipped_short_topk_slices=skipped_short_topk_slices,
        safe_rows=safe_rows,
        layer_summary=layer_summary,
        head_summary=head_summary,
    )

    return {
        "frontier_csv": frontier_csv,
        "safe_csv": safe_csv,
        "layer_csv": layer_csv,
        "head_csv": head_csv,
        "false_vs_elim_png": false_vs_elim_png,
        "theta_png": theta_png,
        "sketch_png": sketch_png,
        "heatmap_png": heatmap_png,
        "frontier_md": frontier_md,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 3, 6, 9])
    parser.add_argument("--sketch-dims", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--theta-min", type=float, default=0.05)
    parser.add_argument("--theta-max", type=float, default=0.95)
    parser.add_argument("--theta-step", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--relevance-mode", type=str, choices=["topk", "percentile"], default="topk")
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--max-prompts", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--prompts-path", type=Path, default=PROMPTS_PATH)
    args = parser.parse_args()

    theta_values = [
        round(value, 4)
        for value in np.arange(args.theta_min, args.theta_max + (0.5 * args.theta_step), args.theta_step)
    ]
    artifacts = run_false_elimination_frontier(
        model_name=args.model,
        layers=args.layers,
        sketch_dims=args.sketch_dims,
        theta_values=theta_values,
        topk=args.topk,
        relevance_mode=args.relevance_mode,
        percentile=args.percentile,
        max_prompts=args.max_prompts,
        output_dir=args.output_dir,
        seed=args.seed,
        epsilon=args.epsilon,
        sigma=args.sigma,
        prompts_path=args.prompts_path,
    )
    for artifact in artifacts.values():
        print(f"Wrote {artifact}")


if __name__ == "__main__":
    main()
