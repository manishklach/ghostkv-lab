"""Validate GhostKV sketches on real transformer attention tensors."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ghostkv.hf_capture import capture_qk_tensors, extract_attention_statistics, load_model_and_tokenizer
from ghostkv.plotting import (
    plot_head_variance,
    plot_layerwise_overlap,
    plot_real_attention_false_elimination,
    plot_real_attention_topk_overlap,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
PROMPTS_PATH = REPO_ROOT / "data" / "prompts.txt"


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _load_prompts(path: Path, max_prompts: int | None = None) -> list[tuple[str, str]]:
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


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows provided for {path.name}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int, float], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["layer_idx"]), int(row["sketch_dim"]), float(row["theta"]))
        grouped.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, object]] = []
    for (layer_idx, sketch_dim, theta), group_rows in sorted(grouped.items()):
        aggregate_rows.append(
            {
                "layer_idx": layer_idx,
                "sketch_dim": sketch_dim,
                "theta": round(theta, 3),
                "topk_overlap_mean": round(float(np.mean([float(row["topk_overlap"]) for row in group_rows])), 6),
                "topk_overlap_std": round(float(np.std([float(row["topk_overlap"]) for row in group_rows])), 6),
                "rank_correlation_mean": round(
                    float(np.mean([float(row["rank_correlation"]) for row in group_rows])),
                    6,
                ),
                "false_elimination_rate_mean": round(
                    float(np.mean([float(row["false_elimination_rate"]) for row in group_rows])),
                    6,
                ),
                "elimination_rate_mean": round(
                    float(np.mean([float(row["elimination_rate"]) for row in group_rows])),
                    6,
                ),
                "num_samples": len(group_rows),
            }
        )
    return aggregate_rows


def _write_summary(
    output_path: Path,
    model_name: str,
    aggregate_rows: list[dict[str, object]],
    num_prompts: int,
    layers: list[int],
) -> None:
    focus_rows = [
        row for row in aggregate_rows if int(row["sketch_dim"]) == 32 and float(row["theta"]) == 0.5
    ]
    focus_rows = sorted(focus_rows, key=lambda row: int(row["layer_idx"]))
    table_lines = [
        "| layer | topk_overlap_mean | false_elimination_rate_mean | elimination_rate_mean |",
        "| --- | --- | --- | --- |",
    ]
    for row in focus_rows:
        table_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["layer_idx"]),
                    str(row["topk_overlap_mean"]),
                    str(row["false_elimination_rate_mean"]),
                    str(row["elimination_rate_mean"]),
                ]
            )
            + " |"
        )

    summary = f"""# Real Attention Validation Summary

These are real-model attention validation results, not throughput benchmarks.

- Model used: `{model_name}`
- Prompts evaluated: `{num_prompts}`
- Layers evaluated: `{layers}`

## Focus slice

The table below highlights `sketch_dim=32` and `theta=0.5`.

{chr(10).join(table_lines)}

## Notes

- Real transformer tensors need not behave like Gaussian synthetic tensors.
- Global similarity structure is often easier to preserve than exact top-attention ranking.
- False elimination remains the main failure mode to monitor.
- Layer and head behavior vary, so aggregate metrics can hide fragile substructures.
"""
    output_path.write_text(summary, encoding="utf-8")


def run_real_attention_validation(
    model_name: str,
    prompts_path: Path,
    layers: list[int],
    sketch_dims: list[int],
    thetas: list[float],
    epsilon: float,
    sigma: float,
    topk: int,
    seed: int,
    max_prompts: int | None,
    results_dir: Path,
) -> dict[str, Path]:
    """Run the real-attention validation experiment and write result artifacts."""
    model, tokenizer = load_model_and_tokenizer(model_name)
    prompts = _load_prompts(prompts_path, max_prompts=max_prompts)

    per_head_rows: list[dict[str, object]] = []
    for prompt_idx, (prompt_type, prompt_text) in enumerate(prompts):
        for layer_idx in layers:
            capture = capture_qk_tensors(model, tokenizer, prompt_text, layer_idx=layer_idx)
            for sketch_dim in sketch_dims:
                for theta in thetas:
                    stats = extract_attention_statistics(
                        capture.query_states,
                        capture.key_states,
                        sketch_dim=sketch_dim,
                        theta=theta,
                        epsilon=epsilon,
                        sigma=sigma,
                        topk=topk,
                        seed=seed + prompt_idx + layer_idx + sketch_dim,
                    )
                    for row in stats["per_head_rows"]:
                        per_head_rows.append(
                            {
                                "model_name": capture.model_name,
                                "prompt_idx": prompt_idx,
                                "prompt_type": prompt_type,
                                "prompt_length": capture.input_ids.shape[-1],
                                "layer_idx": layer_idx,
                                **row,
                            }
                        )

    aggregate_rows = _aggregate_rows(per_head_rows)

    results_dir.mkdir(parents=True, exist_ok=True)
    headwise_csv = results_dir / "headwise_metrics.csv"
    aggregate_csv = results_dir / "real_attention_validation.csv"
    topk_plot = results_dir / "real_attention_topk_overlap.png"
    false_plot = results_dir / "real_attention_false_elimination.png"
    layer_plot = results_dir / "real_attention_layerwise_overlap.png"
    head_variance_plot = results_dir / "head_variance.png"
    summary_md = results_dir / "real_attention_summary.md"

    _write_csv(headwise_csv, per_head_rows)
    _write_csv(aggregate_csv, aggregate_rows)

    topk_rows = [row for row in aggregate_rows if float(row["theta"]) == thetas[0]]
    false_rows = [row for row in aggregate_rows if int(row["sketch_dim"]) == sketch_dims[min(2, len(sketch_dims) - 1)]]
    layer_rows = [
        row
        for row in aggregate_rows
        if int(row["sketch_dim"]) == sketch_dims[min(2, len(sketch_dims) - 1)] and float(row["theta"]) == thetas[0]
    ]
    head_variance_rows = [
        row
        for row in per_head_rows
        if int(row["sketch_dim"]) == sketch_dims[min(2, len(sketch_dims) - 1)] and float(row["theta"]) == thetas[0]
    ]

    plot_real_attention_topk_overlap(topk_rows, topk_plot)
    plot_real_attention_false_elimination(false_rows, false_plot)
    plot_layerwise_overlap(layer_rows, layer_plot)
    plot_head_variance(head_variance_rows, head_variance_plot)
    _write_summary(summary_md, str(getattr(model.config, "ghostkv_resolved_model_name", model_name)), aggregate_rows, len(prompts), layers)

    return {
        "aggregate_csv": aggregate_csv,
        "headwise_csv": headwise_csv,
        "topk_plot": topk_plot,
        "false_plot": false_plot,
        "layer_plot": layer_plot,
        "head_variance_plot": head_variance_plot,
        "summary_md": summary_md,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--prompts-path", type=Path, default=PROMPTS_PATH)
    parser.add_argument("--layers", type=str, default="0,3,6,9")
    parser.add_argument("--sketch-dims", type=str, default="8,16,32,64")
    parser.add_argument("--thetas", type=str, default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-prompts", type=int, default=10)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    artifacts = run_real_attention_validation(
        model_name=args.model_name,
        prompts_path=args.prompts_path,
        layers=_parse_int_list(args.layers),
        sketch_dims=_parse_int_list(args.sketch_dims),
        thetas=_parse_float_list(args.thetas),
        epsilon=args.epsilon,
        sigma=args.sigma,
        topk=args.topk,
        seed=args.seed,
        max_prompts=args.max_prompts,
        results_dir=args.results_dir,
    )
    for artifact in artifacts.values():
        print(f"Wrote {artifact}")


if __name__ == "__main__":
    main()
