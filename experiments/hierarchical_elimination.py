"""Evaluate hierarchical GhostKV-style elimination on real attention tensors."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ghostkv.bounds import false_elimination_rate
from ghostkv.hf_capture import capture_qk_tensors, compute_exact_attention_scores, flatten_attention_heads, load_model_and_tokenizer
from ghostkv.hierarchical import hierarchical_token_elimination, token_level_elimination
from ghostkv.plotting import plot_hierarchical_vs_flat
from ghostkv.sketches import project_keys, project_query, random_projection_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
PROMPTS_PATH = REPO_ROOT / "data" / "prompts.txt"


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _load_prompts(path: Path, max_prompts: int | None = None) -> list[str]:
    prompts = [
        line.split("\t", maxsplit=1)[1]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if max_prompts is not None:
        return prompts[:max_prompts]
    return prompts


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows provided for {path.name}.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_hierarchical_elimination(
    model_name: str,
    prompts_path: Path,
    layers: list[int],
    sketch_dim: int,
    thetas: list[float],
    epsilon: float,
    sigma: float,
    topk: int,
    num_anchors: int,
    seed: int,
    max_prompts: int | None,
    results_dir: Path,
) -> dict[str, Path]:
    """Run a hierarchical-versus-flat elimination comparison on real attention tensors."""
    model, tokenizer = load_model_and_tokenizer(model_name)
    prompts = _load_prompts(prompts_path, max_prompts=max_prompts)
    rows: list[dict[str, object]] = []

    for prompt_idx, prompt in enumerate(prompts):
        for layer_idx in layers:
            capture = capture_qk_tensors(model, tokenizer, prompt, layer_idx)
            query_heads, key_heads = flatten_attention_heads(capture.query_states, capture.key_states)
            projection = random_projection_matrix(query_heads.shape[-1], sketch_dim, seed=seed + prompt_idx + layer_idx)

            for head_idx in range(query_heads.shape[0]):
                exact_scores = compute_exact_attention_scores(query_heads[head_idx], key_heads[head_idx])
                query_sketch = project_query(query_heads[head_idx], projection)
                key_sketches = project_keys(key_heads[head_idx], projection)
                effective_k = min(topk, exact_scores.shape[0])
                true_threshold = np.partition(exact_scores, -effective_k)[-effective_k]

                for theta in thetas:
                    flat_eliminated = token_level_elimination(
                        query_sketch=query_sketch,
                        key_sketches=key_sketches,
                        theta=theta,
                        epsilon=epsilon,
                        sigma=sigma,
                    )
                    hierarchical = hierarchical_token_elimination(
                        query_sketch=query_sketch,
                        key_sketches=key_sketches,
                        num_anchors=num_anchors,
                        theta=theta,
                        epsilon=epsilon,
                        sigma=sigma,
                        seed=seed + head_idx,
                    )

                    rows.extend(
                        [
                            {
                                "method": "flat",
                                "layer_idx": layer_idx,
                                "head_idx": head_idx,
                                "theta": theta,
                                "false_elimination_rate": round(
                                    float(false_elimination_rate(exact_scores, flat_eliminated, true_threshold)),
                                    6,
                                ),
                                "elimination_rate": round(float(np.mean(flat_eliminated)), 6),
                            },
                            {
                                "method": "hierarchical",
                                "layer_idx": layer_idx,
                                "head_idx": head_idx,
                                "theta": theta,
                                "false_elimination_rate": round(
                                    float(
                                        false_elimination_rate(
                                            exact_scores,
                                            hierarchical["eliminated_mask"],
                                            true_threshold,
                                        )
                                    ),
                                    6,
                                ),
                                "elimination_rate": round(
                                    float(np.mean(hierarchical["eliminated_mask"])),
                                    6,
                                ),
                            },
                        ]
                    )

    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["method"]), float(row["theta"]))
        grouped.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, object]] = []
    for (method, theta), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        aggregate_rows.append(
            {
                "method": method,
                "theta": theta,
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

    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "hierarchical_results.csv"
    plot_path = results_dir / "hierarchical_vs_flat.png"
    _write_csv(csv_path, aggregate_rows)
    plot_hierarchical_vs_flat(aggregate_rows, plot_path)
    return {"csv": csv_path, "plot": plot_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--prompts-path", type=Path, default=PROMPTS_PATH)
    parser.add_argument("--layers", type=str, default="0,3,6,9")
    parser.add_argument("--sketch-dim", type=int, default=32)
    parser.add_argument("--thetas", type=str, default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-anchors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-prompts", type=int, default=6)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    artifacts = run_hierarchical_elimination(
        model_name=args.model_name,
        prompts_path=args.prompts_path,
        layers=_parse_int_list(args.layers),
        sketch_dim=args.sketch_dim,
        thetas=_parse_float_list(args.thetas),
        epsilon=args.epsilon,
        sigma=args.sigma,
        topk=args.topk,
        num_anchors=args.num_anchors,
        seed=args.seed,
        max_prompts=args.max_prompts,
        results_dir=args.results_dir,
    )
    for artifact in artifacts.values():
        print(f"Wrote {artifact}")


if __name__ == "__main__":
    main()
