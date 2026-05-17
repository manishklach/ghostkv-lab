"""Audit sketch quality across multiple sketch dimensions."""

from __future__ import annotations

import argparse

import numpy as np

from ghostkv.bounds import topk_overlap
from ghostkv.metrics import pretty_print_table
from ghostkv.sketches import (
    cosine_rank_correlation,
    project_keys,
    project_query,
    random_projection_matrix,
    sketch_similarity,
)


def rank_correlation(exact_scores: np.ndarray, sketch_scores: np.ndarray) -> float:
    """Use SciPy Spearman correlation when available, else a lightweight fallback."""
    try:
        from scipy.stats import spearmanr  # type: ignore

        result = spearmanr(exact_scores, sketch_scores)
        return float(result.statistic)
    except Exception:
        return cosine_rank_correlation(exact_scores, sketch_scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    keys = rng.normal(size=(args.num_tokens, args.dim))
    query = rng.normal(size=(args.dim,))
    exact_scores = (keys @ query) / np.sqrt(args.dim)

    rows: list[dict[str, object]] = []
    for sketch_dim in (8, 16, 32, 64):
        projection = random_projection_matrix(args.dim, sketch_dim, seed=args.seed + sketch_dim)
        key_sketches = project_keys(keys, projection)
        query_sketch = project_query(query, projection)
        approx_scores = sketch_similarity(query_sketch, key_sketches)
        rows.append(
            {
                "sketch_dim": sketch_dim,
                "top8": f"{topk_overlap(exact_scores, approx_scores, 8):.3f}",
                "top16": f"{topk_overlap(exact_scores, approx_scores, 16):.3f}",
                "top32": f"{topk_overlap(exact_scores, approx_scores, 32):.3f}",
                "top64": f"{topk_overlap(exact_scores, approx_scores, 64):.3f}",
                "rank_corr": f"{rank_correlation(exact_scores, approx_scores):.3f}",
            }
        )

    print("GhostKV sketch quality audit")
    print(pretty_print_table(rows, ["sketch_dim", "top8", "top16", "top32", "top64", "rank_corr"]))


if __name__ == "__main__":
    main()

