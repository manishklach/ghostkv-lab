"""False-elimination frontier analysis helpers for GhostKV."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .bounds import compute_attention_upper_bound, eliminate_by_threshold, topk_overlap
from .sketches import cosine_rank_correlation, project_keys, project_query, random_projection_matrix, sketch_similarity


def relevant_token_mask(
    exact_scores: np.ndarray,
    relevance_mode: str = "topk",
    topk: int = 32,
    percentile: float = 95.0,
) -> np.ndarray:
    """Create a boolean mask for tokens considered relevant under an exact-score rule."""
    if exact_scores.ndim != 1:
        raise ValueError("exact_scores must be one-dimensional.")

    if relevance_mode == "topk":
        effective_k = max(1, min(topk, exact_scores.shape[0]))
        relevant_indices = np.argsort(-exact_scores)[:effective_k]
        mask = np.zeros(exact_scores.shape[0], dtype=bool)
        mask[relevant_indices] = True
        return mask
    if relevance_mode == "percentile":
        threshold = np.percentile(exact_scores, percentile)
        return exact_scores >= threshold
    raise ValueError(f"Unsupported relevance_mode: {relevance_mode}")


def compute_layer_head_metrics(
    exact_scores: np.ndarray,
    sketch_scores: np.ndarray,
    bounds: np.ndarray,
    theta_elim: float,
    relevance_mode: str = "topk",
    topk: int = 32,
    percentile: float = 95.0,
) -> dict[str, float | int]:
    """Compute elimination and ranking metrics for one layer/head/query slice."""
    eliminated_mask = eliminate_by_threshold(bounds, theta_elim)
    relevant_mask = relevant_token_mask(
        exact_scores,
        relevance_mode=relevance_mode,
        topk=topk,
        percentile=percentile,
    )
    false_eliminated_mask = relevant_mask & eliminated_mask

    eliminated_count = int(np.count_nonzero(eliminated_mask))
    false_eliminated_count = int(np.count_nonzero(false_eliminated_mask))
    total_tokens = int(exact_scores.shape[0])
    survivor_count = total_tokens - eliminated_count
    relevant_count = int(np.count_nonzero(relevant_mask))

    false_elimination_rate = (
        float(false_eliminated_count / relevant_count) if relevant_count > 0 else 0.0
    )

    return {
        "elimination_rate": float(eliminated_count / total_tokens) if total_tokens > 0 else 0.0,
        "resurrection_rate": float(survivor_count / total_tokens) if total_tokens > 0 else 0.0,
        "false_elimination_rate": false_elimination_rate,
        "top8_overlap": float(topk_overlap(exact_scores, sketch_scores, min(8, total_tokens))),
        "top16_overlap": float(topk_overlap(exact_scores, sketch_scores, min(16, total_tokens))),
        "top32_overlap": float(topk_overlap(exact_scores, sketch_scores, min(32, total_tokens))),
        "rank_correlation": float(cosine_rank_correlation(exact_scores, sketch_scores)),
        "eliminated_count": eliminated_count,
        "false_eliminated_count": false_eliminated_count,
        "total_tokens": total_tokens,
        "survivor_count": survivor_count,
        "relevant_count": relevant_count,
    }


def compute_false_elimination_frontier(
    query_heads: np.ndarray,
    key_heads: np.ndarray,
    sketch_dim: int,
    theta_values: list[float],
    epsilon: float,
    sigma: float,
    relevance_mode: str = "topk",
    topk: int = 32,
    percentile: float = 95.0,
    seed: int = 0,
) -> list[dict[str, float | int]]:
    """Compute frontier rows for all heads of a single prompt/layer slice."""
    if query_heads.ndim != 2 or key_heads.ndim != 3:
        raise ValueError("Expected query_heads shape [heads, dim] and key_heads shape [heads, seq, dim].")

    rows: list[dict[str, float | int]] = []
    projection_matrix = random_projection_matrix(query_heads.shape[-1], sketch_dim, seed=seed)
    for head_idx in range(query_heads.shape[0]):
        exact_scores = (key_heads[head_idx] @ query_heads[head_idx]) / np.sqrt(query_heads.shape[-1])
        key_sketches = project_keys(key_heads[head_idx], projection_matrix)
        query_sketch = project_query(query_heads[head_idx], projection_matrix)
        sketch_scores = sketch_similarity(query_sketch, key_sketches)
        bounds = compute_attention_upper_bound(sketch_scores, epsilon, sigma)

        for theta_elim in theta_values:
            metrics = compute_layer_head_metrics(
                exact_scores=exact_scores,
                sketch_scores=sketch_scores,
                bounds=bounds,
                theta_elim=theta_elim,
                relevance_mode=relevance_mode,
                topk=topk,
                percentile=percentile,
            )
            rows.append(
                {
                    "head_idx": head_idx,
                    "sketch_dim": sketch_dim,
                    "theta_elim": round(float(theta_elim), 4),
                    **metrics,
                }
            )
    return rows


def summarize_frontier(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    """Aggregate frontier rows over the requested grouping keys."""
    if not rows:
        return []

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    metric_fields = [
        "elimination_rate",
        "resurrection_rate",
        "false_elimination_rate",
        "top8_overlap",
        "top16_overlap",
        "top32_overlap",
        "rank_correlation",
    ]
    count_fields = [
        "eliminated_count",
        "false_eliminated_count",
        "total_tokens",
        "survivor_count",
        "relevant_count",
    ]

    for key, group_rows in sorted(grouped.items()):
        summary_row: dict[str, Any] = {group_key: key[idx] for idx, group_key in enumerate(group_keys)}
        for field in metric_fields:
            values = [float(row[field]) for row in group_rows]
            summary_row[f"{field}_mean"] = round(float(np.mean(values)), 6)
            summary_row[f"{field}_std"] = round(float(np.std(values)), 6)
        for field in count_fields:
            values = [int(row[field]) for row in group_rows]
            summary_row[field] = int(np.sum(values))
        summary_row["num_samples"] = len(group_rows)
        summary_rows.append(summary_row)
    return summary_rows


def find_safe_operating_points(
    summary_rows: list[dict[str, Any]],
    target_false_elim: float = 0.05,
    min_elimination_rate: float = 0.30,
) -> list[dict[str, Any]]:
    """Return conservative operating points that satisfy bounded-risk heuristics."""
    safe_rows = [
        row
        for row in summary_rows
        if float(row.get("false_elimination_rate_mean", 1.0)) <= target_false_elim
        and float(row.get("elimination_rate_mean", 0.0)) >= min_elimination_rate
    ]
    safe_rows.sort(
        key=lambda row: (
            float(row["false_elimination_rate_mean"]),
            -float(row["elimination_rate_mean"]),
            float(row.get("theta_elim", 0.0)),
        )
    )
    return safe_rows


def ensure_output_dir(output_dir: Path) -> Path:
    """Create the frontier output directory if needed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
