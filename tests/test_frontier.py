from pathlib import Path

import numpy as np

from ghostkv.frontier import (
    compute_layer_head_metrics,
    ensure_output_dir,
    find_safe_operating_points,
    summarize_frontier,
)


def test_compute_layer_head_metrics_false_elimination() -> None:
    exact_scores = np.array([0.9, 0.8, 0.2, 0.1], dtype=float)
    sketch_scores = np.array([0.85, 0.6, 0.25, 0.05], dtype=float)
    bounds = np.array([0.86, 0.61, 0.24, 0.04], dtype=float)

    metrics = compute_layer_head_metrics(
        exact_scores=exact_scores,
        sketch_scores=sketch_scores,
        bounds=bounds,
        theta_elim=0.7,
        relevance_mode="topk",
        topk=2,
    )

    assert metrics["eliminated_count"] == 3
    assert metrics["false_eliminated_count"] == 1
    assert metrics["false_elimination_rate"] == 0.5


def test_find_safe_operating_points_filters_rows() -> None:
    rows = [
        {"layer_idx": 0, "sketch_dim": 32, "theta_elim": 0.2, "false_elimination_rate_mean": 0.04, "elimination_rate_mean": 0.35},
        {"layer_idx": 0, "sketch_dim": 32, "theta_elim": 0.4, "false_elimination_rate_mean": 0.07, "elimination_rate_mean": 0.40},
    ]
    safe_rows = find_safe_operating_points(rows, target_false_elim=0.05, min_elimination_rate=0.30)

    assert len(safe_rows) == 1
    assert safe_rows[0]["theta_elim"] == 0.2


def test_summarize_frontier_and_empty_safeish() -> None:
    rows = [
        {"layer_idx": 0, "head_idx": 0, "sketch_dim": 8, "theta_elim": 0.1, "elimination_rate": 0.2, "resurrection_rate": 0.8, "false_elimination_rate": 0.2, "top8_overlap": 0.7, "top16_overlap": 0.8, "top32_overlap": 0.9, "rank_correlation": 0.95, "eliminated_count": 2, "false_eliminated_count": 1, "total_tokens": 10, "survivor_count": 8, "relevant_count": 3},
        {"layer_idx": 0, "head_idx": 1, "sketch_dim": 8, "theta_elim": 0.1, "elimination_rate": 0.25, "resurrection_rate": 0.75, "false_elimination_rate": 0.3, "top8_overlap": 0.6, "top16_overlap": 0.7, "top32_overlap": 0.8, "rank_correlation": 0.9, "eliminated_count": 3, "false_eliminated_count": 1, "total_tokens": 10, "survivor_count": 7, "relevant_count": 3},
    ]
    summary = summarize_frontier(rows, ["layer_idx", "sketch_dim", "theta_elim"])
    safe_rows = find_safe_operating_points(summary, target_false_elim=0.05, min_elimination_rate=0.30)

    assert len(summary) == 1
    assert summary[0]["num_samples"] == 2
    assert safe_rows == []


def test_ensure_output_dir_creates_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "frontier"
    ensure_output_dir(output_dir)
    assert output_dir.exists()
