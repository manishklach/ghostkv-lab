"""Upper-bound and elimination helpers for GhostKV."""

from __future__ import annotations

import numpy as np


def compute_attention_upper_bound(
    sketch_scores: np.ndarray, epsilon_res: np.ndarray | float, sigma_anchor: np.ndarray | float
) -> np.ndarray:
    """Compute a conservative additive upper bound on attention score."""
    return np.asarray(sketch_scores) + np.asarray(epsilon_res) + np.asarray(sigma_anchor)


def eliminate_by_threshold(bounds: np.ndarray, theta: float) -> np.ndarray:
    """Return a mask for candidates that can be eliminated."""
    return np.asarray(bounds) < theta


def topk_overlap(exact_scores: np.ndarray, approx_scores: np.ndarray, k: int) -> float:
    """Measure overlap between top-k sets of two score vectors."""
    if k <= 0:
        raise ValueError("k must be positive.")
    k = min(k, exact_scores.size, approx_scores.size)
    exact_topk = set(np.argsort(-exact_scores)[:k].tolist())
    approx_topk = set(np.argsort(-approx_scores)[:k].tolist())
    return len(exact_topk & approx_topk) / float(k)


def false_elimination_rate(
    exact_scores: np.ndarray, eliminated_mask: np.ndarray, true_threshold: float
) -> float:
    """Measure the fraction of truly relevant tokens that were eliminated."""
    exact_scores = np.asarray(exact_scores)
    eliminated_mask = np.asarray(eliminated_mask, dtype=bool)
    if exact_scores.shape != eliminated_mask.shape:
        raise ValueError("Inputs must have matching shapes.")

    truly_relevant = exact_scores >= true_threshold
    relevant_count = int(np.count_nonzero(truly_relevant))
    if relevant_count == 0:
        return 0.0
    false_eliminations = np.count_nonzero(truly_relevant & eliminated_mask)
    return float(false_eliminations / relevant_count)

