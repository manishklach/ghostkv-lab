"""Sketching utilities for synthetic GhostKV experiments."""

from __future__ import annotations

from typing import Optional

import numpy as np


def random_projection_matrix(
    input_dim: int, sketch_dim: int, seed: Optional[int] = None
) -> np.ndarray:
    """Create a Gaussian random projection matrix with variance-preserving scaling."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0 / np.sqrt(sketch_dim), size=(input_dim, sketch_dim))


def project_keys(keys: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
    """Project a matrix of keys into sketch space."""
    return keys @ projection_matrix


def project_query(query: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
    """Project a query vector into sketch space."""
    return query @ projection_matrix


def sketch_similarity(query_sketch: np.ndarray, key_sketches: np.ndarray) -> np.ndarray:
    """Compute approximate similarity scores in sketch space."""
    scale = np.sqrt(query_sketch.shape[-1]) if query_sketch.shape[-1] > 0 else 1.0
    return key_sketches @ query_sketch / scale


def cosine_rank_correlation(exact_scores: np.ndarray, sketch_scores: np.ndarray) -> float:
    """Compare exact and approximate rankings via cosine similarity of rank vectors."""
    if exact_scores.shape != sketch_scores.shape:
        raise ValueError("Score arrays must have the same shape.")

    exact_rank = np.empty_like(exact_scores, dtype=float)
    sketch_rank = np.empty_like(sketch_scores, dtype=float)

    exact_order = np.argsort(-exact_scores)
    sketch_order = np.argsort(-sketch_scores)

    exact_rank[exact_order] = np.arange(1, exact_scores.size + 1, dtype=float)
    sketch_rank[sketch_order] = np.arange(1, sketch_scores.size + 1, dtype=float)

    denom = np.linalg.norm(exact_rank) * np.linalg.norm(sketch_rank)
    if denom == 0.0:
        return 0.0
    return float(np.dot(exact_rank, sketch_rank) / denom)

