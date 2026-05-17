"""Hierarchical anchor-based elimination helpers for GhostKV experiments."""

from __future__ import annotations

from typing import Any

import numpy as np

from .bounds import compute_attention_upper_bound, eliminate_by_threshold
from .sketches import sketch_similarity


def simple_anchor_clustering(
    keys: np.ndarray,
    num_anchors: int,
    seed: int = 0,
    max_iters: int = 20,
) -> dict[str, Any]:
    """Cluster keys into a small number of anchor centroids using lightweight k-means."""
    if keys.ndim != 2:
        raise ValueError("keys must have shape [num_tokens, dim].")
    if num_anchors <= 0:
        raise ValueError("num_anchors must be positive.")

    rng = np.random.default_rng(seed)
    num_anchors = min(num_anchors, keys.shape[0])
    initial_indices = rng.choice(keys.shape[0], size=num_anchors, replace=False)
    centroids = keys[initial_indices].copy()
    assignments = np.zeros(keys.shape[0], dtype=int)

    for _ in range(max_iters):
        distances = np.linalg.norm(keys[:, None, :] - centroids[None, :, :], axis=2)
        new_assignments = np.argmin(distances, axis=1)
        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments
        for cluster_idx in range(num_anchors):
            mask = assignments == cluster_idx
            if np.any(mask):
                centroids[cluster_idx] = keys[mask].mean(axis=0)

    radii = np.zeros(num_anchors, dtype=float)
    for cluster_idx in range(num_anchors):
        mask = assignments == cluster_idx
        if np.any(mask):
            radii[cluster_idx] = float(
                np.max(np.linalg.norm(keys[mask] - centroids[cluster_idx], axis=1))
            )
    return {"assignments": assignments, "centroids": centroids, "radii": radii}


def coarse_elimination(
    query_sketch: np.ndarray,
    centroids: np.ndarray,
    cluster_radii: np.ndarray,
    theta: float,
    epsilon: float,
    sigma: float,
) -> np.ndarray:
    """Eliminate clusters whose coarse upper bound is below the threshold."""
    centroid_scores = sketch_similarity(query_sketch, centroids)
    cluster_bounds = compute_attention_upper_bound(centroid_scores, cluster_radii + epsilon, sigma)
    return eliminate_by_threshold(cluster_bounds, theta)


def token_level_elimination(
    query_sketch: np.ndarray,
    key_sketches: np.ndarray,
    theta: float,
    epsilon: float,
    sigma: float,
) -> np.ndarray:
    """Apply flat token-level elimination in sketch space."""
    token_scores = sketch_similarity(query_sketch, key_sketches)
    token_bounds = compute_attention_upper_bound(token_scores, epsilon, sigma)
    return eliminate_by_threshold(token_bounds, theta)


def hierarchical_token_elimination(
    query_sketch: np.ndarray,
    key_sketches: np.ndarray,
    num_anchors: int,
    theta: float,
    epsilon: float,
    sigma: float,
    seed: int = 0,
) -> dict[str, Any]:
    """Run hierarchical elimination with coarse cluster pruning followed by token filtering."""
    clustering = simple_anchor_clustering(key_sketches, num_anchors=num_anchors, seed=seed)
    cluster_eliminated = coarse_elimination(
        query_sketch=query_sketch,
        centroids=clustering["centroids"],
        cluster_radii=clustering["radii"],
        theta=theta,
        epsilon=epsilon,
        sigma=sigma,
    )

    assignments = clustering["assignments"]
    eliminated_mask = np.ones(key_sketches.shape[0], dtype=bool)
    for cluster_idx, cluster_is_eliminated in enumerate(cluster_eliminated):
        member_mask = assignments == cluster_idx
        if not np.any(member_mask):
            continue
        if cluster_is_eliminated:
            eliminated_mask[member_mask] = True
            continue
        local_eliminated = token_level_elimination(
            query_sketch=query_sketch,
            key_sketches=key_sketches[member_mask],
            theta=theta,
            epsilon=epsilon,
            sigma=sigma,
        )
        eliminated_mask[member_mask] = local_eliminated

    return {
        "eliminated_mask": eliminated_mask,
        "cluster_eliminated": cluster_eliminated,
        "assignments": assignments,
        "centroids": clustering["centroids"],
    }
