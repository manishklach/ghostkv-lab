"""Synthetic GhostKV decode simulator."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bandwidth import bandwidth_reduction_ratio, ghostkv_bytes, standard_kv_bytes
from .bounds import compute_attention_upper_bound, eliminate_by_threshold, false_elimination_rate, topk_overlap
from .lifecycle import GHOST, HOT, WARM, ghostify_tokens, transition_states
from .metrics import summarize_metrics
from .resurrection import estimate_resurrection_bytes, resurrect_candidates
from .sketches import project_query, random_projection_matrix, sketch_similarity


@dataclass
class SyntheticGhostKVSimulator:
    """Synthetic evaluation harness for GhostKV-style elimination."""

    num_tokens: int
    dim: int
    sketch_dim: int
    hot_window: int
    theta_elim: float
    epsilon: float
    sigma: float
    seed: int = 0
    warm_window: int | None = None
    ghost_record_bytes: int | None = None
    rng: np.random.Generator = field(init=False)
    keys: np.ndarray = field(init=False)
    projection_matrix: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.warm_window = self.warm_window or self.num_tokens
        self.ghost_record_bytes = self.ghost_record_bytes or (self.sketch_dim * 4 + 16)
        self.projection_matrix = random_projection_matrix(self.dim, self.sketch_dim, seed=self.seed)
        self.keys = self.generate_synthetic_kv()

    def generate_synthetic_kv(self) -> np.ndarray:
        """Generate synthetic key vectors."""
        return self.rng.normal(size=(self.num_tokens, self.dim))

    def generate_query(self) -> np.ndarray:
        """Generate a synthetic decode-time query vector."""
        return self.rng.normal(size=(self.dim,))

    def run_decode_step(self) -> dict[str, float]:
        """Run a single synthetic decode step and return metrics."""
        query = self.generate_query()
        exact_scores = (self.keys @ query) / np.sqrt(self.dim)

        ages = np.arange(self.num_tokens - 1, -1, -1)
        states = transition_states(ages, hot_window=self.hot_window, warm_window=self.warm_window)
        hot_mask = states == HOT
        candidate_mask = (states == WARM) | (states == GHOST)

        candidate_keys = self.keys[candidate_mask]
        candidate_exact_scores = exact_scores[candidate_mask]

        query_sketch = project_query(query, self.projection_matrix)
        ghost_records = ghostify_tokens(
            candidate_keys,
            self.projection_matrix,
            anchor_ids=np.flatnonzero(candidate_mask),
            epsilon=self.epsilon,
            sigma=self.sigma,
        )

        if ghost_records:
            sketch_matrix = np.vstack([record.sketch for record in ghost_records])
            epsilon_res = np.array([record.epsilon_res for record in ghost_records], dtype=float)
            sigma_anchor = np.array([record.sigma_anchor for record in ghost_records], dtype=float)
            sketch_scores = sketch_similarity(query_sketch, sketch_matrix)
            bounds = compute_attention_upper_bound(sketch_scores, epsilon_res, sigma_anchor)
            eliminated_mask = eliminate_by_threshold(bounds, self.theta_elim)
            survivor_mask = ~eliminated_mask
            resurrected = resurrect_candidates(ghost_records, survivor_mask)
        else:
            sketch_scores = np.array([], dtype=float)
            bounds = np.array([], dtype=float)
            eliminated_mask = np.array([], dtype=bool)
            survivor_mask = np.array([], dtype=bool)
            resurrected = []

        approx_scores = np.full_like(exact_scores, fill_value=-np.inf, dtype=float)
        approx_scores[hot_mask] = exact_scores[hot_mask]
        candidate_indices = np.flatnonzero(candidate_mask)
        if candidate_indices.size:
            approx_scores[candidate_indices[survivor_mask]] = exact_scores[candidate_indices[survivor_mask]]

        topk = min(32, self.num_tokens)
        true_threshold = np.partition(exact_scores, -topk)[-topk]
        fer = (
            false_elimination_rate(candidate_exact_scores, eliminated_mask, true_threshold)
            if candidate_exact_scores.size
            else 0.0
        )

        num_candidates = int(candidate_mask.sum())
        num_hot = int(hot_mask.sum())
        num_resurrected = len(resurrected)
        kv_bytes_per_token = standard_kv_bytes(1, self.dim)
        standard_bytes = standard_kv_bytes(self.num_tokens, self.dim)
        ghost_bytes = ghostkv_bytes(
            num_hot=num_hot,
            num_resurrected=num_resurrected,
            kv_bytes_per_token=kv_bytes_per_token,
            num_ghost=num_candidates,
            ghost_record_bytes=self.ghost_record_bytes,
        )
        resurrection_bytes = estimate_resurrection_bytes(num_resurrected, kv_bytes_per_token)

        return {
            "elimination_rate": float(np.mean(eliminated_mask)) if num_candidates else 0.0,
            "resurrection_rate": float(np.mean(survivor_mask)) if num_candidates else 0.0,
            "false_elimination_rate": fer,
            "topk_overlap": topk_overlap(exact_scores, approx_scores, topk),
            "estimated_standard_bytes": standard_bytes,
            "estimated_ghostkv_bytes": ghost_bytes,
            "estimated_resurrection_bytes": resurrection_bytes,
            "bandwidth_reduction": bandwidth_reduction_ratio(standard_bytes, ghost_bytes),
        }

    def run_many_steps(self, num_steps: int) -> dict[str, float]:
        """Run many synthetic decode steps and summarize the results."""
        per_step: dict[str, list[float]] = {}
        for _ in range(num_steps):
            step_metrics = self.run_decode_step()
            for key, value in step_metrics.items():
                per_step.setdefault(key, []).append(value)

        summary = summarize_metrics(per_step)
        summary["num_steps"] = float(num_steps)
        return summary

