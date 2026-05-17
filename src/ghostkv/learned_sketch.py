# GhostKV Lab — github.com/manishklach/ghostkv-lab
# Patent: IN 202641062451
"""Learned sketch projection baselines for GhostKV attention experiments."""

from __future__ import annotations

import math

import torch
from torch import nn


class LearnedSketchProjection(nn.Module):
    """
    Learn per-head projection matrices for approximate attention sketching.

    This module is intentionally lightweight and experimental. The loss uses a
    differentiable proxy for rank preservation rather than exact Spearman
    correlation, and it upweights false-elimination-style mistakes relative to
    false resurrections.
    """

    def __init__(self, head_dim: int, sketch_dim: int, n_heads: int) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.sketch_dim = sketch_dim
        self.n_heads = n_heads

        weight = torch.randn(n_heads, head_dim, sketch_dim, dtype=torch.float32)
        weight = weight / math.sqrt(max(sketch_dim, 1))
        self.projection = nn.Parameter(weight)

    def forward(self, q_states: torch.Tensor, k_states: torch.Tensor) -> torch.Tensor:
        """
        Return sketch-space similarity scores.

        Args:
            q_states: Tensor with shape [heads, seq_len, head_dim].
            k_states: Tensor with shape [heads, seq_len, head_dim].
        """
        q_sketch = torch.einsum("hsd,hdk->hsk", q_states, self.projection)
        k_sketch = torch.einsum("hsd,hdk->hsk", k_states, self.projection)
        return torch.einsum("hsk,htk->hst", q_sketch, k_sketch) / math.sqrt(max(self.sketch_dim, 1))

    def loss(
        self,
        q_states: torch.Tensor,
        k_states: torch.Tensor,
        exact_scores: torch.Tensor,
        theta: float = 0.1,
        false_elim_weight: float = 10.0,
        false_resur_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute a proxy objective combining rank preservation and asymmetric filtering loss.

        The rank term is a centered Pearson-style correlation surrogate against
        exact-score ranks. The elimination term penalizes queries whose sketch
        maxima fall below `theta` despite exact maxima above `theta` more heavily
        than the reverse case.
        """
        approx_scores = self.forward(q_states, k_states)

        flat_exact = exact_scores.reshape(exact_scores.shape[0], -1)
        flat_approx = approx_scores.reshape(approx_scores.shape[0], -1)
        exact_rank = torch.argsort(torch.argsort(flat_exact, dim=-1), dim=-1).float()
        exact_rank = exact_rank - exact_rank.mean(dim=-1, keepdim=True)
        flat_approx_centered = flat_approx - flat_approx.mean(dim=-1, keepdim=True)
        denom = (
            exact_rank.norm(dim=-1) * flat_approx_centered.norm(dim=-1) + 1e-8
        )
        rank_corr = (exact_rank * flat_approx_centered).sum(dim=-1) / denom
        rank_loss = 1.0 - rank_corr.mean()

        exact_max = exact_scores.max(dim=-1).values
        approx_max = approx_scores.max(dim=-1).values
        relevant_mask = (exact_max > theta).float()
        irrelevant_mask = (exact_max <= theta).float()
        false_elim_penalty = torch.relu(theta - approx_max) * relevant_mask
        false_resur_penalty = torch.relu(approx_max - theta) * irrelevant_mask
        elimination_loss = (
            false_elim_weight * false_elim_penalty.mean()
            + false_resur_weight * false_resur_penalty.mean()
        )
        return rank_loss + elimination_loss
