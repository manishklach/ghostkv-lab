"""Lifecycle state helpers for GhostKV token residency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .sketches import project_keys

HOT = "hot"
WARM = "warm"
GHOST = "ghost"
ARCHIVE = "archive"


@dataclass(frozen=True)
class GhostRecord:
    """Compact witness metadata for a cold KV entry."""

    token_id: int
    sketch: np.ndarray
    anchor_id: int
    epsilon_res: float
    sigma_anchor: float
    state: str


def ghostify_tokens(
    keys: np.ndarray,
    projection_matrix: np.ndarray,
    anchor_ids: Optional[Iterable[int]] = None,
    epsilon: float = 0.05,
    sigma: float = 0.05,
) -> list[GhostRecord]:
    """Convert full keys into synthetic ghost records."""
    key_sketches = project_keys(keys, projection_matrix)
    num_tokens = keys.shape[0]
    anchor_array = np.arange(num_tokens) if anchor_ids is None else np.asarray(list(anchor_ids))
    if anchor_array.shape[0] != num_tokens:
        raise ValueError("anchor_ids must match the number of tokens.")

    records: list[GhostRecord] = []
    for token_id, sketch in enumerate(key_sketches):
        records.append(
            GhostRecord(
                token_id=token_id,
                sketch=np.asarray(sketch, dtype=float),
                anchor_id=int(anchor_array[token_id]),
                epsilon_res=float(epsilon),
                sigma_anchor=float(sigma),
                state=GHOST,
            )
        )
    return records


def transition_states(token_ages: np.ndarray, hot_window: int, warm_window: int) -> np.ndarray:
    """Assign states by recency using a simple four-tier heuristic.

    Ages are measured in decode steps since last use. Lower is newer.
    Tokens newer than `hot_window` are hot. The next `warm_window` are warm.
    The following `warm_window` are ghost candidates. Older tokens are archived.
    """
    token_ages = np.asarray(token_ages)
    states = np.empty(token_ages.shape, dtype=object)
    states[token_ages < hot_window] = HOT
    warm_mask = (token_ages >= hot_window) & (token_ages < hot_window + warm_window)
    states[warm_mask] = WARM
    ghost_mask = (token_ages >= hot_window + warm_window) & (
        token_ages < hot_window + (2 * warm_window)
    )
    states[ghost_mask] = GHOST
    states[token_ages >= hot_window + (2 * warm_window)] = ARCHIVE
    return states

