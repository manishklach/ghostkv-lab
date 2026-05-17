"""Resurrection helpers for synthetic GhostKV experiments."""

from __future__ import annotations

from typing import Iterable

from .lifecycle import GhostRecord


def resurrect_candidates(
    ghost_records: Iterable[GhostRecord], survivor_mask: Iterable[bool]
) -> list[GhostRecord]:
    """Return the ghost records that survive elimination."""
    return [record for record, keep in zip(ghost_records, survivor_mask) if keep]


def estimate_resurrection_bytes(num_survivors: int, kv_bytes_per_token: float) -> float:
    """Estimate the bytes of full KV state that must be resurrected."""
    return float(num_survivors * kv_bytes_per_token)


def placeholder_anchor_residual_reconstruct() -> None:
    """Document the future reconstruction path.

    In this repository, resurrection is simulated by re-admitting surviving tokens
    into the exact attention set. No real anchor-based reconstruction pipeline is
    implemented yet. A future version could model anchor-local residual storage,
    decompression cost, and tiered-memory fetch latency.
    """

