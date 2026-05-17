"""Memory-footprint and movement models for GhostKV experiments."""

from __future__ import annotations


def standard_kv_bytes(
    num_tokens: int,
    dim: int,
    num_layers: int = 1,
    num_heads: int = 1,
    dtype_bytes: int = 2,
) -> float:
    """Estimate bytes for full K and V storage."""
    return float(num_tokens * dim * 2 * num_layers * num_heads * dtype_bytes)


def quantized_kv_bytes(num_tokens: int, dim: int, quant_factor: float) -> float:
    """Estimate bytes for a quantized KV representation."""
    if quant_factor <= 0:
        raise ValueError("quant_factor must be positive.")
    return standard_kv_bytes(num_tokens=num_tokens, dim=dim) / float(quant_factor)


def ghostkv_bytes(
    num_hot: int,
    num_resurrected: int,
    kv_bytes_per_token: float,
    num_ghost: int,
    ghost_record_bytes: float,
) -> float:
    """Estimate bytes touched by a GhostKV decode step."""
    live_kv_bytes = (num_hot + num_resurrected) * kv_bytes_per_token
    witness_bytes = num_ghost * ghost_record_bytes
    return float(live_kv_bytes + witness_bytes)


def bandwidth_reduction_ratio(standard_bytes: float, ghost_bytes: float) -> float:
    """Return the fractional reduction relative to the standard KV path."""
    if standard_bytes <= 0:
        raise ValueError("standard_bytes must be positive.")
    return float(1.0 - (ghost_bytes / standard_bytes))

