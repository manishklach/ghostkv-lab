"""Illustrative bandwidth model comparison for full KV, quantized KV, and GhostKV."""

from __future__ import annotations

import argparse

from ghostkv.bandwidth import (
    bandwidth_reduction_ratio,
    ghostkv_bytes,
    quantized_kv_bytes,
    standard_kv_bytes,
)
from ghostkv.metrics import pretty_print_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, default=128 * 1024)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--hot-window", type=int, default=4096)
    parser.add_argument("--ghost-record-bytes", type=float, default=64.0)
    parser.add_argument("--quant-factor", type=float, default=4.0)
    parser.add_argument("--kv-bytes-per-token", type=float, default=None)
    args = parser.parse_args()

    kv_bytes_per_token = args.kv_bytes_per_token
    if kv_bytes_per_token is None:
        kv_bytes_per_token = standard_kv_bytes(1, args.dim)

    full_kv = standard_kv_bytes(args.context, args.dim)
    int4_like = quantized_kv_bytes(args.context, args.dim, quant_factor=args.quant_factor)

    rows: list[dict[str, object]] = [
        {
            "scheme": "full_kv",
            "resurrection_rate": "-",
            "bytes": f"{full_kv:,.0f}",
            "reduction_vs_full": f"{0.0:.3f}",
        },
        {
            "scheme": "int4_style",
            "resurrection_rate": "-",
            "bytes": f"{int4_like:,.0f}",
            "reduction_vs_full": f"{bandwidth_reduction_ratio(full_kv, int4_like):.3f}",
        },
    ]

    num_ghost = args.context - args.hot_window
    for resurrection_rate in (0.005, 0.01, 0.02, 0.05):
        num_resurrected = int(num_ghost * resurrection_rate)
        ghost_bytes = ghostkv_bytes(
            num_hot=args.hot_window,
            num_resurrected=num_resurrected,
            kv_bytes_per_token=kv_bytes_per_token,
            num_ghost=num_ghost,
            ghost_record_bytes=args.ghost_record_bytes,
        )
        rows.append(
            {
                "scheme": "ghostkv",
                "resurrection_rate": f"{resurrection_rate * 100:.1f}%",
                "bytes": f"{ghost_bytes:,.0f}",
                "reduction_vs_full": f"{bandwidth_reduction_ratio(full_kv, ghost_bytes):.3f}",
            }
        )

    print("GhostKV bandwidth model demo")
    print(pretty_print_table(rows, ["scheme", "resurrection_rate", "bytes", "reduction_vs_full"]))


if __name__ == "__main__":
    main()

