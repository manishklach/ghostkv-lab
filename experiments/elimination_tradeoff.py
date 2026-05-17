"""Sweep elimination thresholds and sketch dimensions."""

from __future__ import annotations

import argparse

import numpy as np

from ghostkv.metrics import pretty_print_table
from ghostkv.simulator import SyntheticGhostKVSimulator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tokens", type=int, default=16384)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--hot-window", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.05)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for sketch_dim in (8, 16, 32, 64):
        for theta in np.linspace(0.1, 0.9, num=9):
            simulator = SyntheticGhostKVSimulator(
                num_tokens=args.num_tokens,
                dim=args.dim,
                sketch_dim=sketch_dim,
                hot_window=args.hot_window,
                theta_elim=float(theta),
                epsilon=args.epsilon,
                sigma=args.sigma,
                seed=args.seed + sketch_dim,
            )
            metrics = simulator.run_many_steps(args.steps)
            rows.append(
                {
                    "sketch_dim": sketch_dim,
                    "theta": f"{theta:.1f}",
                    "elim": f"{metrics['elimination_rate_mean']:.3f}",
                    "resurrect": f"{metrics['resurrection_rate_mean']:.3f}",
                    "false_elim": f"{metrics['false_elimination_rate_mean']:.3f}",
                    "topk": f"{metrics['topk_overlap_mean']:.3f}",
                    "bw_reduction": f"{metrics['bandwidth_reduction_mean']:.3f}",
                }
            )

    print("GhostKV elimination tradeoff sweep")
    print(
        pretty_print_table(
            rows,
            ["sketch_dim", "theta", "elim", "resurrect", "false_elim", "topk", "bw_reduction"],
        )
    )


if __name__ == "__main__":
    main()

