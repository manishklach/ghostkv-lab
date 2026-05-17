"""Run a multi-step synthetic decode simulation."""

from __future__ import annotations

import argparse

from ghostkv.metrics import pretty_print_table
from ghostkv.simulator import SyntheticGhostKVSimulator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tokens", type=int, default=32768)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--sketch-dim", type=int, default=32)
    parser.add_argument("--hot-window", type=int, default=2048)
    parser.add_argument("--theta-elim", type=float, default=0.3)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    simulator = SyntheticGhostKVSimulator(
        num_tokens=args.num_tokens,
        dim=args.dim,
        sketch_dim=args.sketch_dim,
        hot_window=args.hot_window,
        theta_elim=args.theta_elim,
        epsilon=args.epsilon,
        sigma=args.sigma,
        seed=args.seed,
    )
    summary = simulator.run_many_steps(args.steps)

    rows = [{"metric": key, "value": f"{value:.6f}"} for key, value in summary.items()]
    print("GhostKV synthetic decode simulation")
    print(pretty_print_table(rows, ["metric", "value"]))


if __name__ == "__main__":
    main()

