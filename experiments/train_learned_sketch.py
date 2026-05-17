# GhostKV Lab — github.com/manishklach/ghostkv-lab
# Patent: IN 202641062451
"""Train lightweight learned sketch projections on captured Q/K tensors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ghostkv.learned_sketch import LearnedSketchProjection
from ghostkv.sketches import random_projection_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "real_attention"
WEIGHTS_DIR = REPO_ROOT / "data" / "learned_sketches"
OUTPUT_DIR = REPO_ROOT / "results" / "modern"


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _compute_exact_scores(q_states: torch.Tensor, k_states: torch.Tensor) -> torch.Tensor:
    head_dim = q_states.shape[-1]
    return torch.einsum("hsd,htd->hst", q_states, k_states) / math.sqrt(max(head_dim, 1))


def _compute_random_scores(q_states: torch.Tensor, k_states: torch.Tensor, sketch_dim: int, seed: int) -> torch.Tensor:
    projection = torch.tensor(
        random_projection_matrix(q_states.shape[-1], sketch_dim, seed=seed),
        dtype=torch.float32,
    )
    q_sketch = torch.einsum("hsd,dk->hsk", q_states, projection)
    k_sketch = torch.einsum("hsd,dk->hsk", k_states, projection)
    return torch.einsum("hsk,htk->hst", q_sketch, k_sketch) / math.sqrt(max(sketch_dim, 1))


def _false_elimination_rate(exact_scores: torch.Tensor, approx_scores: torch.Tensor, theta: float) -> float:
    exact_max = exact_scores.max(dim=-1).values
    approx_max = approx_scores.max(dim=-1).values
    eliminated = approx_max < theta
    eliminated_count = int(eliminated.sum().item())
    if eliminated_count == 0:
        return 0.0
    false_eliminated = torch.logical_and(eliminated, exact_max > theta)
    return float(false_eliminated.sum().item() / eliminated_count)


def _load_layer_examples(data_dir: Path) -> dict[tuple[str, int], list[tuple[torch.Tensor, torch.Tensor]]]:
    examples: dict[tuple[str, int], list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for path in sorted(data_dir.glob("*.npz")):
        payload = np.load(path, allow_pickle=True)
        model_id = str(payload["model_id"].item())
        layer_idx = int(payload["layer_idx"].item())
        q_states = torch.tensor(payload["q"], dtype=torch.float32).unsqueeze(0)
        k_states = torch.tensor(payload["k"], dtype=torch.float32).unsqueeze(0)
        examples.setdefault((model_id, layer_idx), []).append((q_states, k_states))
    return examples


def _truncate_example(
    q_states: torch.Tensor,
    k_states: torch.Tensor,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    effective_len = min(max_seq_len, q_states.shape[1], k_states.shape[1])
    return q_states[:, :effective_len, :].contiguous(), k_states[:, :effective_len, :].contiguous()


def _train_for_layer(
    model_id: str,
    layer_idx: int,
    examples: list[tuple[torch.Tensor, torch.Tensor]],
    sketch_dim: int,
    epochs: int,
    seed: int,
    max_seq_len: int,
) -> tuple[LearnedSketchProjection, dict[str, float]]:
    split_idx = max(1, int(len(examples) * 0.8))
    train_examples = examples[:split_idx]
    eval_examples = examples[split_idx:] or examples[-1:]

    head_dim = int(train_examples[0][0].shape[-1])
    model = LearnedSketchProjection(head_dim=head_dim, sketch_dim=sketch_dim, n_heads=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for _epoch in range(epochs):
        for q_states, k_states in train_examples:
            q_states, k_states = _truncate_example(q_states, k_states, max_seq_len=max_seq_len)
            exact_scores = _compute_exact_scores(q_states, k_states)
            optimizer.zero_grad()
            loss = model.loss(q_states, k_states, exact_scores, theta=0.1)
            loss.backward()
            optimizer.step()

    learned_rates: list[float] = []
    random_rates: list[float] = []
    with torch.no_grad():
        for example_idx, (q_states, k_states) in enumerate(eval_examples):
            q_states, k_states = _truncate_example(q_states, k_states, max_seq_len=max_seq_len)
            exact_scores = _compute_exact_scores(q_states, k_states)
            learned_scores = model(q_states, k_states)
            random_scores = _compute_random_scores(q_states, k_states, sketch_dim=sketch_dim, seed=seed + example_idx + layer_idx)
            learned_rates.append(_false_elimination_rate(exact_scores, learned_scores, theta=0.1))
            random_rates.append(_false_elimination_rate(exact_scores, random_scores, theta=0.1))

    return model, {
        "model_id": model_id,
        "layer_idx": float(layer_idx),
        "learned_false_elimination_rate": float(np.mean(learned_rates)),
        "random_false_elimination_rate": float(np.mean(random_rates)),
    }


def _plot_comparison(results_df: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-paper")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x_positions = np.arange(len(results_df))
    width = 0.35
    ax.bar(
        x_positions - width / 2,
        results_df["random_false_elimination_rate"],
        width=width,
        label="random",
    )
    ax.bar(
        x_positions + width / 2,
        results_df["learned_false_elimination_rate"],
        width=width,
        label="learned",
    )
    ax.set_title("Learned vs Random Sketch False Elimination")
    ax.set_xlabel("Layer")
    ax.set_ylabel("False Elimination Rate at θ=0.1")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(int(layer)) for layer in results_df["layer_idx"]])
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--weights-dir", type=Path, default=WEIGHTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--sketch-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(args.seed)
    layer_examples = _load_layer_examples(args.data_dir)
    rows: list[dict[str, float]] = []
    for (model_id, layer_idx), examples in sorted(layer_examples.items(), key=lambda item: item[0][1]):
        learned_model, row = _train_for_layer(
            model_id=model_id,
            layer_idx=layer_idx,
            examples=examples,
            sketch_dim=args.sketch_dim,
            epochs=args.epochs,
            seed=args.seed,
            max_seq_len=args.max_seq_len,
        )
        target_dir = args.weights_dir / model_id.replace("/", "__")
        target_dir.mkdir(parents=True, exist_ok=True)
        torch.save(learned_model.state_dict(), target_dir / f"layer{layer_idx}.pt")
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values("layer_idx")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output_dir / "learned_vs_random_sketch.csv", index=False)
    (args.output_dir / "learned_vs_random_sketch.json").write_text(
        json.dumps(results_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    _plot_comparison(results_df, args.output_dir / "learned_vs_random_sketch.png")
    print(f"Wrote {args.output_dir / 'learned_vs_random_sketch.png'}")


if __name__ == "__main__":
    main()
