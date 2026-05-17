# Contributing

Thanks for your interest in improving GhostKV Lab.

This repository is a research-oriented evaluation harness. Contributions that improve experimental clarity, reproducibility, and honest reporting are especially valuable.

## Setup

Install locally from the repo root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
```

## Run Tests

```bash
python -m pytest
```

## Run Synthetic Results

```bash
make results
```

If `make` is unavailable in your shell:

```bash
python experiments/generate_results.py
```

## Run Frontier Analysis

```bash
make frontier
```

If `make` is unavailable in your shell:

```bash
python experiments/false_elimination_frontier.py
```

Frontier outputs are written under `results/frontier/`.

## Good First Issues

- Add TinyLlama Q/K capture experiment
- Implement learned sketch projections
- Improve hierarchical anchor clustering
- Add softmax denominator mass tracking
- Add Llama/Mistral attention export scripts

## Contribution Style

- Keep the tone research-oriented and non-hypey.
- Prefer explicit limitations over optimistic framing.
- Avoid claims about runtime benefit unless backed by the right measurements.
- Keep synthetic and real-attention results clearly separated.

## Commit Message Suggestion

`Add false-elimination frontier analysis for real attention tensors`

