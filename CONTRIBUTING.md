<!-- GhostKV Lab — github.com/manishklach/ghostkv-lab -->
<!-- Patent: IN 202641062451 -->

# Contributing

Thanks for your interest in improving GhostKV Lab.

This repository is a research-oriented evaluation harness. Contributions should improve experimental rigor, reproducibility, and interpretability without overstating the maturity of the underlying GhostKV concept.

## How to run experiments locally

### PowerShell

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m pytest
python experiments/generate_results.py
python experiments/real_attention_validation.py
python experiments/false_elimination_frontier.py
```

### WSL / Linux / macOS

WSL is recommended for reproducible experiment workflows and heavier validation runs.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
make demo
make real-validation
make frontier
```

From Windows, the same WSL workflow can be invoked explicitly:

```bash
wsl -e bash -c "pytest"
wsl -e bash -c "make demo"
wsl -e bash -c "make real-validation"
wsl -e bash -c "make frontier"
```

## Good first issues

- Issue A: "Capture Mistral-7B-Instruct Q/K tensors and run real_attention_validation"
- Issue B: "Implement learned sketch projections (see src/ghostkv/learned_sketch.py stub)"
- Issue C: "Add softmax denominator tracking to the simulator"

## How to add a new model

1. Add or update environment recommendation logic if the model has different memory requirements.
2. Extend architecture detection in the capture pipeline based on `model.config.model_type`.
3. Register hooks on the right attention submodules:
   - `q_proj` and `k_proj` for Llama-family models
   - fused `query_key_value` for GPTNeoX-style models
4. Confirm the captured tensors are raw Q/K representations before scaling or masking.
5. Add a small validation run with at least one prompt and one layer.
6. Update `RESULTS.md` only after the new model produces interpretable metrics.

## Results standards

Before a PR is merged, the branch should provide:

- passing `pytest`
- reproducible commands used for the experiment
- updated `RESULTS.md` if new public-facing results are added
- generated plots or a clear reason they were not produced
- explicit limitations and caveats for any claimed observation

## Patent notice

Do not remove patent attribution from any file where it already exists, and do not remove or rewrite the repository's factual patent notice sections.
