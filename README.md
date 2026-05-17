# GhostKV Lab

“A research simulator for query-time bounded elimination of reconstructable KV-cache witnesses in long-context transformer inference.”

GhostKV Lab is a lightweight Python repository for studying whether sketch-based bounded elimination can reduce KV-cache memory movement while preserving attention quality in long-context decode workloads. It is built as a synthetic evaluation harness first: no heavyweight model downloads, no kernel claims, and no fabricated benchmark results.

## Current Status

Current status:

- Synthetic GhostKV simulator: working
- Sketch quality audit: working
- Elimination tradeoff sweep: working
- Bandwidth model: working
- Real-model validation: pending
- GPU kernel integration: pending

## What GhostKV Is

GhostKV is a systems-oriented hypothesis for KV-cache handling during decode:

- Cold KV-cache entries are converted into compact ghost records.
- Each ghost record stores an attention sketch vector, a semantic anchor identifier, and a residual uncertainty term.
- At query time, the simulator computes a conservative attention upper bound for each ghost record:

`AttnUB(Q, G_i) = sketch_sim(Q, G_i.sketch) + epsilon_res_i + sigma_anchor_i`

- Ghost tokens with an upper bound below `theta_elim` are eliminated.
- Surviving ghost records are resurrected and included in exact attention.

The key property in this repository is exactness over survivors: approximation is confined to the elimination stage. Once candidates survive elimination, the simulator treats attention over `hot + resurrected` tokens as exact.

## What GhostKV Is Not

- Not a production LLM runtime
- Not a CUDA kernel implementation
- Not a proof of speedup
- Not a substitute for real-model validation

This repository uses synthetic tensors first. Real-model validation is future work.

## Architecture

```text
KV Cache
  |
  +--> Hot / Warm / Ghost / Archive
                    |
Query --> Sketch --> Bound --> Eliminate or Resurrect --> Exact Attend
```

## Repository Layout

```text
ghostkv-lab/
  docs/
  src/ghostkv/
  experiments/
  tests/
  results/
  data/
```

## Quickstart

These are the commands validated for this repository from the repo root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m pytest
python experiments/sketch_quality_audit.py
python experiments/elimination_tradeoff.py
python experiments/bandwidth_model_demo.py
python experiments/synthetic_decode_simulation.py
python experiments/generate_results.py
```

If you prefer not to create a virtual environment, the same install and run commands work with the active Python environment as long as it is Python 3.10+.

## Core Idea

Ghost records are compact witnesses for cold KV entries:

1. `attention sketch vector`
2. `semantic anchor id`
3. `residual uncertainty value`

At each decode step:

1. Project the query into sketch space.
2. Compute conservative upper bounds for ghost records.
3. Eliminate records with bounds below `theta_elim`.
4. Resurrect survivors.
5. Run exact attention over hot tokens plus resurrected tokens.

## Why This Repo Exists

Long-context inference can become bottlenecked by KV-cache movement rather than only by arithmetic throughput. This repository exists to evaluate whether bounded elimination can reduce the amount of KV state that must be moved or re-read on each decode step without aggressively approximating the final attention calculation.

## Experiments

- `experiments/sketch_quality_audit.py`: compares exact scores and sketch-space scores across sketch dimensions
- `experiments/elimination_tradeoff.py`: sweeps elimination thresholds and sketch dimensions
- `experiments/bandwidth_model_demo.py`: compares illustrative memory footprints for full KV, quantized KV, and GhostKV
- `experiments/synthetic_decode_simulation.py`: runs a multi-step decode simulation and summarizes aggregate metrics
- `experiments/generate_results.py`: regenerates synthetic CSV outputs, PNG plots, and `RESULTS.md`

All experiments use synthetic tensors and are intended to inform feasibility, not to claim production benefit.

## Generate Results

```bash
make demo
```

This runs the test suite and then generates synthetic CSV outputs, PNG plots, and a refreshed [RESULTS.md](RESULTS.md) summary. If you only want to regenerate artifacts, use `make results`.

If `make` is not available in your shell, the equivalent commands are:

```bash
python -m pytest
python experiments/generate_results.py
```

## Roadmap

- Integrate HuggingFace attention capture
- LongBench evaluation
- Needle-in-a-Haystack validation
- FlashAttention-compatible prototype
- CXL / near-memory simulation

Additional detail is in [docs/roadmap.md](docs/roadmap.md).

## Development Notes

- Python 3.10+
- Main dependencies: `numpy`, `matplotlib`
- Test runner: `pytest`
- Editable install supported via `pip install -e ".[dev]"`

## License

MIT. See [LICENSE](LICENSE).
