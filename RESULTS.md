# GhostKV Synthetic Results

## Status

Synthetic simulator working. Real-model validation pending.

## Important disclaimer

These are synthetic simulation results, not real-model results.

These results use synthetic key/query tensors. They test whether the GhostKV evaluation pipeline works and whether sketch-based elimination behaves plausibly. They do not prove production speedups or model-quality preservation.

## Experiment 1: Sketch quality audit

This sweep varies `sketch_dim` across `8, 16, 32, 64` and measures how well sketch-space ranking preserves top-k exact score membership. Higher sketch dimensions should usually improve overlap, though synthetic noise and random projections can still introduce variance.

| sketch_dim | top8_overlap | top16_overlap | top32_overlap | top64_overlap |
| --- | --- | --- | --- | --- |
| 8 | 0.0 | 0.0 | 0.0625 | 0.0625 |
| 16 | 0.125 | 0.125 | 0.15625 | 0.125 |
| 32 | 0.0 | 0.0 | 0.03125 | 0.09375 |
| 64 | 0.0 | 0.0 | 0.09375 | 0.125 |

Plot: [results/sketch_dim_vs_topk_overlap.png](results/sketch_dim_vs_topk_overlap.png)

## Experiment 2: Elimination tradeoff

This sweep varies `theta` from `0.1` to `0.9` and reports elimination rate, resurrection rate, false elimination rate, and top-k overlap. The table below shows representative rows; the full grid is stored in `results/elimination_tradeoff.csv`.

| theta | sketch_dim | elimination_rate | resurrection_rate | false_elimination_rate | topk_overlap |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 16 | 0.500301 | 0.499699 | 0.140824 | 0.87625 |
| 0.1 | 32 | 0.50079 | 0.49921 | 0.055209 | 0.9525 |
| 0.1 | 64 | 0.499043 | 0.500957 | 0.009888 | 0.99125 |
| 0.5 | 16 | 0.521761 | 0.478239 | 0.157022 | 0.8625 |
| 0.5 | 32 | 0.538465 | 0.461535 | 0.073759 | 0.93625 |
| 0.5 | 64 | 0.566002 | 0.433998 | 0.01551 | 0.98625 |
| 0.9 | 16 | 0.543491 | 0.456509 | 0.165759 | 0.855 |
| 0.9 | 32 | 0.575751 | 0.424249 | 0.089123 | 0.9225 |
| 0.9 | 64 | 0.630815 | 0.369185 | 0.026856 | 0.97625 |

Plot: [results/theta_vs_elimination_rate.png](results/theta_vs_elimination_rate.png)

## Experiment 3: Bandwidth model

This illustrative model compares full KV traffic, an INT4-style compressed baseline, and GhostKV under several resurrection rates. The purpose is to show why query-time movement reduction can matter even when exact attention still runs on survivors.

| scheme | resurrection_rate | bytes | reduction_vs_full |
| --- | --- | --- | --- |
| full_kv | 0.0 | 67108864.0 | 0.0 |
| int4_style | 0.0 | 16777216.0 | 0.75 |
| ghostkv | 0.005 | 10548224.0 | 0.842819 |
| ghostkv | 0.01 | 10873344.0 | 0.837975 |
| ghostkv | 0.02 | 11523584.0 | 0.828285 |
| ghostkv | 0.05 | 13473792.0 | 0.799225 |

Plot: [results/resurrection_rate_vs_bandwidth.png](results/resurrection_rate_vs_bandwidth.png)

## Interpretation

- Synthetic results validate the harness, not the algorithm.
- Higher sketch dimensions are a positive sign when they preserve top-k overlap more reliably.
- Elimination is encouraging only when false elimination remains controlled while useful pruning still occurs.
- The bandwidth model is illustrative, but it helps motivate why reducing movement may matter more than only compressing bytes at rest.

## Limitations

- Random tensors are not transformer tensors.
- Real attention distributions may be more structured or more adversarial.
- Softmax denominator handling is not fully modeled.
- Resurrection latency is estimated, not benchmarked.
- No HuggingFace or real LLM validation exists in this repository yet.

## Next milestone

Real K/Q tensor capture from a small transformer:

- GPT-2 small or TinyLlama
- capture attention Q/K tensors
- compare exact QK ranking vs sketch ranking
- evaluate false elimination rate on real attention
