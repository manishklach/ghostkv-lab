# GhostKV Results

## Status

Synthetic simulator working. Real-attention validation supported on lightweight HuggingFace models.

## Important disclaimer

Synthetic results and real-model attention-validation results are reported separately below.

This repository does not benchmark throughput and does not establish production viability. The real-model path focuses on attention-ranking preservation and bounded elimination behavior on captured Q/K tensors.

## Key Findings So Far

- Low-dimensional sketches tend to preserve coarse similarity structure more reliably than exact top-attention ranking.
- False elimination remains the primary challenge in bounded filtering.
- Head-wise behavior varies significantly across layers and prompts.
- Real transformer tensors can behave differently from Gaussian synthetic tensors.
- The current simple hierarchical baseline does not yet outperform flat elimination, but the design space remains open.

## Synthetic Results

These are synthetic simulation results, not real-model results.

The synthetic experiments use Gaussian key/query tensors to validate the harness and to study threshold sensitivity under controlled conditions. They do not prove production speedups or model-quality preservation.

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

## Real Attention Validation

The real-attention path captures Q/K tensors from lightweight HuggingFace transformer models and measures sketch behavior on actual attention states rather than Gaussian tensors.

| layer_idx | topk_overlap_mean | rank_correlation_mean | false_elimination_rate_mean | elimination_rate_mean |
| --- | --- | --- | --- | --- |
| 0 | 0.804167 | 0.9506 | 0.275 | 0.347407 |
| 3 | 0.772917 | 0.943453 | 0.622917 | 0.786142 |
| 6 | 0.747917 | 0.935672 | 0.323958 | 0.541508 |
| 9 | 0.729167 | 0.930016 | 0.179167 | 0.294509 |

Plots:

- [results/real_attention_topk_overlap.png](results/real_attention_topk_overlap.png)
- [results/real_attention_false_elimination.png](results/real_attention_false_elimination.png)
- [results/real_attention_layerwise_overlap.png](results/real_attention_layerwise_overlap.png)
- [results/head_variance.png](results/head_variance.png)
- [results/real_attention_summary.md](results/real_attention_summary.md)

Observations:

- Random projections often preserve broad similarity structure better than exact top-k ordering.
- Real transformer tensors are layer-dependent and head-dependent.
- Threshold choice can change false elimination behavior materially.

## Hierarchical Filtering

The hierarchical experiment adds simple anchor grouping before token-level elimination to test whether coarse filtering can improve elimination quality.

| method | theta | false_elimination_rate_mean | elimination_rate_mean |
| --- | --- | --- | --- |
| flat | 0.1 | 0.407552 | 0.524548 |
| flat | 0.3 | 0.40842 | 0.524847 |
| flat | 0.5 | 0.41059 | 0.52568 |
| flat | 0.7 | 0.411458 | 0.526594 |
| flat | 0.9 | 0.41276 | 0.527562 |
| hierarchical | 0.1 | 0.440104 | 0.544109 |
| hierarchical | 0.3 | 0.440538 | 0.544242 |
| hierarchical | 0.5 | 0.442274 | 0.544942 |
| hierarchical | 0.7 | 0.443142 | 0.545855 |
| hierarchical | 0.9 | 0.444444 | 0.546823 |

Plot:

- [results/hierarchical_vs_flat.png](results/hierarchical_vs_flat.png)

Current note:

- In the present lightweight baseline, hierarchical filtering remains exploratory and does not yet improve false elimination behavior relative to flat filtering.

## Limitations And Next Steps

- Random tensors are not transformer tensors.
- GPT-2 is not representative of all modern LLMs.
- Small models differ from large long-context models.
- Real attention distributions may be more structured or more adversarial.
- Softmax denominator handling is not fully modeled.
- Resurrection latency is estimated, not benchmarked.
- No actual memory-movement reduction is measured on hardware.
- Resurrection remains simulated.
- No FlashAttention integration exists yet.

## Next milestone

- Expand real-model validation beyond GPT-2.
- Capture real attention tensors from additional decoder architectures.
- Compare layer/head fragility across models and prompt families.
- Study hierarchical ghost indexes and learned sketch functions.

## Modern Architecture Validation

The modern real-attention pipeline uses architecture-aware hooks to capture raw query/key projections and evaluate sketch preservation on non-synthetic tensors. In this environment, the memory-safe loader fell back to GPT-2, so broader modern-model validation remains pending even though the hook pathway now supports larger decoder families.

### Summary table

| layer_idx | false_elimination_rate | elimination_rate | top32_overlap | rank_correlation |
| --- | --- | --- | --- | --- |
| 0 | 0.0000 | 0.0000 | 0.2106 | 0.9523 |
| 4 | 0.0370 | 0.0000 | 0.0926 | 0.8990 |
| 8 | 0.0000 | 0.0000 | 0.1481 | 0.9169 |

### Key findings

- Current modern-capture runs resolved to `gpt2` (requested `meta-llama/Llama-3.2-1B-Instruct`).
- Mean rank correlation is `0.8699`, while mean top-32 overlap is `0.1154`.
- Mean elimination rate is only `0.000046` under the current positive-threshold sweep, with a maximum observed elimination rate of `0.003906`.
- Mean false elimination rate is `0.0086`, but the current run does not reach a meaningful elimination regime; the methodology is currently more informative than the operating point.
- Requested context lengths were `[512, 1024, 2048]`, with effective lengths `[512, 1024]` after model-specific truncation.
- Maximum observed false elimination rate in the captured rows was `1.0000`.

### Plots

- [results/modern/false_elim_vs_theta_by_layer.png](results/modern/false_elim_vs_theta_by_layer.png)
- [results/modern/elimination_rate_vs_false_elim.png](results/modern/elimination_rate_vs_false_elim.png)
- [results/modern/topk_overlap_heatmap.png](results/modern/topk_overlap_heatmap.png)
- [results/modern/context_length_scaling.png](results/modern/context_length_scaling.png)
- [results/modern/sketch_dim_comparison.png](results/modern/sketch_dim_comparison.png)
- [results/real_attention_modern/metrics.csv](results/real_attention_modern/metrics.csv)
- [results/real_attention_modern/summary.json](results/real_attention_modern/summary.json)
