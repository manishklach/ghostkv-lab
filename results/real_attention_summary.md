# Real Attention Validation Summary

These are real-model attention validation results, not throughput benchmarks.

- Model used: `gpt2`
- Prompts evaluated: `10`
- Layers evaluated: `[0, 3, 6, 9]`

## Focus slice

The table below highlights `sketch_dim=32` and `theta=0.5`.

| layer | topk_overlap_mean | false_elimination_rate_mean | elimination_rate_mean |
| --- | --- | --- | --- |
| 0 | 0.804167 | 0.275 | 0.347407 |
| 3 | 0.772917 | 0.622917 | 0.786142 |
| 6 | 0.747917 | 0.323958 | 0.541508 |
| 9 | 0.729167 | 0.179167 | 0.294509 |

## Notes

- Real transformer tensors need not behave like Gaussian synthetic tensors.
- Global similarity structure is often easier to preserve than exact top-attention ranking.
- False elimination remains the main failure mode to monitor.
- Layer and head behavior vary, so aggregate metrics can hide fragile substructures.
