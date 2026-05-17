# False Elimination Frontier

## Status

Real-attention frontier analysis generated for `gpt2`.

## Methodology

This experiment captures real query/key tensors from selected transformer layers, applies GhostKV-style sketch projections, sweeps `theta_elim`, and records the tradeoff between elimination and false elimination at layer/head granularity.

## Relevance definition

- Relevance mode: `topk`
- Top-k: `32`
- Percentile: `95.0`
- Informative prompt-layer slices analyzed: `24`
- Prompt-layer slices skipped because `seq_len <= topk`: `56`

By default, relevance is defined by exact top-k membership, not by the approximate sketch score.

## Key findings

- This analysis does not prove GhostKV correctness.
- The goal is to locate operating regions where elimination is meaningful while false elimination remains controlled.
- False elimination remains the primary technical risk.
- High rank correlation alone is not enough to establish acceptable elimination behavior.

## Safe-ish operating points

| layer | sketch_dim | theta_elim | elimination_rate_mean | false_elimination_rate_mean | top32_overlap_mean |
| --- | --- | --- | --- | --- | --- |
| _none found_ |  |  |  |  |  |

## Layer-wise observations

- Layer 9, sketch 64, theta 0.7: false elim 0.258681, elimination 0.388562
- Layer 9, sketch 64, theta 0.75: false elim 0.258681, elimination 0.388562
- Layer 9, sketch 64, theta 0.8: false elim 0.258681, elimination 0.388562
- Layer 9, sketch 64, theta 0.85: false elim 0.258681, elimination 0.388562
- Layer 9, sketch 64, theta 0.9: false elim 0.258681, elimination 0.388562
- Layer 9, sketch 64, theta 0.95: false elim 0.258681, elimination 0.388562

## Head-wise observations

- Layer 0 head 5 at sketch 32 and theta 0.25: false elim 0.0, elimination 0.03073
- Layer 0 head 5 at sketch 32 and theta 0.3: false elim 0.0, elimination 0.03073
- Layer 0 head 5 at sketch 32 and theta 0.2: false elim 0.0, elimination 0.027857
- Layer 0 head 5 at sketch 32 and theta 0.05: false elim 0.0, elimination 0.025293
- Layer 0 head 5 at sketch 32 and theta 0.1: false elim 0.0, elimination 0.025293
- Layer 0 head 5 at sketch 32 and theta 0.15: false elim 0.0, elimination 0.025293

## Limitations

- This frontier is measured on GPT-2 attention tensors, not on larger long-context models.
- The analysis does not measure runtime or memory movement directly.
- Resurrection is still simulated.
- Extreme-rank preservation is sensitive to prompt, layer, and head behavior.

## Next steps

- Expand the frontier analysis to additional decoder architectures.
- Compare top-k relevance with percentile-based relevance.
- Study whether hierarchical or learned sketches can lower false elimination without collapsing elimination rate.
- Use the frontier to decide whether deeper systems integration is warranted.
