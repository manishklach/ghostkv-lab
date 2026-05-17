# Algorithm Notes

## Attention Upper Bound

For ghost record `G_i`, define:

- `s_i = sketch_sim(Q, G_i.sketch)`
- `epsilon_res_i` as a residual uncertainty term
- `sigma_anchor_i` as an anchor uncertainty term

Then the conservative bound is:

`AttnUB(Q, G_i) = s_i + epsilon_res_i + sigma_anchor_i`

Eliminate `G_i` if:

`AttnUB(Q, G_i) < theta_elim`

## Decode Pseudocode

```text
input: query Q, hot tokens H, ghost records G, threshold theta

Q_s = project_query(Q)

for each G_i in G:
    ub_i = sketch_sim(Q_s, G_i.sketch) + epsilon_res_i + sigma_anchor_i

survivors = {G_i | ub_i >= theta}
eliminated = G \ survivors

resurrect(survivors)
exact_set = H union survivors

run exact attention over exact_set
```

## Complexity

- Standard attention working-set scan: `O(N * d)`
- Quantized KV model: `O(N * d / q)`
- GhostKV simulator: `O(H * d + R * d + N_ghost * d_sketch)`

Definitions:

- `H` = number of hot tokens
- `R` = number of resurrected tokens
- `N_ghost` = number of ghost records inspected at query time
- `d` = full KV dimensionality
- `d_sketch` = sketch dimensionality

The core hypothesis is that `d_sketch << d`, allowing the elimination stage to touch a smaller representation than the full KV state.

## Notes On Exactness

This repository intentionally confines approximation to elimination. For tokens that survive elimination, the simulator uses exact scores to model final attention participation. This avoids overstating quality retention by conflating sketch scoring with final attention computation.

## Notes On Conservatism

The upper bound is only as credible as the uncertainty terms:

- if too small, false eliminations increase
- if too large, elimination becomes weak

This tension is part of the intended experimental study.

