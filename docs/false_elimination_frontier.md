# False Elimination Frontier

## Why This Analysis Matters

False elimination is the central risk for GhostKV-style bounded filtering.

High rank correlation by itself is not enough. A sketch can preserve broad geometry while still missing the extreme-score tokens that matter most for exact attention behavior.

Transformer attention often depends on the extreme tail of QK scores, so false elimination must be measured directly.

## Core Tradeoff

The frontier experiment measures the relationship between:

- elimination rate
- false elimination rate
- top-k overlap
- sketch dimension
- elimination threshold
- layer and head behavior

The main question is not whether sketches look correlated on average. The main question is whether meaningful elimination can happen while false elimination remains controlled.

## Relevance Definitions

The experiment supports two relevance rules:

1. `topk`
   Relevant tokens are the exact top-k tokens under the real QK scores.
2. `percentile`
   Relevant tokens are those above an exact-score percentile threshold.

The default is `topk` with `k=32`.

## Why Top-k Attention Is Fragile

Random projections can preserve global geometry while still failing at extreme-rank preservation. Transformer attention often depends on the extreme tail of QK scores, so false elimination must be measured directly.

This is why the frontier analysis emphasizes top-k overlap and false elimination rather than only global correlation.

## Theta Sensitivity

`theta_elim` determines how aggressively GhostKV-style bounds eliminate candidates.

- lower `theta_elim` usually means less elimination and lower false elimination
- higher `theta_elim` usually means more elimination and greater risk

The frontier therefore helps identify whether any conservative operating region exists for a given model, layer, and sketch dimension.

## Safe-ish Operating Region

This repository uses the term `safe-ish` rather than `safe`.

A safe-ish operating point is a heuristic configuration where:

- false elimination rate is at or below a target bound
- elimination rate remains meaningfully above zero

This does not prove correctness. It is only a way to locate operating regions that may be worth further study before making any systems or latency claims.

## Why This Comes Before Latency Claims

If false elimination cannot be controlled, then runtime optimization work is premature. The frontier analysis is intended to answer a methodology question first:

Can GhostKV eliminate meaningful amounts of cold KV state while keeping false elimination acceptably low on real attention tensors?
