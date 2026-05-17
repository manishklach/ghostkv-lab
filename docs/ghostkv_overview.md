# GhostKV Overview

## Problem

Autoregressive long-context inference can become limited by KV-cache movement: reading, transferring, and staging a growing set of keys and values at every decode step. As context length increases, the cost of touching a large working set may dominate even when the per-step arithmetic is modest.

## KV Cache Memory Bottleneck

Many runtime discussions focus on total KV storage footprint, but movement can matter just as much:

- bytes fetched into on-chip memory
- bytes moved across memory hierarchies
- bytes resurrected from colder tiers

If decode repeatedly touches a large context, the question is not only how many bytes exist, but how many bytes must move on the critical path.

## Ghost Records

GhostKV proposes converting cold KV entries into compact witness records called ghost records. A ghost record stores:

1. an attention sketch vector
2. a semantic anchor id
3. a residual uncertainty value

The sketch approximates a coarse attention signal. The anchor and residual terms capture additional uncertainty so that elimination can be conservative rather than aggressively approximate.

## Bounded Elimination

At query time, GhostKV computes an upper bound:

`AttnUB(Q, G_i) = sketch_sim(Q, G_i.sketch) + epsilon_res_i + sigma_anchor_i`

If the bound for a ghost record is below an elimination threshold, the token can be removed from consideration without reconstructing full KV for that token in the simulator. Tokens whose bounds remain above the threshold survive and are resurrected.

## Resurrection

Resurrection is the step where surviving ghost candidates are promoted back into the exact attention set. In this repository, resurrection is simulated rather than implemented against a real storage hierarchy. The purpose is to measure how often candidates survive and how that affects modeled bandwidth.

## Why Movement Matters More Than Storage

Storage reduction by itself does not guarantee decode acceleration. A method can save bytes at rest but still incur large movement costs if too many candidates must be reconstructed. This repository therefore emphasizes:

- elimination rate
- resurrection rate
- false elimination rate
- top-k overlap
- estimated bandwidth reduction

The goal is to study whether a compact witness mechanism can reduce the amount of KV state that must be moved while keeping attention quality degradation acceptably low.

