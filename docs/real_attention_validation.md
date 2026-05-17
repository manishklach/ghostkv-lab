# Real Attention Validation

## Goal

The real-attention validation path captures query and key tensors from lightweight HuggingFace transformer models and evaluates whether GhostKV-style sketches preserve attention ranking under real model distributions.

The current implementation is intentionally narrow:

- no throughput benchmarking
- no production runtime claims
- no memory-movement measurements from an integrated kernel

Instead, the focus is on attention-ranking preservation and bounded elimination behavior.

## Q/K Capture Method

For a prompt and selected layer:

1. tokenize the prompt
2. run the model in inference mode on CPU
3. collect the hidden state that feeds the selected attention layer
4. reconstruct query and key tensors from the layer's projection weights
5. evaluate last-token query attention against the full prompt key set

For GPT-2 style models, the code uses the `c_attn` projection and splits it into query, key, and value blocks.

For Llama-family models, the code uses `q_proj` and `k_proj`, then applies the model's rotary embedding path before measuring attention scores.

## Metrics

The real-attention experiments report:

- top-k overlap
- rank correlation
- false elimination rate
- elimination rate
- head-wise variance across layers

These metrics are reported per head and in aggregate.

## Why Synthetic Tensors Differ

Synthetic Gaussian tensors are useful for validating the mechanics of a harness, but they are not substitutes for transformer attention states.

Real transformer tensors:

- contain layer-specific structure
- can have head-specific specialization
- may be more anisotropic
- may have sharper or more unstable ranking behavior

As a result, good synthetic behavior does not imply good real-model behavior.

## Why Rank Preservation Matters

GhostKV-style bounded elimination depends on identifying tokens that can be safely discarded before reconstructing or touching full KV state.

If sketches preserve coarse geometry but fail on exact high-attention tokens, then false elimination becomes a central risk.

This is why the evaluation emphasizes top-k overlap and false elimination rather than only global correlation.

## Limitations

- GPT-2 is not representative of all modern LLMs.
- Small models differ from larger long-context models.
- The current repo does not integrate with a decode kernel.
- No actual memory-movement reduction is measured on hardware.
- Resurrection remains simulated.
- FlashAttention integration is still future work.
