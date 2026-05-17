# Roadmap

## Phase 1: Synthetic Sketch Quality

- Implement projection-based sketches
- Audit overlap between exact and sketch-space rankings
- Sweep thresholds and sketch dimensions

## Phase 2: Attention Tensor Capture From A Small HuggingFace Model

- Capture real attention tensors from a small decoder-only model
- Replace synthetic anchor assumptions with measured statistics
- Study token-score distributions on real prompts

## Phase 3: LongBench / Needle Validation

- Run bounded-elimination studies on longer prompts
- Measure retrieval sensitivity and top-k preservation
- Compare failure modes across task types

## Phase 4: Latency Simulator

- Add explicit tiered-memory timing assumptions
- Model resurrection delay and overlap opportunities
- Estimate when elimination is net beneficial

## Phase 5: GPU Kernel / FlashAttention Integration

- Prototype an interface compatible with exact attention backends
- Explore FlashAttention-compatible survivor paths
- Evaluate CXL or near-memory execution models

