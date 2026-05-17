# Limitations

This repository is intentionally narrow in scope and should be read as an experimental harness, not as validation of a production mechanism.

## Known Limitations

- Sketch collisions can make unrelated tokens appear similar in sketch space.
- Loose bounds may preserve too many candidates and reduce elimination.
- False eliminations can harm attention quality.
- Resurrection latency may dominate at batch size 1.
- Synthetic benchmarks are not proof of real-model benefit.
- Softmax denominator treatment still needs rigorous validation.

## Additional Caveats

- The current simulator models query-time selection behavior, not full end-to-end GPU execution.
- Semantic anchors are represented synthetically rather than learned from a real model.
- Archive behavior is heuristic and included for lifecycle modeling, not for validated runtime policy.
- Memory-bandwidth numbers are illustrative and intended for relative comparison.
- GPT-2 is not representative of all modern long-context LLMs.
- Small-model attention structure may differ materially from larger models.
- The real-attention validation path does not integrate with a decode kernel.
- No actual memory-movement reduction is measured on hardware.
- Resurrection remains simulated.
- FlashAttention integration has not been implemented yet.
