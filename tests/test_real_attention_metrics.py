import torch

from ghostkv.hf_capture import extract_attention_statistics


def test_extract_attention_statistics_returns_per_head_and_aggregate() -> None:
    query_states = torch.randn(1, 3, 7, 8)
    key_states = torch.randn(1, 3, 7, 8)

    stats = extract_attention_statistics(
        query_states=query_states,
        key_states=key_states,
        sketch_dim=4,
        theta=0.3,
        epsilon=0.05,
        sigma=0.05,
        topk=4,
        seed=7,
    )

    assert len(stats["per_head_rows"]) == 3
    assert stats["aggregate"]["num_heads"] == 3
    assert 0.0 <= stats["aggregate"]["topk_overlap_mean"] <= 1.0
    assert 0.0 <= stats["aggregate"]["elimination_rate_mean"] <= 1.0
