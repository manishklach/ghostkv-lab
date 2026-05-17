from ghostkv.bandwidth import (
    bandwidth_reduction_ratio,
    ghostkv_bytes,
    quantized_kv_bytes,
    standard_kv_bytes,
)
from ghostkv.simulator import SyntheticGhostKVSimulator


def test_bandwidth_arithmetic() -> None:
    standard = standard_kv_bytes(num_tokens=100, dim=16)
    quantized = quantized_kv_bytes(num_tokens=100, dim=16, quant_factor=4)
    ghost = ghostkv_bytes(
        num_hot=10,
        num_resurrected=5,
        kv_bytes_per_token=standard_kv_bytes(1, 16),
        num_ghost=90,
        ghost_record_bytes=64,
    )

    assert standard == 6400.0
    assert quantized == 1600.0
    assert ghost > 0.0
    assert bandwidth_reduction_ratio(standard, ghost) < 1.0


def test_simulator_metric_keys() -> None:
    simulator = SyntheticGhostKVSimulator(
        num_tokens=256,
        dim=32,
        sketch_dim=8,
        hot_window=32,
        theta_elim=0.3,
        epsilon=0.05,
        sigma=0.05,
        seed=5,
    )
    metrics = simulator.run_decode_step()

    expected = {
        "elimination_rate",
        "resurrection_rate",
        "false_elimination_rate",
        "topk_overlap",
        "estimated_standard_bytes",
        "estimated_ghostkv_bytes",
        "estimated_resurrection_bytes",
        "bandwidth_reduction",
    }
    assert expected.issubset(metrics.keys())
