from pathlib import Path

from ghostkv.plotting import (
    plot_head_variance,
    plot_hierarchical_vs_flat,
    plot_layerwise_overlap,
    plot_resurrection_rate_vs_bandwidth,
    plot_real_attention_false_elimination,
    plot_real_attention_topk_overlap,
    plot_sketch_dim_vs_topk_overlap,
    plot_theta_vs_elimination_rate,
)


def test_plotting_functions_write_pngs(tmp_path: Path) -> None:
    sketch_rows = [
        {"sketch_dim": 8, "top32_overlap": 0.10},
        {"sketch_dim": 16, "top32_overlap": 0.20},
    ]
    elimination_rows = [
        {"sketch_dim": 8, "theta": 0.1, "elimination_rate": 0.4},
        {"sketch_dim": 8, "theta": 0.5, "elimination_rate": 0.5},
        {"sketch_dim": 16, "theta": 0.1, "elimination_rate": 0.45},
        {"sketch_dim": 16, "theta": 0.5, "elimination_rate": 0.55},
    ]
    bandwidth_rows = [
        {"scheme": "full_kv", "resurrection_rate": 0.0, "bytes": 1000.0},
        {"scheme": "ghostkv", "resurrection_rate": 0.01, "bytes": 700.0},
        {"scheme": "ghostkv", "resurrection_rate": 0.05, "bytes": 900.0},
    ]
    real_attention_rows = [
        {"layer_idx": 0, "sketch_dim": 8, "theta": 0.1, "topk_overlap_mean": 0.5, "false_elimination_rate_mean": 0.2},
        {"layer_idx": 0, "sketch_dim": 32, "theta": 0.5, "topk_overlap_mean": 0.7, "false_elimination_rate_mean": 0.1},
        {"layer_idx": 3, "sketch_dim": 8, "theta": 0.1, "topk_overlap_mean": 0.4, "false_elimination_rate_mean": 0.3},
        {"layer_idx": 3, "sketch_dim": 32, "theta": 0.5, "topk_overlap_mean": 0.6, "false_elimination_rate_mean": 0.15},
    ]
    head_variance_rows = [
        {"layer_idx": 0, "topk_overlap": 0.4},
        {"layer_idx": 0, "topk_overlap": 0.6},
        {"layer_idx": 3, "topk_overlap": 0.5},
        {"layer_idx": 3, "topk_overlap": 0.8},
    ]
    hierarchical_rows = [
        {"method": "flat", "theta": 0.1, "false_elimination_rate_mean": 0.25},
        {"method": "flat", "theta": 0.5, "false_elimination_rate_mean": 0.35},
        {"method": "hierarchical", "theta": 0.1, "false_elimination_rate_mean": 0.20},
        {"method": "hierarchical", "theta": 0.5, "false_elimination_rate_mean": 0.28},
    ]

    sketch_path = tmp_path / "sketch.png"
    theta_path = tmp_path / "theta.png"
    bandwidth_path = tmp_path / "bandwidth.png"
    real_topk_path = tmp_path / "real_topk.png"
    real_false_path = tmp_path / "real_false.png"
    layerwise_path = tmp_path / "layerwise.png"
    variance_path = tmp_path / "variance.png"
    hierarchical_path = tmp_path / "hierarchical.png"

    plot_sketch_dim_vs_topk_overlap(sketch_rows, sketch_path)
    plot_theta_vs_elimination_rate(elimination_rows, theta_path)
    plot_resurrection_rate_vs_bandwidth(bandwidth_rows, bandwidth_path)
    plot_real_attention_topk_overlap(real_attention_rows, real_topk_path)
    plot_real_attention_false_elimination(real_attention_rows, real_false_path)
    plot_layerwise_overlap(real_attention_rows[1::2], layerwise_path)
    plot_head_variance(head_variance_rows, variance_path)
    plot_hierarchical_vs_flat(hierarchical_rows, hierarchical_path)

    assert sketch_path.exists()
    assert theta_path.exists()
    assert bandwidth_path.exists()
    assert real_topk_path.exists()
    assert real_false_path.exists()
    assert layerwise_path.exists()
    assert variance_path.exists()
    assert hierarchical_path.exists()
