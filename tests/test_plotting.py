from pathlib import Path

from ghostkv.plotting import (
    plot_resurrection_rate_vs_bandwidth,
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

    sketch_path = tmp_path / "sketch.png"
    theta_path = tmp_path / "theta.png"
    bandwidth_path = tmp_path / "bandwidth.png"

    plot_sketch_dim_vs_topk_overlap(sketch_rows, sketch_path)
    plot_theta_vs_elimination_rate(elimination_rows, theta_path)
    plot_resurrection_rate_vs_bandwidth(bandwidth_rows, bandwidth_path)

    assert sketch_path.exists()
    assert theta_path.exists()
    assert bandwidth_path.exists()
