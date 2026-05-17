from pathlib import Path
import re

from experiments.generate_results import generate_all_results


def test_generate_all_results_writes_expected_artifacts(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_md = tmp_path / "RESULTS.md"

    artifacts = generate_all_results(
        results_dir=results_dir,
        results_md_path=results_md,
        sketch_num_tokens=256,
        tradeoff_num_tokens=512,
        bandwidth_context=4096,
        dim=32,
        hot_window=128,
        bandwidth_hot_window=256,
        tradeoff_steps=3,
        epsilon=0.05,
        sigma=0.05,
        seed=123,
        ghost_record_bytes=64.0,
        quant_factor=4.0,
    )

    expected_names = {
        "sketch_csv": "sketch_quality.csv",
        "elimination_csv": "elimination_tradeoff.csv",
        "bandwidth_csv": "bandwidth_sweep.csv",
        "sketch_plot": "sketch_dim_vs_topk_overlap.png",
        "theta_plot": "theta_vs_elimination_rate.png",
        "bandwidth_plot": "resurrection_rate_vs_bandwidth.png",
        "results_md": "RESULTS.md",
    }
    for key, name in expected_names.items():
        assert artifacts[key].name == name
        assert artifacts[key].exists()

    summary = results_md.read_text(encoding="utf-8")
    assert "These are synthetic simulation results, not real-model results." in summary
    assert "results/sketch_dim_vs_topk_overlap.png" in summary

    referenced_files = re.findall(r"\((results/[^)]+)\)", summary)
    for relative_path in referenced_files:
        assert (tmp_path / relative_path).exists()
