test:
	python -m pytest

results:
	python experiments/generate_results.py

real-validation:
	python experiments/real_attention_validation.py

hierarchical:
	python experiments/hierarchical_elimination.py

frontier:
	python experiments/false_elimination_frontier.py

all-results:
	python experiments/generate_results.py
	python experiments/real_attention_validation.py
	python experiments/hierarchical_elimination.py
	python experiments/false_elimination_frontier.py
	python experiments/generate_results.py

demo:
	python -m pytest
	python experiments/generate_results.py

clean-results:
	python -c "from pathlib import Path; root = Path('results'); names = ['sketch_quality.csv', 'elimination_tradeoff.csv', 'bandwidth_sweep.csv', 'sketch_dim_vs_topk_overlap.png', 'theta_vs_elimination_rate.png', 'resurrection_rate_vs_bandwidth.png']; [path.unlink() for path in [root / name for name in names] if path.exists()]"

clean-frontier:
	python -c "from pathlib import Path; root = Path('results/frontier'); [path.unlink() for path in root.glob('*') if path.is_file()]; root.rmdir() if root.exists() and not any(root.iterdir()) else None"
