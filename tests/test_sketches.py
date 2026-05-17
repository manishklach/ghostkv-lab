import numpy as np

from ghostkv.sketches import (
    cosine_rank_correlation,
    project_keys,
    project_query,
    random_projection_matrix,
    sketch_similarity,
)


def test_projection_shapes() -> None:
    keys = np.ones((10, 8))
    query = np.ones(8)
    projection = random_projection_matrix(8, 4, seed=1)

    key_sketches = project_keys(keys, projection)
    query_sketch = project_query(query, projection)

    assert projection.shape == (8, 4)
    assert key_sketches.shape == (10, 4)
    assert query_sketch.shape == (4,)


def test_similarity_and_rank_correlation_are_bounded() -> None:
    projection = random_projection_matrix(6, 3, seed=2)
    keys = np.arange(24, dtype=float).reshape(4, 6)
    query = np.linspace(0.0, 1.0, 6)

    scores = sketch_similarity(project_query(query, projection), project_keys(keys, projection))
    corr = cosine_rank_correlation(np.array([4.0, 3.0, 2.0, 1.0]), scores)

    assert scores.shape == (4,)
    assert -1.0 <= corr <= 1.0

