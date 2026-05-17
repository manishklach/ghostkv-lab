import numpy as np

from ghostkv.hierarchical import (
    coarse_elimination,
    hierarchical_token_elimination,
    simple_anchor_clustering,
    token_level_elimination,
)


def test_simple_anchor_clustering_shapes() -> None:
    keys = np.arange(60, dtype=float).reshape(10, 6)
    clustering = simple_anchor_clustering(keys, num_anchors=3, seed=5)

    assert clustering["assignments"].shape == (10,)
    assert clustering["centroids"].shape == (3, 6)
    assert clustering["radii"].shape == (3,)
    assert np.all(clustering["assignments"] >= 0)


def test_hierarchical_elimination_outputs_valid_masks() -> None:
    rng = np.random.default_rng(3)
    key_sketches = rng.normal(size=(12, 4))
    query_sketch = rng.normal(size=(4,))

    flat_mask = token_level_elimination(query_sketch, key_sketches, theta=0.2, epsilon=0.05, sigma=0.05)
    hierarchical = hierarchical_token_elimination(
        query_sketch=query_sketch,
        key_sketches=key_sketches,
        num_anchors=4,
        theta=0.2,
        epsilon=0.05,
        sigma=0.05,
        seed=9,
    )
    cluster_mask = coarse_elimination(
        query_sketch=query_sketch,
        centroids=hierarchical["centroids"],
        cluster_radii=np.zeros(hierarchical["centroids"].shape[0], dtype=float),
        theta=0.2,
        epsilon=0.05,
        sigma=0.05,
    )

    assert flat_mask.shape == (12,)
    assert hierarchical["eliminated_mask"].shape == (12,)
    assert cluster_mask.ndim == 1
