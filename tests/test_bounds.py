import numpy as np

from ghostkv.bounds import (
    compute_attention_upper_bound,
    eliminate_by_threshold,
    false_elimination_rate,
    topk_overlap,
)


def test_bound_shape_and_threshold() -> None:
    scores = np.array([0.1, 0.3, 0.5])
    bounds = compute_attention_upper_bound(scores, 0.05, np.array([0.01, 0.02, 0.03]))

    assert bounds.shape == scores.shape
    np.testing.assert_allclose(bounds, np.array([0.16, 0.37, 0.58]))
    np.testing.assert_array_equal(eliminate_by_threshold(bounds, 0.4), np.array([True, True, False]))


def test_false_elimination_and_topk_overlap() -> None:
    exact = np.array([0.9, 0.7, 0.2, 0.1])
    eliminated = np.array([True, False, True, False])

    fer = false_elimination_rate(exact, eliminated, true_threshold=0.6)
    overlap = topk_overlap(exact, np.array([0.1, 0.8, 0.3, 0.2]), k=2)

    assert fer == 0.5
    assert 0.0 <= overlap <= 1.0

