import numpy as np

from ghostkv.lifecycle import ARCHIVE, GHOST, HOT, WARM, ghostify_tokens, transition_states
from ghostkv.sketches import random_projection_matrix


def test_transition_states_assign_all_tiers() -> None:
    ages = np.array([0, 1, 2, 3, 4, 5, 6])
    states = transition_states(ages, hot_window=2, warm_window=2)

    assert list(states[:2]) == [HOT, HOT]
    assert list(states[2:4]) == [WARM, WARM]
    assert list(states[4:6]) == [GHOST, GHOST]
    assert states[6] == ARCHIVE


def test_ghostify_tokens_builds_records() -> None:
    keys = np.ones((3, 4))
    projection = random_projection_matrix(4, 2, seed=3)
    records = ghostify_tokens(keys, projection, anchor_ids=[10, 11, 12], epsilon=0.1, sigma=0.2)

    assert len(records) == 3
    assert records[0].anchor_id == 10
    assert records[0].sketch.shape == (2,)
    assert records[0].epsilon_res == 0.1
    assert records[0].sigma_anchor == 0.2

