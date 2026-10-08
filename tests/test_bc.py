import numpy as np

from h2r.bc import chunked


def test_chunked_pairs_each_state_with_the_next_actions_padding_with_the_last():
    states, actions = np.zeros((3, 2)), np.arange(3 * 7, dtype=float).reshape(3, 7)
    X, Y = chunked([(states, actions)], chunk=2)
    assert X.shape == (3, 2) and Y.shape == (3, 14)
    np.testing.assert_array_equal(Y[0], np.r_[actions[0], actions[1]])
    np.testing.assert_array_equal(Y[2], np.r_[actions[2], actions[2]])  # past the end: the last action repeated
