import numpy as np

from h2r.kalman import rts_smooth


def test_a_noiseless_constant_velocity_track_is_left_untouched():
    t = np.arange(0, 2, 1 / 30)
    track = np.outer(t, [0.1, -0.2, 0.05]) + [0.3, 0.1, 0.0]
    # within 0.1 mm: the zero initial-velocity prior costs a few hundredths of a millimetre at the start
    np.testing.assert_allclose(rts_smooth(track, dt=1 / 30), track, atol=1e-4)


def test_a_gap_in_the_track_is_bridged():
    t = np.arange(0, 2, 1 / 30)
    track = np.outer(t, [0.1, -0.2, 0.05])
    holed = track.copy()
    holed[20:35] = np.nan  # half a second without the hand
    np.testing.assert_allclose(rts_smooth(holed, dt=1 / 30), track, atol=1e-3)


def test_smoothing_divides_centimetre_tracking_noise():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 60)[:, None]
    truth = 0.25 * (10 * t**3 - 15 * t**4 + 6 * t**5) * [1, 0.5, 0.2]  # a 25 cm minimum-jerk reach
    noisy = truth + rng.normal(0, 0.01, truth.shape)
    rms = lambda x: np.sqrt(np.mean(np.sum((x - truth) ** 2, axis=1)))  # noqa: E731
    assert rms(rts_smooth(noisy, dt=1 / 60, meas_std=0.01)) < 0.4 * rms(noisy)
