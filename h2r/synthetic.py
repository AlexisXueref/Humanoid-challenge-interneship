"""A synthetic human demonstration, to test the whole chain before any real recording is plugged in.

It follows the recording protocol: right hand resting to the right of the bowl, pinch on the near rim, lift by
about 10 cm, bowl set down on the plate centre, hand withdrawn. Units are metres, in the human table frame.
"""
import numpy as np

from h2r.retarget import REAL_BOWL_RADIUS as BOWL_RADIUS, Demo

BOWL_HEIGHT, PLATE_THICKNESS = 0.085, 0.02  # the real objects, measured with a caliper
FPS = 30


def _min_jerk(a, b, n):
    s = np.linspace(0, 1, n)[:, None]
    return a + (b - a) * (10 * s**3 - 15 * s**4 + 6 * s**5)


def synthetic_demo(bowl_y=0.15, plate_y=0.38, noise=0.0, seed=0):
    bowl, plate = np.array([0.0, bowl_y, 0.0]), np.array([0.0, plate_y, 0.0])
    near_rim = np.array([0.0, -(BOWL_RADIUS - 0.012), BOWL_HEIGHT - 0.02])  # pinch point relative to the bowl base
    grasp = bowl + near_rim
    put = plate + near_rim + [0, 0, PLATE_THICKNESS + 0.005]
    waypoints = [  # (position, seconds to get there, pinch closed while moving there)
        (np.array([0.16, 0.08, 0.03]), 0.0, False),
        (grasp + [0, 0, 0.06], 1.2, False),
        (grasp, 0.6, False),
        (grasp, 0.4, True),            # fingers close
        (grasp + [0, 0, 0.10], 0.7, True),
        (put + [0, 0, 0.08], 1.2, True),
        (put, 0.6, True),
        (put, 0.3, False),             # fingers open
        (put + [0.05, -0.05, 0.10], 0.9, False),
    ]
    pinch, closed = [waypoints[0][0][None]], [np.array([False])]
    for (a, _, _), (b, duration, is_closed) in zip(waypoints[:-1], waypoints[1:]):
        n = max(int(duration * FPS), 2)
        pinch.append(_min_jerk(a, b, n)[1:])
        closed.append(np.full(n - 1, is_closed))
    pinch, closed = np.concatenate(pinch), np.concatenate(closed)
    aperture = np.where(closed, 0.025, 0.09)
    aperture = np.convolve(np.r_[np.full(4, aperture[0]), aperture, np.full(4, aperture[-1])], np.ones(9) / 9, "valid")
    rng = np.random.default_rng(seed)
    pinch = pinch + rng.normal(0, noise, pinch.shape)
    return Demo(t=np.arange(len(pinch)) / FPS, pinch=pinch, aperture=aperture, bowl=bowl, plate=plate)
