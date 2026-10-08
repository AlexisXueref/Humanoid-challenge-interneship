"""Object-centric retargeting of a human pick-and-place onto the simulated Panda.

The shape of the demonstration is kept and its anchors are moved. The approach is replayed relative to the grasp
point, which lands on the simulated bowl's rim on the side the hand took. The carry is bent progressively so that
the bowl ends over the plate centre: LIBERO only counts a bowl within 3 cm of it, and the real plate is 25 cm wide
against 13.7 cm in simulation, so where the hand put the bowl on the real plate cannot be copied as is.
"""
from dataclasses import dataclass

import numpy as np

from h2r import sim

HZ = 20  # control rate of the simulated arm
# Human table frame: x to the demonstrator's right, y along the tape away from them, z up.
# Robot frame: x forward, y to the robot's left, z up. Demonstrator and robot sit on the same side of the table.
HUMAN_TO_SIM = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
GRASP_INSET = 0.012     # fingers this far inside the rim, horizontally
GRASP_DEPTH = 0.02      # and this far below it
PLACE_CLEARANCE = 0.01  # release the bowl this high above the plate
REAL_BOWL_RADIUS, REAL_PLATE_RADIUS = 0.08, 0.125  # the demonstrator's bowl and plate, measured with a caliper
MIN_CLEARANCE = 0.03    # no target lower than this above the table: the hand rests on it, the gripper cannot
START_MOTION = 0.03     # the demonstration starts once the hand has moved this far from where it rested


@dataclass
class Demo:
    t: np.ndarray         # (N,) seconds
    pinch: np.ndarray     # (N, 3) midpoint between thumb and index tips, human table frame, metres
    aperture: np.ndarray  # (N,) thumb-index distance, metres
    bowl: np.ndarray      # (3,) centre of the bowl's base at the start
    plate: np.ndarray     # (3,) centre of the plate's base


@dataclass
class Retargeted:
    targets: np.ndarray   # (T, 3) end-effector positions at HZ, robot world frame
    yaw: float            # gripper rotation about the vertical, radians
    gripper: np.ndarray   # (T,) +1 closed, -1 open
    grasp: int
    release: int


def load_demo(path):
    """A processed recording, data/processed/<take>.npz (written by h2r.extract)."""
    z = np.load(path)
    return Demo(t=z["t"], pinch=z["pinch"], aperture=z["aperture"], bowl=z["bowl"], plate=z["plate"])


def grasp_events(aperture, low=0.35, high=0.65):
    """Indices where the pinch closes, then reopens, with hysteresis between two fractions of its range."""
    lo, hi = np.percentile(aperture, [5, 95])
    closed = np.flatnonzero(aperture < lo + low * (hi - lo))
    if closed.size == 0:
        raise ValueError("the pinch never closes")
    reopened = np.flatnonzero(aperture[closed[0]:] > lo + high * (hi - lo))
    if reopened.size == 0:
        raise ValueError("the pinch never reopens")
    return int(closed[0]), int(closed[0] + reopened[0])


def release_event(pinch, plate, grasp, plate_radius=REAL_PLATE_RADIUS):
    """Where the bowl is set down: the lowest pinch point after the grasp, among those over the real plate.

    The pinch aperture is no use here: once the bowl is down, the hand stays half closed (3 to 4 cm against
    2.5 cm while holding, take a767851eec), so it does not reopen past any useful threshold.
    """
    over = grasp + np.flatnonzero(np.linalg.norm(pinch[grasp:, :2] - plate[:2], axis=1) < plate_radius)
    if over.size == 0:
        raise ValueError("the hand never comes over the plate")
    return int(over[np.argmin(pinch[over, 2])])


def events(pinch, aperture, plate):
    """(grasp, release) indices: the pinch closing, then the bowl set down over the plate."""
    grasp, _ = grasp_events(aperture)
    return grasp, release_event(pinch, plate, grasp)


def resample(demo, hz):
    t = np.arange(demo.t[0], demo.t[-1], 1.0 / hz)
    pinch = np.stack([np.interp(t, demo.t, demo.pinch[:, k]) for k in range(3)], axis=1)
    return t, pinch, np.interp(t, demo.t, demo.aperture)


def grasp_yaw(radial_xy):
    """Yaw turning the fingers' closing axis (world y at rest) onto the rim's radial direction, in (-90, 90] degrees.

    The gripper is symmetric, so both signs work in principle; +90 degrees on the near rim succeeded 10/10 against
    9/10 for -90 in the scripted probe of 2026-10-07, hence the half-open interval.
    """
    yaw = np.mod(np.arctan2(radial_xy[1], radial_xy[0]) + np.pi / 2, np.pi)
    return yaw - np.pi if yaw > np.pi / 2 + 1e-9 else yaw


def sim_bowl_xy(demo, sim_plate, bowl_radius, plate_radius):
    """Where to put the simulated bowl so that it sits, relative to the plate, as the real one did.

    The objects differ in size, so the edge-to-edge gap is copied rather than the centre distance.
    `bowl_radius` and `plate_radius` are the real objects' radii.
    """
    v = (HUMAN_TO_SIM @ (demo.bowl - demo.plate))[:2]
    distance = np.linalg.norm(v)
    gap = distance - (bowl_radius + plate_radius)
    return sim_plate[:2] + v / distance * (gap + sim.BOWL_RADIUS + sim.PLATE_RADIUS)


def retarget(demo, sim_bowl, sim_plate, hz=HZ, slowdown=1.0, rim_side=None, start=None):
    """`slowdown` stretches the demonstration in time (a robot arm is slower than a hand). `rim_side`, a horizontal
    direction in the human frame, fixes which part of the rim is grasped instead of estimating it from the data.
    `start`, the gripper's position before the replay, bends the approach so that it sets off from there.
    """
    t, pinch, aperture = resample(demo, hz * slowdown)
    g, r = events(pinch, aperture, demo.plate)

    radial = (pinch[g] - demo.bowl)[:2] if rim_side is None else np.asarray(rim_side, float)
    radial = HUMAN_TO_SIM[:2, :2] @ (radial / np.linalg.norm(radial))
    grasp = sim_bowl + np.r_[radial * (sim.BOWL_RADIUS - GRASP_INSET), sim.BOWL_RIM_HEIGHT - GRASP_DEPTH]
    targets = grasp + (pinch - pinch[g]) @ HUMAN_TO_SIM.T

    # Bend the carry so the bowl centre ends over the plate centre; the retreat follows the same shift.
    release = sim_plate + [0, 0, sim.PLATE_HEIGHT + PLACE_CLEARANCE] + (grasp - sim_bowl)
    shift = release - targets[r]
    travelled = np.linalg.norm(pinch[g:r, :2] - pinch[g, :2], axis=1)
    progress = np.clip(travelled / max(np.linalg.norm(pinch[r, :2] - pinch[g, :2]), 1e-6), 0, 1)
    targets[g:r] += progress[:, None] * shift
    targets[r:] += shift

    # A resting hand lies on the table, where the gripper cannot go: keep a clearance, and start the replay when
    # the hand sets off (a policy learning from a still start has no clue when to leave it).
    targets[:, 2] = np.maximum(targets[:, 2], sim_bowl[2] + MIN_CLEARANCE)
    moved = np.flatnonzero(np.linalg.norm(pinch[:g] - pinch[0], axis=1) > START_MOTION)
    s = int(moved[0]) if moved.size else 0
    gripper = np.where((np.arange(t.size) >= g) & (np.arange(t.size) < r), 1.0, -1.0)
    targets, gripper, g, r = targets[s:], gripper[s:], g - s, r - s
    if start is not None:
        # Blend from the gripper's actual position into the hand's path, fully on it at the grasp. Without this the
        # arm first flies to where the hand started, then comes back: the same places visited with opposite
        # actions, which a policy learning from the states alone cannot tell apart.
        w = np.linspace(0, 1, g + 1)[:, None]
        w = w * w * (3 - 2 * w)  # smoothstep
        targets[:g + 1] += (1 - w) * (np.asarray(start) - targets[0])
    return Retargeted(targets=targets, yaw=grasp_yaw(radial), gripper=gripper, grasp=g, release=r)
