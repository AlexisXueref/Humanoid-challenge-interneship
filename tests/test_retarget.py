import numpy as np
import pytest

from h2r import retarget
from h2r.retarget import HUMAN_TO_SIM, Demo, grasp_events, grasp_yaw


def test_human_axes_map_to_robot_axes():
    np.testing.assert_allclose(HUMAN_TO_SIM @ [0, 1, 0], [1, 0, 0])   # forward along the tape -> robot forward (+x)
    np.testing.assert_allclose(HUMAN_TO_SIM @ [1, 0, 0], [0, -1, 0])  # to the demonstrator's right -> robot right (-y)


def test_grasp_events_close_then_reopen_with_hysteresis():
    rng = np.random.default_rng(0)
    aperture = np.r_[np.full(20, 0.09), np.linspace(0.09, 0.025, 6), np.full(30, 0.025),
                     np.linspace(0.025, 0.09, 6), np.full(20, 0.09)] + rng.normal(0, 0.002, 82)
    grasp, release = grasp_events(aperture)
    assert 21 <= grasp <= 25        # during the closing ramp (indices 20-25)
    assert 57 <= release <= 61      # during the opening ramp (indices 56-61)


def test_grasp_events_reject_a_pinch_that_never_reopens():
    with pytest.raises(ValueError):
        grasp_events(np.r_[np.full(20, 0.09), np.full(20, 0.02)])


@pytest.mark.parametrize("radial, expected_deg", [((-1, 0), 90), ((0, -1), 0), ((0, 1), 0), ((1, 0), 90)])
def test_grasp_yaw_puts_the_closing_axis_across_the_rim(radial, expected_deg):
    assert np.degrees(grasp_yaw(np.array(radial, float))) == pytest.approx(expected_deg)


def _demo():
    """Straight-line toy demonstration: approach, close, carry 23 cm forward, open, retreat."""
    t = np.arange(0, 6, 1 / 30)
    bowl, plate = np.array([0.0, 0.15, 0.0]), np.array([0.0, 0.38, 0.0])
    grasp_point = bowl + [0, -0.068, 0.065]
    pinch = np.zeros((t.size, 3))
    approach, carry = t < 2, (t >= 2) & (t < 4)
    pinch[approach] = grasp_point + np.outer(2 - t[approach], [0.05, -0.02, 0.03])
    s = (t[carry] - 2) / 2  # carry progress, 0 to 1: forward 23 cm in an arc up to 10 cm high
    pinch[carry] = grasp_point + np.c_[0 * s, 0.23 * s, 0.10 * np.sin(np.pi * s)]
    pinch[t >= 4] = grasp_point + [0, 0.23, 0] + np.outer(t[t >= 4] - 4, [0.03, 0, 0.04])
    aperture = np.where(carry, 0.025, 0.09)
    return Demo(t=t, pinch=pinch, aperture=aperture, bowl=bowl, plate=plate)


def test_sim_layout_copies_the_edge_gap_not_the_centre_distance():
    demo = _demo()  # real bowl 23 cm in front of the plate centre: 2.5 cm edge gap with radii 8 and 12.5 cm
    xy = retarget.sim_bowl_xy(demo, sim_plate=np.array([0.05, -0.02, 0.9]), bowl_radius=0.08, plate_radius=0.125)
    expected = 0.025 + retarget.sim.BOWL_RADIUS + retarget.sim.PLATE_RADIUS
    np.testing.assert_allclose(xy, [0.05 - expected, -0.02])  # towards the robot, on the same axis


def test_retarget_keeps_the_approach_shape_around_the_grasp_point():
    demo = _demo()
    out = retarget.retarget(demo, sim_bowl=np.array([-0.09, 0.0, 0.9]), sim_plate=np.array([0.05, -0.02, 0.9]))
    _, pinch, _ = retarget.resample(demo, retarget.HZ)
    pinch = pinch[len(pinch) - len(out.targets):]  # the toy hand sets off at once: nothing trimmed but its 1st step
    g = out.grasp
    np.testing.assert_allclose(out.targets[:g] - out.targets[g], (pinch[:g] - pinch[g]) @ HUMAN_TO_SIM.T, atol=1e-9)


def test_retarget_sets_off_from_the_gripper_and_joins_the_hand_path_at_the_grasp():
    kwargs = dict(sim_bowl=np.array([-0.09, 0.0, 0.9]), sim_plate=np.array([0.05, -0.02, 0.9]))
    free = retarget.retarget(_demo(), **kwargs)
    start = np.array([-0.2, 0.0, 1.17])  # LIBERO's resting gripper
    bent = retarget.retarget(_demo(), start=start, **kwargs)
    np.testing.assert_allclose(bent.targets[0], start)
    np.testing.assert_allclose(bent.targets[bent.grasp:], free.targets[free.grasp:])


def test_retarget_starts_when_the_hand_sets_off_and_never_goes_into_the_table():
    demo = _demo()
    rest = 30  # one second with the hand lying still on the table, as in the real recordings
    demo.t = np.r_[np.arange(rest) / 30, demo.t + rest / 30]
    demo.pinch = np.r_[np.repeat(demo.pinch[:1] - [0, 0, 0.10], rest, axis=0), demo.pinch]
    demo.aperture = np.r_[np.full(rest, 0.09), demo.aperture]
    sim_bowl = np.array([-0.09, 0.0, 0.9])
    out = retarget.retarget(demo, sim_bowl=sim_bowl, sim_plate=np.array([0.05, -0.02, 0.9]))
    assert out.targets[:, 2].min() >= sim_bowl[2] + retarget.MIN_CLEARANCE - 1e-12
    assert np.linalg.norm(out.targets[1] - out.targets[0]) > 0  # no still segment left at the start


def test_retarget_grasps_the_near_rim_and_releases_over_the_plate_centre():
    sim_bowl, sim_plate = np.array([-0.09, 0.0, 0.9]), np.array([0.05, -0.02, 0.9])
    out = retarget.retarget(_demo(), sim_bowl=sim_bowl, sim_plate=sim_plate)
    grasp, release = out.targets[out.grasp], out.targets[out.release]
    # Near rim = towards the robot (-x), at the rim height minus the grasp depth
    np.testing.assert_allclose(grasp[:2], sim_bowl[:2] + [-(retarget.sim.BOWL_RADIUS - retarget.GRASP_INSET), 0])
    # The gripper keeps its offset to the bowl, so the bowl centre ends right over the plate centre
    np.testing.assert_allclose(release[:2] - (grasp[:2] - sim_bowl[:2]), sim_plate[:2], atol=1e-9)
    assert out.yaw == pytest.approx(np.pi / 2)
    assert out.gripper[out.grasp] == 1 and out.gripper[out.release] == -1 and out.gripper[0] == -1
