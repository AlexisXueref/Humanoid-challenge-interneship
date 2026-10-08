"""LIBERO's put_the_bowl_on_the_plate task, driven by end-effector targets.

The Panda runs LIBERO's operational-space controller in delta mode: one action moves the end-effector target by
at most 5 cm and 0.5 rad, at 20 Hz. The gripper points down and its fingers close along the world y axis.
Success is LIBERO's On(bowl, plate): contact, bowl above, and bowl centre within 3 cm of the plate centre.
"""
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

SUITE, TASK_ID = "libero_goal", 8
BOWL, PLATE = "akita_black_bowl_1", "plate_1"
SETTLE_STEPS = 10                 # LIBERO's evaluation lets objects fall onto the table first
MAX_DPOS, MAX_DROT = 0.05, 0.5    # controller output_max per unit action, metres and radians
POS_GAIN = ROT_GAIN = 2.0
# Measured on the simulated meshes: bowl rim 5.3 cm above its body origin (its base), outer radius 5.6 cm.
BOWL_RIM_HEIGHT, BOWL_RADIUS = 0.053, 0.056
PLATE_RADIUS, PLATE_HEIGHT = 0.0685, 0.017


def make_env(camera=False, size=128):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(TASK_ID)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    if camera:
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=size, camera_widths=size)
    else:
        env = OffScreenRenderEnv(bddl_file_name=bddl, has_offscreen_renderer=False, use_camera_obs=False)
    return env, suite.get_task_init_states(TASK_ID)


def reset(env, init_state, bowl_xy=None):
    """Reset to one of LIBERO's initial states, optionally moving the bowl, then let objects settle."""
    env.reset()
    obs = env.set_init_state(init_state)
    if bowl_xy is not None:
        joint = BOWL + "_joint0"
        qpos = env.sim.data.get_joint_qpos(joint).copy()
        qpos[:2] = bowl_xy
        env.sim.data.set_joint_qpos(joint, qpos)
        env.sim.forward()
    for _ in range(SETTLE_STEPS):
        obs, *_ = env.step(np.zeros(7))
    return obs


def gripper_rotation(obs, yaw):
    """Current gripper orientation turned by `yaw` radians about the world vertical."""
    return R.from_euler("z", yaw) * R.from_quat(obs["robot0_eef_quat"])  # robosuite quaternions are (x, y, z, w)


def action_towards(obs, pos, rot, gripper):
    """Proportional action that moves the end-effector towards (pos, rot); gripper -1 opens, +1 closes."""
    rot_err = (rot * R.from_quat(obs["robot0_eef_quat"]).inv()).as_rotvec()  # the controller applies it in world frame
    return np.r_[np.clip((pos - obs["robot0_eef_pos"]) / MAX_DPOS * POS_GAIN, -1, 1),
                 np.clip(rot_err / MAX_DROT * ROT_GAIN, -1, 1), gripper]


def state(obs):
    """Low-dimensional state for the policies: gripper pose and fingers, then objects relative to what they are
    approached from. The orientation is there so that a policy knows when the gripper has turned enough."""
    eef = obs["robot0_eef_pos"]
    return np.r_[eef, obs["robot0_eef_quat"], obs["robot0_gripper_qpos"],
                 obs[BOWL + "_pos"] - eef, obs[PLATE + "_pos"] - obs[BOWL + "_pos"]]
