"""Stage 1: replay a retargeted demonstration on the simulated Panda and measure success.

Usage: python -m h2r.replay --synthetic [-n 10] [--noise 0.003] [--smooth] [--layout]
       python -m h2r.replay --demos data/processed/*.npz [-n 10] [--layout]
"""
import argparse

import numpy as np

from h2r import retarget, sim

DWELL = 12          # control steps given to the fingers when they close or open (0.6 s)
SETTLE = 40         # steps after the last target, for the bowl to come to rest
REACH_START = 80    # at most this many steps to bring the gripper to the first target


def replay(env, obs, plan, action_noise=0.0, rng=None):
    """Track the targets once each, holding on gripper changes. Returns success, states, actions.

    With `action_noise`, the executed arm actions are perturbed while the clean ones are recorded, so that the
    recorded data also shows how to come back to the trajectory (DART-style data for imitation).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    rot = sim.gripper_rotation(obs, plan.yaw)
    states, actions = [], []

    def step(pos, grip):
        nonlocal obs
        action = sim.action_towards(obs, pos, rot, grip)
        states.append(sim.state(obs))
        actions.append(action)
        executed = action.copy()
        executed[:6] = np.clip(executed[:6] + rng.normal(0, action_noise, 6), -1, 1) if action_noise else executed[:6]
        obs, *_ = env.step(executed)

    for _ in range(REACH_START):  # the robot does not start where the hand did
        if np.linalg.norm(plan.targets[0] - obs["robot0_eef_pos"]) < 0.01:
            break
        step(plan.targets[0], -1.0)
    previous = plan.gripper[0]
    for pos, grip in zip(plan.targets, plan.gripper):
        for _ in range(DWELL if grip != previous else 1):
            step(pos, grip)
        previous = grip
    for _ in range(SETTLE):
        step(plan.targets[-1], -1.0)
    return bool(env.check_success()), np.array(states), np.array(actions)


if __name__ == "__main__":
    from h2r.synthetic import synthetic_demo

    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--demos", nargs="+", help="processed recordings, data/processed/<take>.npz")
    parser.add_argument("-n", type=int, default=10, help="number of LIBERO initial states")
    parser.add_argument("--noise", type=float, default=0.0, help="hand tracking noise, metres (synthetic only)")
    parser.add_argument("--layout", action="store_true", help="place the bowl as in the demonstration")
    parser.add_argument("--smooth", action="store_true", help="Kalman-smooth the synthetic hand track first")
    parser.add_argument("--slowdown", type=float, default=1.0, help="replay this many times slower than the hand")
    parser.add_argument("--near-rim", action="store_true",
                        help="grasp the rim nearest the demonstrator, as the protocol asked, instead of estimating it")
    args = parser.parse_args()
    rim_side = (0.0, -1.0) if args.near_rim else None  # human frame: towards the demonstrator

    if args.demos:  # real recordings are already smoothed by h2r.extract
        demos = {path: retarget.load_demo(path) for path in args.demos}
    else:
        demo = synthetic_demo(noise=args.noise)
        if args.smooth:
            from h2r.kalman import rts_smooth

            demo.pinch = rts_smooth(demo.pinch, dt=demo.t[1] - demo.t[0], meas_std=max(args.noise, 1e-3))
        demos = {f"synthetic, noise {1000 * args.noise:.0f} mm": demo}

    env, init_states = sim.make_env()
    for name, demo in demos.items():
        successes = []
        for k in range(args.n):
            obs = sim.reset(env, init_states[k])
            if args.layout:
                obs = sim.reset(env, init_states[k], bowl_xy=retarget.sim_bowl_xy(
                    demo, obs[sim.PLATE + "_pos"], bowl_radius=retarget.REAL_BOWL_RADIUS,
                    plate_radius=retarget.REAL_PLATE_RADIUS))
            plan = retarget.retarget(demo, obs[sim.BOWL + "_pos"], obs[sim.PLATE + "_pos"], slowdown=args.slowdown,
                                     rim_side=rim_side, start=obs["robot0_eef_pos"])
            ok, states, _ = replay(env, obs, plan)
            successes.append(ok)
        print(f"stage 1 replay, {name}: {sum(successes)}/{len(successes)}", flush=True)
