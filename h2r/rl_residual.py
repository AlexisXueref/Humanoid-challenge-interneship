"""Stage 3: a bounded residual on top of the imitation policy, learned with PPO.

The residual adds at most RESIDUAL_SCALE to the arm part of the imitation policy's action; the gripper stays the
imitation policy's. Training runs in a light copy of the scene (table, bowl and plate only, same goal), with LIBERO's
placement regions (h2r/assets/bowl_on_plate_light.bddl) or with the bowl's region widened to 7 x 20 cm
(bowl_on_plate_wide.bddl). Evaluation runs in LIBERO's own scene and initial states, the bowl optionally shifted.

The reward uses the demonstrations a third time, retargeted to the current layout: the bowl is rewarded for progressing
along the path the demonstrator's bowl took, while within PATH_TOLERANCE of it; LIBERO's success comes on top.
A first version also rewarded closing in on the demonstrated grasp point and penalised the bowl's distance to the
path at every step, with a discount of 0.99: in 200,000 steps the training success rate fell from about 0.45 to 0
(2026-10-08). Probable causes: the per-step penalty made carrying the bowl costly in itself, and with success around
step 200 the discount left it worth 0.99^200 = 0.13 of its value seen from the start. Hence the present reward, a
discount of 0.995, 1024 steps per environment between updates instead of 256, and a learning rate of 1e-4, not 3e-4.

Usage: python -m h2r.rl_residual train --base results/bc_x.pt [--demos data/processed/x.npz] [--wide] --steps 200000
       python -m h2r.rl_residual eval  --base results/bc_x.pt [--residual results/residual_seed0.zip] [--shifts]
"""
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from h2r import bc, retarget, sim

ROOT = Path(__file__).resolve().parent.parent
LIGHT_BDDL = Path(__file__).resolve().parent / "assets" / "bowl_on_plate_light.bddl"
WIDE_BDDL = Path(__file__).resolve().parent / "assets" / "bowl_on_plate_wide.bddl"
RESIDUAL_SCALE = 0.3   # at most 30 % of the controller's range added to the arm action
EXECUTE = 5            # actions of the imitation policy's chunk played before it predicts again
SUCCESS_REWARD = 10.0
PROGRESS_WEIGHT = 5.0
PATH_TOLERANCE = 0.05  # metres; a bowl knocked away from the path earns no progress
# Bowl shifts for evaluation, metres in the robot frame; imitation from one demo was measured on them on 2026-10-08
SHIFTS = [(0, 0), (0, 0.05), (0, -0.05), (0, 0.10), (0, -0.10), (-0.05, 0)]


def load_policy(path):
    ck = torch.load(path, weights_only=False)
    policy = bc.Policy(ck["n_in"], ck["chunk"])
    policy.load_state_dict(ck["state_dict"])
    return policy.eval()


def bowl_path(demo, obs, slowdown=2.0):
    """The demonstrator's bowl path during the carry, retargeted to this layout (bowl base positions)."""
    plan = retarget.retarget(demo, obs[sim.BOWL + "_pos"], obs[sim.PLATE + "_pos"], slowdown=slowdown,
                             start=obs["robot0_eef_pos"])
    offset = plan.targets[plan.grasp] - obs[sim.BOWL + "_pos"]
    return plan.targets[plan.grasp:plan.release + 1] - offset


class ResidualEnv(gym.Env):
    """Observation: low-dimensional state and the imitation policy's arm action. Action: the residual."""

    def __init__(self, base_path, demo_paths, bddl=LIGHT_BDDL, seed=0):
        """bddl: a light scene, placements drawn from its regions at each reset. bddl=None: LIBERO's scene; resets
        draw from `self.init_states` (all 50 by default), the bowl moved by `self.shift` (metres) if set."""
        from libero.libero.envs import OffScreenRenderEnv

        self.light = bddl is not None
        if self.light:
            self.env = OffScreenRenderEnv(bddl_file_name=str(bddl), has_offscreen_renderer=False, use_camera_obs=False)
            self.init_states = None
        else:
            self.env, self.init_states = sim.make_env()
        self.shift = None
        self.base = load_policy(base_path)
        self.demos = [retarget.load_demo(p) for p in demo_paths]
        self.rng = np.random.default_rng(seed)
        n = len(sim.state(self._fresh_obs()))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n + 6,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)

    def _fresh_obs(self):
        if self.light:
            obs = self.env.reset()
            for _ in range(sim.SETTLE_STEPS):
                obs, *_ = self.env.step(np.zeros(7))
            return obs
        init = self.init_states[self.rng.integers(len(self.init_states))]
        obs = sim.reset(self.env, init)
        if self.shift is not None:
            obs = sim.reset(self.env, init, bowl_xy=obs[sim.BOWL + "_pos"][:2] + self.shift)
        return obs

    def _observe(self):
        if not self.queue:
            self.queue = list(self.base.act(sim.state(self.obs))[:EXECUTE])
        self.base_action = self.queue[0]
        return np.r_[sim.state(self.obs), self.base_action[:6]].astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.obs = self._fresh_obs()
        self.path = bowl_path(self.demos[self.rng.integers(len(self.demos))], self.obs)
        self.progress, self.steps, self.queue = 0, 0, []
        return self._observe(), {}

    def step(self, residual):
        action = self.base_action.copy()
        action[:6] = np.clip(action[:6] + RESIDUAL_SCALE * np.asarray(residual), -1, 1)
        self.queue.pop(0)
        self.obs, *_ = self.env.step(action)
        self.steps += 1
        success = bool(self.env.check_success())
        bowl = self.obs[sim.BOWL + "_pos"]
        d = np.linalg.norm(self.path - bowl, axis=1)
        nearest = int(np.argmin(d))
        reward = SUCCESS_REWARD * success
        if d[nearest] < PATH_TOLERANCE:
            reward += PROGRESS_WEIGHT * max(nearest - self.progress, 0) / len(self.path)
            self.progress = max(self.progress, nearest)
        truncated = self.steps >= bc.MAX_STEPS
        return self._observe(), float(reward), success, truncated, {"is_success": success}


def reward_demos(args):
    return args.demos or sorted(str(p) for p in (ROOT / "data" / "processed").glob("*.npz"))


def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv

    demos, bddl = reward_demos(args), WIDE_BDDL if args.wide else LIGHT_BDDL
    # Monitor: episode return and success rate in PPO's log
    make = lambda i: (lambda: Monitor(ResidualEnv(args.base, demos, bddl=bddl, seed=args.seed * 100 + i)))  # noqa: E731
    venv = SubprocVecEnv([make(i) for i in range(args.envs)])
    model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=256, learning_rate=1e-4, gamma=0.995, seed=args.seed,
                policy_kwargs=dict(net_arch=[64, 64], log_std_init=-1.0), verbose=1)
    model.learn(total_timesteps=args.steps)
    out = ROOT / "results" / (args.out or f"residual_seed{args.seed}")
    model.save(out)
    print("saved", out.with_suffix(".zip"))


def evaluate(args):
    from stable_baselines3 import PPO

    env = ResidualEnv(args.base, reward_demos(args), bddl=None)
    held_out = env.init_states[40:50]
    model = PPO.load(args.residual) if args.residual else None
    label = "imitation + residual" if model else "imitation alone"
    total = []
    for shift in SHIFTS if args.shifts else [None]:
        env.shift = None if shift is None else np.array(shift, float)
        successes = []
        for state in held_out:
            env.init_states = [state]
            obs, _ = env.reset()
            done = False
            while not done:
                residual = model.predict(obs, deterministic=True)[0] if model else np.zeros(6)
                obs, _, success, truncated, _ = env.step(residual)
                done = success or truncated
            successes.append(success)
        total += successes
        where = "" if shift is None else f", bowl shifted ({100 * shift[0]:+.0f}, {100 * shift[1]:+.0f}) cm"
        print(f"stage 3 evaluation, {label}, held-out states 40-49{where}: {sum(successes)}/{len(successes)}",
              flush=True)
    if args.shifts:
        print(f"stage 3 evaluation, {label}, all shifts: {sum(total)}/{len(total)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--base", required=True)
    parser.add_argument("--demos", nargs="+", help="demonstrations for the reward (default: all processed takes)")
    parser.add_argument("--residual")
    parser.add_argument("--wide", action="store_true", help="train with the bowl's placement region widened")
    parser.add_argument("--shifts", action="store_true", help="evaluate with the bowl shifted by each of SHIFTS")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="residual name in results/ (default residual_seed<seed>)")
    args = parser.parse_args()
    train(args) if args.mode == "train" else evaluate(args)
