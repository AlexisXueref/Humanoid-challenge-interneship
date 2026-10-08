"""Stage 2: a small policy learned by imitation of the replayed demonstrations.

It maps the low-dimensional state (sim.state) to a chunk of the next controller actions, executes the first few,
then predicts again. Chunks because a step-by-step policy did not even reproduce its own training data on synthetic
demos (1/10 on training states, 2026-10-07).
Training data comes from replaying the retargeted demonstrations with noise on the executed actions
(see replay.replay), keeping successful replays only.

Usage: python -m h2r.bc --demos data/processed/a767851eec.npz --slowdown 2 --repeat 2 [--wide] [--seed 0] [--out x.pt]
       python -m h2r.bc --synthetic [--train 40] [--test 10] [--chunk 10] [--execute 5] [--seed 0]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from h2r import replay, retarget, sim

MAX_STEPS = 300  # LeRobot's episode limit for libero_goal (TASK_SUITE_MAX_STEPS), the one SmolVLA was scored with
STD_FLOOR = 0.05
WIDE_BOWL_REGION = ((-0.15, -0.10), (-0.08, 0.10))  # bowl x, y minima then maxima, as in assets/bowl_on_plate_wide.bddl


class Policy(nn.Module):
    def __init__(self, n_in, chunk, hidden=256):
        super().__init__()
        self.chunk = chunk
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 7 * chunk), nn.Tanh())
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)

    def act(self, state):
        """The next `chunk` actions, gripper commands snapped to open or closed."""
        with torch.no_grad():
            a = self(torch.as_tensor(state, dtype=torch.float32)[None])[0].numpy().astype(float)
        a = a.reshape(self.chunk, 7)
        a[:, 6] = np.where(a[:, 6] > 0, 1.0, -1.0)
        return a


def collect(env, init_states, demos, action_noise=0.1, seed=0, slowdown=1.0, rim_side=None, wide=False):
    """Replay each demonstration on the given initial states; keep the episodes of successful replays.
    `wide` draws the bowl's position in WIDE_BOWL_REGION instead of keeping LIBERO's."""
    rng = np.random.default_rng(seed)
    episodes = []
    for k, init in enumerate(init_states):
        obs = sim.reset(env, init, bowl_xy=rng.uniform(*WIDE_BOWL_REGION) if wide else None)
        plan = retarget.retarget(demos[k % len(demos)], obs[sim.BOWL + "_pos"], obs[sim.PLATE + "_pos"],
                                 slowdown=slowdown, start=obs["robot0_eef_pos"], rim_side=rim_side)
        ok, s, a = replay.replay(env, obs, plan, action_noise=action_noise, rng=rng)
        if ok:
            episodes.append((s, a))
    return episodes


def chunked(episodes, chunk):
    """Pair each state with the `chunk` actions that follow it in its episode (the last action repeated at the end)."""
    X, Y = [], []
    for s, a in episodes:
        padded = np.concatenate([a, np.repeat(a[-1:], chunk - 1, axis=0)])
        X.append(s)
        Y.append(np.stack([padded[t:t + chunk].ravel() for t in range(len(a))]))
    return np.concatenate(X), np.concatenate(Y)


def train(states, targets, chunk, epochs=150, batch=256, seed=0):
    torch.manual_seed(seed)
    X, Y = torch.as_tensor(states, dtype=torch.float32), torch.as_tensor(targets, dtype=torch.float32)
    policy = Policy(X.shape[1], chunk)
    policy.mean.copy_(X.mean(0))
    # Floor on the scale: some inputs barely move in the demonstrations (two quaternion components vary by
    # ~0.005), so a 0.03 deviation at test time would read as 6 standard deviations.
    policy.std.copy_(X.std(0).clamp(min=STD_FLOOR))
    optimiser = torch.optim.Adam(policy.parameters(), lr=1e-3)
    for _ in range(epochs):
        order = torch.randperm(len(X))
        for i in range(0, len(X), batch):
            idx = order[i:i + batch]
            loss = ((policy(X[idx]) - Y[idx]) ** 2).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    return policy, loss.item()


def evaluate(env, init_states, policy, execute):
    """Success per initial state, counted as in LIBERO's evaluation: success at any step ends the episode."""
    results = []
    for init in init_states:
        obs, steps, done = sim.reset(env, init), 0, False
        while steps < MAX_STEPS and not done:
            for action in policy.act(sim.state(obs))[:execute]:
                obs, *_ = env.step(action)
                steps += 1
                done = env.check_success()
                if done:
                    break
        results.append(bool(done))
    return results


if __name__ == "__main__":
    from h2r.kalman import rts_smooth
    from h2r.synthetic import synthetic_demo

    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--demos", nargs="+", help="processed recordings, data/processed/<take>.npz")
    parser.add_argument("--train", type=int, default=40, help="LIBERO initial states used for data collection")
    parser.add_argument("--test", type=int, default=10, help="held-out initial states for evaluation")
    parser.add_argument("--chunk", type=int, default=10, help="actions predicted at once")
    parser.add_argument("--execute", type=int, default=5, help="actions executed before predicting again")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slowdown", type=float, default=1.0)
    parser.add_argument("--repeat", type=int, default=1, help="passes over the training states, demos cycling")
    parser.add_argument("--near-rim", action="store_true", help="grasp the near rim in every demonstration")
    parser.add_argument("--out", help="checkpoint name in results/ (default from the options)")
    parser.add_argument("--wide", action="store_true", help="collect with the bowl drawn in WIDE_BOWL_REGION")
    args = parser.parse_args()

    env, init_states = sim.make_env()
    demos = [retarget.load_demo(path) for path in args.demos or []]
    if not args.demos:
        for k in range(5):  # five synthetic takes, with 10 mm tracking noise then smoothing, as real ones will be
            d = synthetic_demo(bowl_y=0.15, plate_y=(0.36, 0.38, 0.40)[k % 3], noise=0.01, seed=k)
            d.pinch = rts_smooth(d.pinch, dt=d.t[1] - d.t[0])
            demos.append(d)
    train_states, test_states = init_states[:args.train], init_states[args.train:args.train + args.test]
    episodes = collect(env, list(train_states) * args.repeat, demos, seed=args.seed, slowdown=args.slowdown,
                       rim_side=(0.0, -1.0) if args.near_rim else None, wide=args.wide)
    X, Y = chunked(episodes, args.chunk)
    print(f"collected {len(X)} samples from {len(episodes)}/{len(train_states) * args.repeat} successful noisy replays",
          flush=True)
    policy, loss = train(X, Y, args.chunk, seed=args.seed)
    print(f"final training loss {loss:.4f}", flush=True)
    out =Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    name = args.out or f"bc_{'real' if args.demos else 'synthetic'}{'_nearrim' if args.near_rim else ''}_seed{args.seed}.pt"
    torch.save({"state_dict": policy.state_dict(), "n_in": X.shape[1], "chunk": args.chunk, "X": X, "Y": Y},
               out / name)
    seen = evaluate(env, train_states[:args.test], policy, args.execute)
    print(f"stage 2 imitation (chunk {args.chunk}, execute {args.execute}), training states: {sum(seen)}/{len(seen)}",
          flush=True)
    results = evaluate(env, test_states, policy, args.execute)
    print(f"stage 2 imitation (chunk {args.chunk}, execute {args.execute}), held-out states: "
          f"{sum(results)}/{len(results)}")
