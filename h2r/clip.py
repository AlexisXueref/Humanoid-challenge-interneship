"""A short clip of an imitation policy in LIBERO's scene, from the agent-view camera.

Usage: python -m h2r.clip --base results/bc_a767851eec_seed0.pt --state 40 --out media/imitation_state40.gif
"""
import argparse

from PIL import Image

from h2r import bc, rl_residual, sim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--state", type=int, default=40, help="LIBERO initial state (40-49 are held out)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    policy = rl_residual.load_policy(args.base)
    env, init_states = sim.make_env(camera=True, size=256)
    obs, steps, done, frames = sim.reset(env, init_states[args.state]), 0, False, []
    while steps < bc.MAX_STEPS and not done:
        for a in policy.act(sim.state(obs))[:5]:
            obs, *_ = env.step(a)
            steps += 1
            done = env.check_success()
            if steps % 2 == 0 or done:  # every other control step: real time at 10 frames per second
                frames.append(Image.fromarray(obs["agentview_image"][::-1]))  # robosuite renders upside down
            if done:
                break
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=100, loop=0)
    print(f"{len(frames)} frames, success {done} after {steps} steps: {args.out}")
