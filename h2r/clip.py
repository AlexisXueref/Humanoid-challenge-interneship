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
    parser.add_argument("--after", type=int, default=40, help="steps still filmed after success, to show the release")
    args = parser.parse_args()

    policy = rl_residual.load_policy(args.base)
    env, init_states = sim.make_env(camera=True, size=256)
    obs, steps, success, frames = sim.reset(env, init_states[args.state]), 0, None, []
    # LIBERO counts success at the first step the bowl rests on the plate, possibly still in the gripper: keep going
    while steps < (bc.MAX_STEPS if success is None else success + args.after):
        for a in policy.act(sim.state(obs))[:5]:
            obs, *_ = env.step(a)
            steps += 1
            if success is None and env.check_success():
                success = steps
            if steps % 2 == 0:  # every other control step: real time at 10 frames per second
                frames.append(Image.fromarray(obs["agentview_image"][::-1]))  # robosuite renders upside down
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=100, loop=0)
    print(f"{len(frames)} frames, success at step {success}, bowl still on the plate at the end: "
          f"{env.check_success()}, gripper fingers {obs['robot0_gripper_qpos'].round(3)}: {args.out}")
