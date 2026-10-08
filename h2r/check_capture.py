"""Check that recordings are usable before any extraction.

A demo passes when the hand is found on at least 90 % of the frames of the gesture (from the first to the last
frame it is seen: recordings start and end before the hand enters), the LiDAR depth under the wrist has the
highest confidence on at least 80 % of those frames, and the table plane fits the first depth map with an RMS
error under 1 cm.

Usage: python -m h2r.check_capture data/raw/<demo> [data/raw/<demo> ...]
"""
import sys

import numpy as np

from h2r import capture
from h2r.hand import WRIST, track_hand

HAND_MIN, CONFIDENCE_MIN, PLANE_RMS_MAX = 0.90, 0.80, 0.01


def check(root):
    width, height, n_frames, fps = capture.video_info(root)
    depths = capture.depth_paths(root)
    hands = track_hand(capture.rgb_frames(root, scale=0.5), fps)  # keypoints are normalised
    seen = [i for i, kp in enumerate(hands) if kp is not None]
    gesture = hands[seen[0]:seen[-1] + 1] if seen else hands

    confident = 0
    for i in seen:
        j = min(round(i * len(depths) / len(hands)), len(depths) - 1)  # video and depth may differ in rate
        u, v = np.clip(hands[i][0][WRIST] * capture.DEPTH_SIZE, 0, np.array(capture.DEPTH_SIZE) - 1).astype(int)
        confident += capture.read_confidence(depths[j])[v, u] == capture.HIGH_CONFIDENCE

    depth = capture.read_depth(depths[0])
    K = capture.depth_intrinsics(capture.rgb_intrinsics(root), (width, height))
    valid = (capture.read_confidence(depths[0]).ravel() == capture.HIGH_CONFIDENCE) & (depth.ravel() > 0)
    _, _, inliers, rms = capture.fit_plane(capture.backproject(depth, K)[valid])

    hand_rate = len(seen) / max(len(gesture), 1)
    confidence_rate = confident / max(len(seen), 1)
    return {
        "video": f"{width}x{height} {fps:.0f} fps, {len(hands)} frames", "depth maps": len(depths),
        "hand": hand_rate, "wrist confidence": confidence_rate,
        "plane rms (cm)": 100 * rms, "plane inliers": inliers.mean(),
        "ok": hand_rate >= HAND_MIN and confidence_rate >= CONFIDENCE_MIN and rms < PLANE_RMS_MAX,
    }


if __name__ == "__main__":
    passed = 0
    for root in sys.argv[1:]:
        r = check(root)
        passed += r["ok"]
        print(f"{'PASS' if r['ok'] else 'FAIL'}  {root}  | {r['video']}, {r['depth maps']} depth maps | "
              f"hand {r['hand']:.0%} | wrist confidence {r['wrist confidence']:.0%} | "
              f"plane rms {r['plane rms (cm)']:.2f} cm, inliers {r['plane inliers']:.0%}")
    print(f"{passed}/{len(sys.argv) - 1} demos pass")
