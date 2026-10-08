"""Stage P2: from a Stray Scanner recording to a demonstration in the table frame (h2r.retarget.Demo).

The camera did not move during the recordings (ARKit odometry: under 0.4 cm and 1.5 degrees per take), so every
frame is lifted into the camera frame of the first one. The hand is placed in depth by the LiDAR under the palm,
its fingertips by MediaPipe's metric offsets from the palm. Bowl and plate are found by colour and height above
the table, and their centres by circle fits: the bowl on its rim, the plate on its outline.

Usage: python -m h2r.extract data/raw/<take> [...] [--frame=camera]
       (writes data/processed[_camera]/<take>.npz and a check image)
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from h2r import capture
from h2r.hand import INDEX_TIP, PALM, THUMB_TIP, track_hand
from h2r.kalman import rts_smooth
from h2r.retarget import Demo

BOWL_HSV = ((95, 60, 30), (135, 255, 200))   # dark blue ceramic under the room's warm light
PLATE_HSV = ((0, 0, 160), (40, 80, 255))     # white plate
BOWL_RIM_MIN = 0.06                           # rim points: higher than this above the table (bowl is 8.5 cm)


def circle_fit(xy, reject=0.01, rounds=3):
    """Least-squares circle through 2D points (Kasa), refitted without points off by more than `reject`."""
    keep = np.ones(len(xy), bool)
    for _ in range(rounds):
        A = np.c_[2 * xy[keep], np.ones(keep.sum())]
        (cx, cy, c), *_ = np.linalg.lstsq(A, (xy[keep] ** 2).sum(1), rcond=None)
        centre, radius = np.array([cx, cy]), float(np.sqrt(c + cx ** 2 + cy ** 2))
        keep = np.abs(np.linalg.norm(xy - centre, axis=1) - radius) < reject
    return centre, radius


def table_frame(normal, forward):
    """Rotation whose rows are the table axes in camera coordinates: x right, y forward, z up."""
    z = normal / np.linalg.norm(normal)
    y = forward - (forward @ z) * z
    y /= np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z])


def lift(uv, depth, conf, K, patch=2):
    """3D camera point under a normalised image point, from the median of nearby confident depths (NaN if none)."""
    h, w = depth.shape
    u, v = int(uv[0] * w), int(uv[1] * h)
    if not (0 <= u < w and 0 <= v < h):
        return np.full(3, np.nan)
    win = np.s_[max(v - patch, 0):v + patch + 1, max(u - patch, 0):u + patch + 1]
    ok = (conf[win] >= 1) & (depth[win] > 0)
    if not ok.any():
        return np.full(3, np.nan)
    z = np.median(depth[win][ok])
    return np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])


def hand_points(hand, depth, conf, K):
    """Pinch point (between thumb and index tips) and aperture, in the camera frame.

    MediaPipe's metric frame is aligned with the camera: on a hand lying flat (take a767851eec, 1.0-1.7 s),
    palm + metric offsets put the fingertips 3.2 cm (median) from their direct LiDAR measure, against 6.7 and
    9.3 cm with the z, or y and z, axes flipped.
    """
    image, metric = hand
    palm = lift(image[PALM].mean(0), depth, conf, K)
    tips = metric[[THUMB_TIP, INDEX_TIP]]
    pinch = palm + tips.mean(0) - metric[PALM].mean(0)
    return pinch, float(np.linalg.norm(tips[0] - tips[1]))


def scene(rgb, depth, conf, K):
    """Table plane, bowl and plate in one frame without the hand.

    Returns the plane (unit normal towards the camera, offset: height = n . p + d), the bowl base centre, the plate
    centre and the plate diameter, all in the camera frame.
    """
    points = capture.backproject(depth, K)
    valid = (conf.ravel() == capture.HIGH_CONFIDENCE) & (depth.ravel() > 0)
    n, d, _, _ = capture.fit_plane(points[valid])
    n, d = (n, d) if d > 0 else (-n, -d)  # the camera, at the origin, is above the table
    height = (points @ n + d).reshape(depth.shape)
    hsv = cv2.cvtColor(cv2.resize(rgb, depth.shape[::-1], interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2HSV)
    usable = (conf >= 1) & (depth > 0)

    def largest(mask):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        return labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) if count > 1 else mask

    def on_table(p):  # project camera points onto the table plane
        return p - np.outer(p @ n + d, n)

    basis = table_frame(n, np.array([0.0, 0.0, 1.0]))[:2]  # any in-plane axes, for 2D circle fits

    bowl = largest(cv2.inRange(hsv, *BOWL_HSV).astype(bool) & usable & (height > 0.005) & (height < 0.12))
    rim = bowl & (height > BOWL_RIM_MIN)
    rim_points = on_table(points[rim.ravel()])
    centre_2d, rim_r = circle_fit(rim_points @ basis.T)
    bowl_base = on_table(rim_points.mean(0)[None])[0]
    bowl_base += basis.T @ (centre_2d - bowl_base @ basis.T)

    # Below the table plane lie the floor and the fridge front, white too: keep heights from -1 cm.
    plate = largest(cv2.inRange(hsv, *PLATE_HSV).astype(bool) & usable & (height > -0.01) & (height < 0.04) & ~bowl)
    outline = plate & ~cv2.erode(plate.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    outline_points = on_table(points[outline.ravel()])
    plate_2d, plate_r = circle_fit(outline_points @ basis.T)
    plate_centre = on_table(outline_points.mean(0)[None])[0]
    plate_centre += basis.T @ (plate_2d - plate_centre @ basis.T)
    return (n, d), bowl_base, plate_centre, 2 * plate_r, {"bowl": bowl, "plate": plate, "rim": rim,
                                                           "rim_diameter": 2 * rim_r}


def extract(root, frame="objects"):
    """One recording to a Demo in the table frame (origin at the bowl's start, z up).

    The tape's colour is too close to the table's under the room's light to give the forward axis, so it comes
    either from the objects (frame="objects": y from bowl to plate; the bowl's sideways offsets are lost and a plate
    set between the demonstrator and the bowl turns the frame round) or from the camera (frame="camera": y along
    the camera's horizontal axis projected on the table, which the protocol put parallel to the tape, landscape,
    the far end of the table on the right of the image).
    """
    width, height, _, fps = capture.video_info(root)
    K = capture.depth_intrinsics(capture.rgb_intrinsics(root), (width, height))
    frames = list(capture.rgb_frames(root, scale=0.5))
    depths = capture.depth_paths(root)
    hands = track_hand(frames, fps)
    seen = [i for i, h in enumerate(hands) if h is not None]
    free = [i for i, h in enumerate(hands) if h is None] or [0]

    def depth_at(i):
        path = depths[min(i, len(depths) - 1)]
        return capture.read_depth(path), capture.read_confidence(path)

    (n, d), bowl, plate, plate_diameter, info = scene(frames[free[0]], *depth_at(free[0]), K)
    _, bowl_end, _, _, _ = scene(frames[free[-1]], *depth_at(free[-1]), K)

    pinch = np.full((len(hands), 3), np.nan)
    aperture = np.full(len(hands), np.nan)
    for i in seen:
        pinch[i], aperture[i] = hand_points(hands[i], *depth_at(i), K)

    R = table_frame(n, np.array([1.0, 0.0, 0.0]) if frame == "camera" else plate - bowl)
    span = slice(seen[0], seen[-1] + 1)  # the gesture, from the first to the last frame the hand is seen
    track = (pinch[span] - bowl) @ R.T
    good = ~np.isnan(aperture[span])
    t = np.arange(len(track)) / fps
    demo = Demo(t=t, pinch=rts_smooth(track, dt=1 / fps),
                aperture=np.interp(t, t[good], aperture[span][good]),
                bowl=np.zeros(3), plate=(plate - bowl) @ R.T)
    checks = {"plate_diameter": plate_diameter, "rim_diameter": info["rim_diameter"],
              "bowl_to_plate": float(np.linalg.norm(plate - bowl)),
              "bowl_end_to_plate": float(np.linalg.norm((bowl_end - plate) @ R.T[:, :2])),
              "hand_coverage": float(good.mean()), "fps": fps}
    camera = {"K": K, "R": R, "origin": bowl, "first": seen[0], "frame": frames[free[0]]}
    return demo, checks, camera


def check_image(demo, camera, path):
    """The smoothed pinch track drawn back onto the first frame: green open, red closed."""
    from h2r.retarget import events

    img = cv2.cvtColor(camera["frame"], cv2.COLOR_RGB2BGR)
    s = img.shape[1] / capture.DEPTH_SIZE[0]
    P = demo.pinch @ camera["R"] + camera["origin"]  # back to the camera frame
    uv = np.c_[camera["K"][0, 0] * P[:, 0] / P[:, 2] + camera["K"][0, 2],
               camera["K"][1, 1] * P[:, 1] / P[:, 2] + camera["K"][1, 2]] * s
    g, r = events(demo.pinch, demo.aperture, demo.plate)
    for k in range(1, len(uv)):
        colour = (0, 0, 255) if g <= k < r else (0, 255, 0)
        cv2.line(img, tuple(uv[k - 1].astype(int)), tuple(uv[k].astype(int)), colour, 2)
    for k, label in ((g, "grasp"), (r, "release")):
        cv2.circle(img, tuple(uv[k].astype(int)), 7, (255, 255, 255), 2)
        cv2.putText(img, label, tuple(uv[k].astype(int) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(path), img)


if __name__ == "__main__":
    frame = "camera" if "--frame=camera" in sys.argv else "objects"
    out = Path(__file__).resolve().parent.parent / "data" / ("processed_camera" if frame == "camera" else "processed")
    out.mkdir(parents=True, exist_ok=True)
    for root in (a for a in sys.argv[1:] if not a.startswith("--")):
        name = Path(root).name
        demo, checks, camera = extract(root, frame)
        np.savez(out / f"{name}.npz", t=demo.t, pinch=demo.pinch, aperture=demo.aperture, bowl=demo.bowl,
                 plate=demo.plate, **checks)
        check_image(demo, camera, out / f"{name}.jpg")
        print(f"{name} | " + " | ".join(f"{k} {v:.3f}" for k, v in checks.items()), flush=True)
