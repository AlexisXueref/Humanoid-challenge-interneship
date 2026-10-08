"""Reading a Stray Scanner recording (iPhone LiDAR) and the geometry every stage shares.

A recording folder holds rgb.mp4, depth/NNNNNN.png (uint16, millimetres), confidence/NNNNNN.png
(0, 1 or 2), camera_matrix.csv and odometry.csv. Format: github.com/strayrobots/scanner, docs/format.md.
"""
from pathlib import Path

import cv2
import numpy as np

DEPTH_SIZE = (256, 192)  # (width, height) of the LiDAR depth and confidence maps
HIGH_CONFIDENCE = 2


def rgb_intrinsics(root):
    """3x3 camera matrix of the RGB video."""
    return np.loadtxt(Path(root) / "camera_matrix.csv", delimiter=",")


def video_info(root):
    """(width, height, frame count, fps) of rgb.mp4."""
    cap = cv2.VideoCapture(str(Path(root) / "rgb.mp4"))
    info = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return info


def rgb_frames(root, scale=1.0):
    """Yield the video frames as RGB uint8 arrays, optionally downscaled."""
    cap = cv2.VideoCapture(str(Path(root) / "rgb.mp4"))
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                return
            if scale != 1.0:
                bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def depth_paths(root):
    return sorted((Path(root) / "depth").glob("*.png"))


def read_depth(path):
    """Depth map in metres (0 where the LiDAR returned nothing)."""
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0


def read_confidence(depth_path):
    """Confidence map matching a depth file (same name under confidence/)."""
    path = Path(depth_path).parent.parent / "confidence" / Path(depth_path).name
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def depth_intrinsics(K_rgb, rgb_size):
    """Scale the RGB camera matrix to the LiDAR depth resolution."""
    K = np.array(K_rgb, dtype=float)
    K[0] *= DEPTH_SIZE[0] / rgb_size[0]
    K[1] *= DEPTH_SIZE[1] / rgb_size[1]
    return K


def backproject(depth, K):
    """Every depth pixel as a 3D point in the camera frame, row-major, shape (H*W, 3)."""
    v, u = np.indices(depth.shape)
    x = (u - K[0, 2]) * depth / K[0, 0]
    y = (v - K[1, 2]) * depth / K[1, 1]
    return np.stack([x, y, depth], axis=-1).reshape(-1, 3)


def fit_plane(points, threshold=0.02, iterations=300, rng=None):
    """RANSAC plane refined by least squares on its inliers.

    Returns (unit normal n, offset d, inlier mask, inlier RMS distance) with n . p + d = 0.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    best = None
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        inliers = np.abs((points - a) @ (n / np.linalg.norm(n))) < threshold
        if best is None or inliers.sum() > best.sum():
            best = inliers
    centroid = points[best].mean(axis=0)
    # full_matrices=False: the default builds an N x N matrix, 3.7 GB for a 22,000-point depth map
    normal = np.linalg.svd(points[best] - centroid, full_matrices=False)[2][-1]
    dist = points @ normal - normal @ centroid
    inliers = np.abs(dist) < threshold
    return normal, -normal @ centroid, inliers, float(np.sqrt(np.mean(dist[inliers] ** 2)))
