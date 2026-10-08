"""Hand keypoints from MediaPipe's hand landmarker: 21 points per frame.

`image` points are normalised to the frame (x right, y down, in [0, 1]); `metric` points are MediaPipe's own
metric estimate, in metres, around the hand's centre. The depth of the hand is not known from the image alone:
it comes from the LiDAR in h2r.extract.
"""
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
WRIST, THUMB_TIP, INDEX_TIP = 0, 4, 8
PALM = [0, 5, 9, 13, 17]  # wrist and the four knuckles: the most rigid part of the hand


def track_hand(frames, fps):
    """Per frame, (image (21, 2), metric (21, 3)) or None where no hand is found."""
    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_hands=1)
    out = []
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for i, rgb in enumerate(frames):
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / fps))
            if not result.hand_landmarks:
                out.append(None)
                continue
            image = np.array([[p.x, p.y] for p in result.hand_landmarks[0]])
            metric = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]])
            out.append((image, metric))
    return out
