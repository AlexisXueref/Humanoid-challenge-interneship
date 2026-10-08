"""Constant-velocity Kalman filter with a Rauch-Tung-Striebel backward pass, to smooth a recorded 3D track.

The whole demonstration is available offline, so the backward pass uses future samples too; each axis is
filtered independently. `accel_std` is the hand's unmodelled acceleration, `meas_std` the tracking noise.
NaN samples (frames where the hand was not found) get no update: the motion model carries the estimate across.
"""
import numpy as np


def rts_smooth(track, dt, accel_std=3.0, meas_std=0.01):
    track = np.asarray(track, float)
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = accel_std**2 * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
    H, R = np.array([[1.0, 0.0]]), meas_std**2
    out = np.empty_like(track)
    for axis in range(track.shape[1]):
        z = track[:, axis]
        n = len(z)
        x_f, P_f = np.zeros((n, 2)), np.zeros((n, 2, 2))
        x_p, P_p = np.zeros((n, 2)), np.zeros((n, 2, 2))
        first = np.flatnonzero(~np.isnan(z))[0]
        x, P = np.array([z[first], 0.0]), np.diag([R, 1.0])
        for k in range(n):
            if k:
                x, P = F @ x, F @ P @ F.T + Q
            x_p[k], P_p[k] = x, P
            if not np.isnan(z[k]):
                K = P @ H.T / (H @ P @ H.T + R)
                x = x + (K * (z[k] - H @ x)).ravel()
                P = (np.eye(2) - K @ H) @ P
            x_f[k], P_f[k] = x, P
        xs = x_f.copy()
        for k in range(n - 2, -1, -1):
            C = P_f[k] @ F.T @ np.linalg.inv(P_p[k + 1])
            xs[k] = x_f[k] + C @ (xs[k + 1] - x_p[k + 1])
        out[:, axis] = xs[:, 0]
    return out
