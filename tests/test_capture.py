import numpy as np

from h2r.capture import backproject, depth_intrinsics, fit_plane


def test_depth_intrinsics_scale_to_lidar_resolution():
    K = np.array([[1400.0, 0, 960], [0, 1400.0, 720], [0, 0, 1]])
    Kd = depth_intrinsics(K, rgb_size=(1920, 1440))
    # 256/1920 = 192/1440 = 2/15
    np.testing.assert_allclose(Kd, [[1400 * 2 / 15, 0, 128], [0, 1400 * 2 / 15, 96], [0, 0, 1]])


def test_backproject_principal_point_lies_on_optical_axis():
    K = np.array([[200.0, 0, 128], [0, 200.0, 96], [0, 0, 1]])
    points = backproject(np.full((192, 256), 0.5), K)
    np.testing.assert_allclose(points[96 * 256 + 128], [0.0, 0.0, 0.5])
    np.testing.assert_allclose(points[96 * 256 + 128 + 100], [0.25, 0.0, 0.5])  # x = 100 px * z / fx


def test_fit_plane_scales_to_a_full_depth_map():
    # A whole 256 x 192 depth map: a full SVD would ask for a 60,000 x 60,000 matrix (27 GB)
    rng = np.random.default_rng(0)
    points = np.c_[rng.uniform(-0.5, 0.5, (60000, 2)), np.full(60000, 0.6)]
    normal, d, inliers, rms = fit_plane(points, iterations=20, rng=rng)
    assert abs(abs(normal[2]) - 1) < 1e-9 and inliers.all() and rms < 1e-9


def test_fit_plane_finds_the_table_under_objects():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-0.4, 0.4, (2000, 2))
    # Tilted table z = 0.6 + 0.3 x with 3 mm of noise; objects sit 8 to 28 cm closer to the camera.
    table = np.c_[xy, 0.6 + 0.3 * xy[:, 0] + rng.normal(0, 0.003, 2000)]
    objects = rng.uniform([-0.4, -0.4, 0.2], [0.4, 0.4, 0.4], (500, 3))
    normal, d, inliers, rms = fit_plane(np.r_[table, objects], rng=rng)
    expected = np.array([0.3, 0.0, -1.0]) / np.linalg.norm([0.3, 0.0, -1.0])
    assert abs(abs(normal @ expected) - 1) < 1e-3
    assert inliers[:2000].mean() > 0.95
    assert not inliers[2000:].any()
    assert rms < 0.005
