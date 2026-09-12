"""Tests for the OpenCV -> TMIV conversions in sharp_video.miv (camera, depth, colour)."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import make_frames

from sharp_video.miv.camera import from_tmiv_pose, tmiv_view_json, to_tmiv_pose
from sharp_video.miv.depth import depth_to_geometry, geometry_to_depth
from sharp_video.miv.yuv import rgb_to_yuv420, yuv420_to_rgb


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _tmiv_project(point_world, view):
    """TMIV's perspective projection (Projector.h) with a camera->world Euler pose."""
    yaw, pitch, roll = np.radians(view["Rotation"])
    r_c2w = _rz(yaw) @ _ry(pitch) @ _rx(roll)
    p = r_c2w.T @ (point_world - np.asarray(view["Position"]))
    fx, fy = view["Focal"]
    cx, cy = view["Principle_point"]
    return np.array([-fx * p[1] / p[0] + cx, -fy * p[2] / p[0] + cy]), p[0]


def _opencv_project(point_world, K, R, C):
    p = R @ (point_world - C)
    return (K @ (p / p[2]))[:2], p[2]


P = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # OpenCV -> TMIV


@pytest.mark.parametrize("drift", [0.0, 0.7])
def test_pose_round_trip_on_stage1_rig(drift):
    frame = make_frames(2, drift=drift)[1]
    for view in frame.views:
        position, rotation = to_tmiv_pose(view.R, view.C)
        R, C = from_tmiv_pose(position, rotation)
        np.testing.assert_allclose(R, view.R, atol=1e-5)
        np.testing.assert_allclose(C, view.C, atol=1e-5)


def test_reference_view_maps_to_identity():
    position, rotation = to_tmiv_pose(np.eye(3, dtype=np.float32), np.zeros(3, np.float32))
    np.testing.assert_allclose(position, 0.0, atol=1e-7)
    np.testing.assert_allclose(rotation, 0.0, atol=1e-7)


def test_tmiv_projection_equals_opencv_projection():
    frame = make_frames(2, drift=0.4)[1]
    rng = np.random.default_rng(0)
    points_cv = rng.uniform([-1, -1, 3], [1, 1, 8], size=(20, 3))
    for view in frame.views:
        cam = tmiv_view_json("v0", view.K, view.R, view.C, 32, 32, (0.1, 100.0))
        for point in points_cv:
            uv_cv, z_cv = _opencv_project(point, view.K, view.R, view.C)
            uv_tmiv, x_tmiv = _tmiv_project(P @ point, cam)
            np.testing.assert_allclose(uv_tmiv, uv_cv, atol=1e-3)
            assert x_tmiv == pytest.approx(z_cv, rel=1e-5)  # TMIV depth == camera Z


def test_view_json_fields():
    view = make_frames(1)[0].views[0]
    cam = tmiv_view_json("v0", view.K, view.R, view.C, 32, 32, (0.5, 50.0))
    assert cam["Name"] == "v0"
    assert cam["Resolution"] == [32, 32]
    assert cam["Projection"] == "Perspective"
    assert cam["Depth_range"] == [0.5, 50.0]
    assert cam["HasInvalidDepth"] is True
    assert cam["BitDepthColor"] == 10 and cam["BitDepthDepth"] == 16
    assert cam["ColorSpace"] == "YUV420" and cam["DepthColorSpace"] == "YUV420"
    assert cam["Focal"] == pytest.approx([float(view.K[0, 0]), float(view.K[1, 1])])
    assert cam["Principle_point"] == pytest.approx([float(view.K[0, 2]), float(view.K[1, 2])])


def test_depth_round_trip_and_invalid():
    depth = np.array([[0.0, 0.5, 2.0], [7.3, 40.0, 0.0]], dtype=np.float32)
    mask = (depth > 0).astype(np.uint8)
    samples, clamped = depth_to_geometry(depth, mask, near=0.1, far=1000.0)
    assert samples.dtype == np.uint16
    assert clamped == 0
    assert samples[0, 0] == 0 and samples[1, 2] == 0  # invalid -> 0
    assert samples[mask == 1].min() >= 1
    back, back_mask = geometry_to_depth(samples, near=0.1, far=1000.0)
    np.testing.assert_array_equal(back_mask, mask)
    rel = np.abs(back[mask == 1] - depth[mask == 1]) / depth[mask == 1]
    # 16-bit linear disparity over [0.1, 1000] m: half a step is ~7.6e-5 * z relative.
    assert rel.max() < 5e-3
    assert np.all(back[mask == 0] == 0.0)


def test_depth_out_of_range_is_clamped_and_counted():
    depth = np.array([[0.01, 5000.0, 3.0]], dtype=np.float32)
    samples, clamped = depth_to_geometry(depth, np.ones_like(depth, np.uint8), 0.1, 1000.0)
    assert clamped == 2
    # Near clamps to the maximum sample; far clamps to the smallest *valid* sample
    # (0 is reserved for "invalid"), exactly as TMIV's quantization law defines.
    assert samples[0, 0] == 65535
    assert samples[0, 1] == 1
    back, _ = geometry_to_depth(samples, 0.1, 1000.0)
    assert back[0, 0] == pytest.approx(0.1, rel=1e-6)
    assert back[0, 2] == pytest.approx(3.0, rel=1e-3)


def test_depth_range_must_be_ordered():
    with pytest.raises(ValueError):
        depth_to_geometry(np.ones((2, 2), np.float32), np.ones((2, 2), np.uint8), 5.0, 1.0)


def test_yuv_round_trip_is_exact_on_constant_2x2_blocks():
    rng = np.random.default_rng(1)
    blocks = rng.integers(0, 256, size=(16, 24, 3), dtype=np.uint8)
    rgb = np.repeat(np.repeat(blocks, 2, axis=0), 2, axis=1)  # no chroma detail lost
    back = yuv420_to_rgb(*rgb_to_yuv420(rgb, bit_depth=10), bit_depth=10)
    assert np.abs(back.astype(np.int16) - rgb).max() <= 1


def test_yuv_round_trip_on_gradient_image():
    y, x = np.mgrid[0:32, 0:48]
    rgb = np.stack([x * 5, y * 7, 255 - x * 5], axis=-1).clip(0, 255).astype(np.uint8)
    Y, U, V = rgb_to_yuv420(rgb, bit_depth=10)
    assert Y.shape == (32, 48) and U.shape == (16, 24) and V.shape == (16, 24)
    assert Y.dtype == np.uint16 and Y.max() <= 940 and Y.min() >= 64  # limited range
    back = yuv420_to_rgb(Y, U, V, bit_depth=10)
    # Steep saturated gradients lose some chroma detail to 4:2:0 subsampling.
    assert np.abs(back.astype(np.float32) - rgb).mean() < 3.0


def test_yuv_primaries_are_bt709():
    white = np.full((2, 2, 3), 255, np.uint8)
    red = np.zeros((2, 2, 3), np.uint8)
    red[..., 0] = 255
    Yw, Uw, Vw = rgb_to_yuv420(white, bit_depth=10)
    assert Yw[0, 0] == 940 and Uw[0, 0] == 512 and Vw[0, 0] == 512
    Yr, _, _ = rgb_to_yuv420(red, bit_depth=10)
    assert Yr[0, 0] == round(64 + 876 * 0.2126)


def test_yuv_odd_size_rejected():
    with pytest.raises(ValueError, match="even"):
        rgb_to_yuv420(np.zeros((3, 4, 3), np.uint8))
