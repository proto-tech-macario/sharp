"""Tests for sharp_spatialize.cameras."""

from __future__ import annotations

import numpy as np
import torch

from sharp_spatialize.cameras import DEFAULT_LAYOUT, build_camera_rig, camera_at, focus_depth

PLANE_Z = 5.0


def _flat_mean_vectors() -> torch.Tensor:
    xs = torch.linspace(-3.0, 3.0, 20)
    ys = torch.linspace(-3.0, 3.0, 20)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    points = torch.stack(
        [grid_x.flatten(), grid_y.flatten(), torch.full_like(grid_x.flatten(), PLANE_Z)], dim=-1
    )
    return points.unsqueeze(0)


def test_focus_depth_is_near_the_flat_scenes_depth():
    depth = focus_depth(_flat_mean_vectors())
    assert depth == PLANE_Z


def test_focus_depth_raises_when_no_positive_depth_points():
    mean_vectors = torch.zeros(1, 5, 3)  # all z == 0, nothing strictly positive
    try:
        focus_depth(mean_vectors)
        raise AssertionError("expected ValueError for a scene with no positive-depth points")
    except ValueError:
        pass


def test_default_layout_matches_spec_v0_through_v8():
    assert DEFAULT_LAYOUT == (
        (-1.0, 1.0), (0.0, 1.0), (1.0, 1.0),
        (-1.0, 0.0), (0.0, 0.0), (1.0, 0.0),
        (-1.0, -1.0), (0.0, -1.0), (1.0, -1.0),
    )


def test_build_camera_rig_returns_exactly_nine_views():
    rig = build_camera_rig(_flat_mean_vectors(), 40.0, 32, 32, 10.0, 32, 32)
    assert len(rig) == 9


def test_v4_is_the_reference_camera():
    rig = build_camera_rig(_flat_mean_vectors(), 40.0, 32, 32, 10.0, 32, 32)
    v4 = rig[4]
    np.testing.assert_allclose(v4.R, np.eye(3, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(v4.C, np.zeros(3, dtype=np.float32), atol=1e-6)


def test_every_view_has_an_orthonormal_rotation_with_determinant_one():
    rig = build_camera_rig(_flat_mean_vectors(), 40.0, 32, 32, 10.0, 32, 32)
    for pose in rig:
        orth_error = np.max(np.abs(pose.R.T @ pose.R - np.eye(3)))
        assert orth_error < 1e-4
        assert abs(np.linalg.det(pose.R) - 1.0) < 1e-4


def test_camera_at_agrees_with_build_camera_rig_at_a_grid_angle():
    mean_vectors = _flat_mean_vectors()
    rig = build_camera_rig(mean_vectors, 40.0, 32, 32, 10.0, 32, 32)
    v0_direct = camera_at(mean_vectors, 40.0, 32, 32, -10.0, 10.0, 32, 32)
    np.testing.assert_allclose(v0_direct.R, rig[0].R)
    np.testing.assert_allclose(v0_direct.C, rig[0].C)
    np.testing.assert_allclose(v0_direct.K, rig[0].K)


def test_camera_at_supports_an_arbitrary_intermediate_angle():
    """Acceptance criterion #10: any angle within the configured range must render."""
    pose = camera_at(_flat_mean_vectors(), 40.0, 32, 32, 3.7, -4.2, 32, 32)
    orth_error = np.max(np.abs(pose.R.T @ pose.R - np.eye(3)))
    assert orth_error < 1e-4


def test_rescales_intrinsics_for_a_different_output_size():
    rig = build_camera_rig(_flat_mean_vectors(), 40.0, 32, 32, 10.0, 64, 16)
    v4 = rig[4]
    assert v4.K[0, 0] == 80.0  # fx doubled: output_width 64 / source_width 32
    assert v4.K[1, 1] == 20.0  # fy halved: output_height 16 / source_height 32
    assert v4.K[0, 2] == 32.0  # cx = output_width / 2
    assert v4.K[1, 2] == 8.0   # cy = output_height / 2
