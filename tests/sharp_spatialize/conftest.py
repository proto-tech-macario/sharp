"""Shared pytest fixtures for sharp_spatialize tests.

`consistent_spatial_photo_result` builds a small synthetic scene (a flat
checkerboard at world z=PLANE_Z) and ray-casts it through the *real* camera
rig from `cameras.py`, so tests get geometrically self-consistent K/R/C/
depth/rgb data without needing SHARP or a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sharp_spatialize.cameras import build_camera_rig
from sharp_spatialize.hdf5_io import SpatialPhotoResult

PLANE_Z = 5.0
PLANE_HALF_EXTENT = 1.8
OUTPUT_SIZE = 32


def _synthetic_mean_vectors() -> torch.Tensor:
    xs = torch.linspace(-3.0, 3.0, 20)
    ys = torch.linspace(-3.0, 3.0, 20)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    points = torch.stack(
        [grid_x.flatten(), grid_y.flatten(), torch.full_like(grid_x.flatten(), PLANE_Z)], dim=-1
    )
    return points.unsqueeze(0)


def _render_plane_view(K: np.ndarray, R: np.ndarray, C: np.ndarray, size: int):
    """Ray-cast a checkerboard plane at world z=PLANE_Z through camera (K, R, C)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    v_grid, u_grid = np.mgrid[0:size, 0:size].astype(np.float64)
    dir_cam = np.stack([(u_grid - cx) / fx, (v_grid - cy) / fy, np.ones_like(u_grid)], axis=-1)
    dir_world = dir_cam @ R  # == R.T @ dir_cam per-pixel, since x_world = R.T @ x_cam + C

    denom = dir_world[..., 2]
    valid = np.abs(denom) > 1e-8
    t = np.divide(PLANE_Z - C[2], denom, out=np.full_like(denom, -1.0), where=valid)
    valid &= t > 1e-6

    world_xy = C[:2] + t[..., None] * dir_world[..., :2]
    valid &= (np.abs(world_xy[..., 0]) <= PLANE_HALF_EXTENT) & (
        np.abs(world_xy[..., 1]) <= PLANE_HALF_EXTENT
    )

    checker = (
        np.floor(world_xy[..., 0]).astype(np.int64) + np.floor(world_xy[..., 1]).astype(np.int64)
    ) % 2
    rgb = np.where(
        checker[..., None] == 0, np.array([220, 60, 60]), np.array([60, 60, 220])
    ).astype(np.uint8)
    rgb = np.broadcast_to(rgb, (size, size, 3)).copy()
    depth = np.where(valid, t, 0.0).astype(np.float32)
    mask = valid.astype(np.uint8)
    return rgb, depth, mask


def _full_metadata() -> dict:
    return {
        "format_version": "1.0", "model_name": "SHARP", "model_version": "test",
        "source_width": OUTPUT_SIZE, "source_height": OUTPUT_SIZE,
        "output_width": OUTPUT_SIZE, "output_height": OUTPUT_SIZE,
        "num_views": 9, "view_layout": "3x3",
        "horizontal_angle": 10.0, "vertical_angle": 10.0,
        "coordinate_system": "OpenCV",
        "pose_convention": "R world->camera, C camera position",
        "depth_definition": "camera_z", "depth_unit": "meter",
    }


def make_consistent_result() -> SpatialPhotoResult:
    """A 9-view SpatialPhotoResult that is fully geometrically self-consistent."""
    mean_vectors = _synthetic_mean_vectors()
    rig = build_camera_rig(
        mean_vectors, f_px=40.0, source_width=OUTPUT_SIZE, source_height=OUTPUT_SIZE,
        angle_deg=10.0, output_width=OUTPUT_SIZE, output_height=OUTPUT_SIZE,
    )
    rgb_list, depth_list, mask_list = [], [], []
    for pose in rig:
        rgb, depth, mask = _render_plane_view(pose.K, pose.R, pose.C, OUTPUT_SIZE)
        rgb_list.append(rgb)
        depth_list.append(depth)
        mask_list.append(mask)
    return SpatialPhotoResult(
        rgb=np.stack(rgb_list), depth=np.stack(depth_list), mask=np.stack(mask_list),
        K=np.stack([p.K for p in rig]), R=np.stack([p.R for p in rig]),
        C=np.stack([p.C for p in rig]), metadata=_full_metadata(),
    )


@pytest.fixture
def consistent_spatial_photo_result() -> SpatialPhotoResult:
    """Fresh copy of a geometrically consistent 9-view result for each test."""
    return make_consistent_result()
