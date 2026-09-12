"""Synthetic, geometrically consistent spatial frames for sharp_video tests.

Each frame is a checkerboard plane ray-cast through the *real* Stage 1 camera
rig (`sharp_spatialize.cameras.build_camera_rig`). Moving the plane between
frames changes Stage 1's focus depth, so the 8 outer cameras move from frame
to frame exactly as they do on real video.
"""

from __future__ import annotations

import numpy as np
import torch
from sharp_spatialize.cameras import build_camera_rig
from sharp_spatialize.hdf5_io import SpatialPhotoResult
from sharp_video.contract import SequenceInfo, SpatialFrame

PLANE_HALF_EXTENT = 1.8
F_PX = 40.0


def _mean_vectors(plane_z: float) -> torch.Tensor:
    xs = torch.linspace(-3.0, 3.0, 20)
    grid_x, grid_y = torch.meshgrid(xs, xs, indexing="xy")
    points = torch.stack(
        [grid_x.flatten(), grid_y.flatten(), torch.full_like(grid_x.flatten(), plane_z)], dim=-1
    )
    return points.unsqueeze(0)


def _render_plane(K, R, C, width, height, plane_z, phase, tilt=0.0):
    """Ray-cast the plane z = plane_z + tilt * x (world) through camera (K, R, C)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    v_grid, u_grid = np.mgrid[0:height, 0:width].astype(np.float64)
    dir_cam = np.stack([(u_grid - cx) / fx, (v_grid - cy) / fy, np.ones_like(u_grid)], axis=-1)
    dir_world = dir_cam @ R
    # Plane n . X = plane_z with n = (-tilt, 0, 1).
    denom = dir_world[..., 2] - tilt * dir_world[..., 0]
    valid = np.abs(denom) > 1e-8
    t = np.divide(plane_z - (C[2] - tilt * C[0]), denom, out=np.full_like(denom, -1.0),
                  where=valid)
    valid &= t > 1e-6
    world_xy = C[:2] + t[..., None] * dir_world[..., :2]
    valid &= (np.abs(world_xy[..., 0]) <= PLANE_HALF_EXTENT) & (
        np.abs(world_xy[..., 1]) <= PLANE_HALF_EXTENT
    )
    checker = (
        np.floor(world_xy[..., 0] + phase).astype(np.int64)
        + np.floor(world_xy[..., 1]).astype(np.int64)
    ) % 2
    rgb = np.where(
        checker[..., None] == 0, np.array([220, 60, 60]), np.array([60, 60, 220])
    ).astype(np.uint8)
    rgb = np.where(valid[..., None], rgb, np.uint8(0)).astype(np.uint8)
    # depth is camera-space Z of the hit point: t * (unit-z ray) has z_cam == t.
    depth = np.where(valid, t, 0.0).astype(np.float32)
    return rgb, depth, valid.astype(np.uint8)


def stage1_metadata(width: int, height: int) -> dict:
    """A complete Stage 1 metadata dict for a synthetic frame."""
    return {
        "format_version": "1.0", "model_name": "SHARP", "model_version": "test",
        "source_width": width, "source_height": height,
        "source_filename": "frame.png",
        "output_width": width, "output_height": height,
        "num_views": 9, "view_layout": "3x3",
        "horizontal_angle": 10.0, "vertical_angle": 10.0,
        "coordinate_system": "OpenCV",
        "pose_convention": "R is world->camera; C is the camera's world position",
        "depth_definition": "camera_z", "depth_unit": "meter",
    }


def make_stage1_result(
    width: int = 32, height: int = 32, plane_z: float = 5.0, phase: float = 0.0,
    tilt: float = 0.0,
) -> SpatialPhotoResult:
    """One consistent 9-view Stage 1 result of a plane at depth `plane_z`.

    `tilt` slopes the plane (z = plane_z + tilt * x) so that no view -- the
    centre one included -- sees a single constant depth.
    """
    rig = build_camera_rig(
        _mean_vectors(plane_z), F_PX, width, height, 10.0, width, height,
    )
    rendered = [_render_plane(p.K, p.R, p.C, width, height, plane_z, phase, tilt)
                for p in rig]
    return SpatialPhotoResult(
        rgb=np.stack([r[0] for r in rendered]),
        depth=np.stack([r[1] for r in rendered]),
        mask=np.stack([r[2] for r in rendered]),
        K=np.stack([p.K for p in rig]),
        R=np.stack([p.R for p in rig]),
        C=np.stack([p.C for p in rig]),
        metadata=stage1_metadata(width, height),
    )


def make_frames(
    num_frames: int,
    width: int = 32,
    height: int = 32,
    fps: float = 30.0,
    drift: float = 0.0,
    motion: float = 0.0,
    tilt: float = 0.0,
) -> list[SpatialFrame]:
    """`num_frames` consecutive synthetic frames of a plane.

    `drift` moves the plane (and so the rig) by that many metres per frame,
    `motion` slides the texture per frame, and `tilt` slopes the plane (see
    `make_stage1_result`).
    """
    return [
        SpatialFrame.from_stage1(
            make_stage1_result(width, height, 5.0 + drift * t, motion * t, tilt),
            timestamp=t / fps,
            pts=t * 512,
            source_frame_index=t,
        )
        for t in range(num_frames)
    ]


def make_info(width: int = 32, height: int = 32, fps: float = 30.0) -> SequenceInfo:
    """Sequence-level info matching `make_frames`."""
    return SequenceInfo(
        width=width, height=height, fps=fps, time_base="1/15360",
        source_filename="clip.mp4",
        extra={"horizontal_angle": 10.0, "vertical_angle": 10.0,
               "model_name": "SHARP", "model_version": "test"},
    )
