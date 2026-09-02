"""Builds the 3x3 virtual-camera rig for a SHARP scene.

Every camera orbits the same fixed focus point at the scene's focus depth and
looks back at it -- this produces real parallax/occlusion between views,
unlike a pure pan/tilt rotation in place. Geometry is expressed directly in
the reference camera's own coordinate frame (X right, Y down, Z forward, per
the OpenCV convention SHARP already uses), which doubles as "world" here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sharp.utils.camera import create_camera_matrix

FOCUS_DEPTH_QUANTILE = 0.1

# (h_sign, v_sign) for V0..V8, per the spec's fixed 3x3 table:
#   V0=(-a,+a) V1=(0,+a) V2=(+a,+a)
#   V3=(-a, 0) V4=(0, 0) V5=(+a, 0)
#   V6=(-a,-a) V7=(0,-a) V8=(+a,-a)
DEFAULT_LAYOUT: tuple[tuple[float, float], ...] = (
    (-1.0, 1.0), (0.0, 1.0), (1.0, 1.0),
    (-1.0, 0.0), (0.0, 0.0), (1.0, 0.0),
    (-1.0, -1.0), (0.0, -1.0), (1.0, -1.0),
)


@dataclass
class CameraPose:
    """One virtual camera's parameters, OpenCV convention (world->camera)."""

    K: np.ndarray  # (3, 3) float32
    R: np.ndarray  # (3, 3) float32
    C: np.ndarray  # (3,) float32


def focus_depth(mean_vectors: torch.Tensor) -> float:
    """The reference-camera-space depth every view orbits around.

    Uses the 10th percentile of the scene's positive-depth points, same
    heuristic as `sharp.utils.camera.PinholeCameraModel`.
    """
    depths = mean_vectors.reshape(-1, 3)[:, 2]
    depths = depths[depths > 0]
    if depths.numel() == 0:
        raise ValueError("Scene has no positive-depth points; cannot compute a focus depth.")
    return float(torch.quantile(depths.float().cpu(), FOCUS_DEPTH_QUANTILE))


def camera_at(
    mean_vectors: torch.Tensor,
    f_px: float,
    source_width: int,
    source_height: int,
    h_angle_deg: float,
    v_angle_deg: float,
    output_width: int,
    output_height: int,
) -> CameraPose:
    """Build one camera pose orbiting the scene's focus point by the given angles.

    `(0, 0)` reduces to the reference camera's own pose (identity R, C at the
    origin). Any angle works, not just the 9 grid values -- this is what
    lets an arbitrary intermediate view be rendered for testing.
    """
    depth = focus_depth(mean_vectors)

    h_rad = np.radians(h_angle_deg)
    v_rad = np.radians(v_angle_deg)
    eye_pos = torch.tensor(
        [depth * np.tan(h_rad), depth * np.tan(v_rad), 0.0], dtype=torch.float32
    )
    look_at = torch.tensor([0.0, 0.0, depth], dtype=torch.float32)
    world_up = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32)

    extrinsics = create_camera_matrix(eye_pos, look_at, world_up, inverse=True)
    R = extrinsics[:3, :3].numpy().astype(np.float32)
    C = eye_pos.numpy().astype(np.float32)

    fx = f_px * (output_width / source_width)
    fy = f_px * (output_height / source_height)
    K = np.array(
        [[fx, 0.0, output_width / 2.0], [0.0, fy, output_height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return CameraPose(K=K, R=R, C=C)


def build_camera_rig(
    mean_vectors: torch.Tensor,
    f_px: float,
    source_width: int,
    source_height: int,
    angle_deg: float,
    output_width: int,
    output_height: int,
) -> list[CameraPose]:
    """Build the 9-camera rig V0..V8 for the given angle (applied to both axes)."""
    if angle_deg <= 0:
        raise ValueError(f"angle_deg must be positive, got {angle_deg}")
    return [
        camera_at(
            mean_vectors, f_px, source_width, source_height,
            h_sign * angle_deg, v_sign * angle_deg, output_width, output_height,
        )
        for h_sign, v_sign in DEFAULT_LAYOUT
    ]
