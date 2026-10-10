"""Authoritative camera mathematics (spec Part III section 4/9).

Every function has a vectorised `*_many` form; the scalar forms are thin
wrappers, so Builder and Presenter cannot disagree on conventions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import GeometryError
from ..model.camera import PROJECTION_PINHOLE, Camera

ROTATION_TOLERANCE = 1e-4


def validate(camera: Camera) -> None:
    if camera.projection_type != PROJECTION_PINHOLE:
        raise GeometryError(f"unsupported projection_type {camera.projection_type!r}")
    if camera.width <= 0 or camera.height <= 0:
        raise GeometryError(f"camera dimensions must be positive, got {camera.width}x{camera.height}")
    scalars = np.array([camera.fx, camera.fy, camera.cx, camera.cy, camera.skew], dtype=np.float64)
    if not np.isfinite(scalars).all():
        raise GeometryError("camera intrinsics must be finite")
    if camera.fx <= 0 or camera.fy <= 0:
        raise GeometryError("camera focal lengths must be positive")
    R = np.asarray(camera.rotation, dtype=np.float64)
    t = np.asarray(camera.translation, dtype=np.float64)
    if R.shape != (3, 3) or t.shape != (3,) or not (np.isfinite(R).all() and np.isfinite(t).all()):
        raise GeometryError("camera pose must be a finite 3x3 rotation and 3-vector translation")
    if np.abs(R @ R.T - np.eye(3)).max() > ROTATION_TOLERANCE or abs(np.linalg.det(R) - 1.0) > ROTATION_TOLERANCE:
        raise GeometryError("camera rotation is not a proper rotation matrix")


def camera_center(camera: Camera) -> np.ndarray:
    """C_world = -R^T t."""
    return -np.asarray(camera.rotation).T @ np.asarray(camera.translation)


def forward_direction(camera: Camera) -> np.ndarray:
    """Unit viewing direction (camera +Z axis) in world coordinates."""
    f = np.asarray(camera.rotation)[2, :]
    return f / np.linalg.norm(f)


def view_angle(base_camera: Camera, source_camera: Camera) -> float:
    d = float(np.dot(forward_direction(base_camera), forward_direction(source_camera)))
    return float(np.arccos(np.clip(d, -1.0, 1.0)))


def valid_depth(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth)
    return np.isfinite(d) & (d > 0)


def backproject_many(camera: Camera, u, v, depth) -> np.ndarray:
    """Camera-space points (N,3) for pixels (u,v) with camera-space depth Z."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    z = np.asarray(depth, dtype=np.float64)
    y = (v - camera.cy) * z / camera.fy
    x = ((u - camera.cx) * z - camera.skew * y) / camera.fx
    return np.stack([x, y, z], axis=-1)


def backproject(camera: Camera, pixel: tuple[float, float], depth: float) -> np.ndarray:
    if not np.isfinite(depth) or depth <= 0:
        raise GeometryError(f"cannot backproject non-positive or non-finite depth {depth}")
    return backproject_many(camera, [pixel[0]], [pixel[1]], [depth])[0]


@dataclass
class ProjectedPoint:
    valid: bool
    u: float
    v: float
    depth: float
