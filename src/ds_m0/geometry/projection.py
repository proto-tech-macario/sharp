"""Projection and the single deterministic pixel-rounding rule."""

from __future__ import annotations

import numpy as np

from ..errors import ConfigurationError
from ..model.camera import Camera
from .camera_math import ProjectedPoint

ROUNDING_MODES = ("half_up", "half_even")


def round_to_pixel(x: np.ndarray, mode: str = "half_up") -> np.ndarray:
    """Platform-independent rounding. `half_up` = floor(x + 0.5) (reference rule)."""
    if mode == "half_up":
        return np.floor(np.asarray(x, np.float64) + 0.5).astype(np.int64)
    if mode == "half_even":
        return np.rint(np.asarray(x, np.float64)).astype(np.int64)
    raise ConfigurationError(f"unknown rounding_mode {mode!r}; expected one of {ROUNDING_MODES}")


def project_many(camera: Camera, p_camera: np.ndarray):
    """Returns (u, v, z, valid); `valid` means Z > 0 and finite results."""
    p = np.asarray(p_camera, np.float64)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (camera.fx * x + camera.skew * y) / z + camera.cx
        v = camera.fy * y / z + camera.cy
    valid = np.isfinite(z) & (z > 0) & np.isfinite(u) & np.isfinite(v)
    return u, v, z, valid


def project(camera: Camera, p_camera) -> ProjectedPoint:
    u, v, z, valid = project_many(camera, np.asarray(p_camera, np.float64).reshape(1, 3))
    return ProjectedPoint(bool(valid[0]), float(u[0]), float(v[0]), float(z[0]))


def to_pixels(u, v, width: int, height: int, mode: str = "half_up"):
    """Round continuous coordinates to pixels; returns (x, y, inside)."""
    x = round_to_pixel(u, mode)
    y = round_to_pixel(v, mode)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x, y, inside
