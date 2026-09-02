"""Validates a SpatialPhotoResult against the Stage 1 spec's checks: file
shapes/metadata, camera rotation validity, depth sanity, and a reprojection
consistency test between two views.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import hdf5_io
from .hdf5_io import SpatialPhotoResult

REQUIRED_METADATA_KEYS = (
    "format_version", "model_name", "model_version",
    "source_width", "source_height", "output_width", "output_height",
    "num_views", "view_layout", "horizontal_angle", "vertical_angle",
    "coordinate_system", "pose_convention", "depth_definition", "depth_unit",
)

ROTATION_TOLERANCE = 1e-4
REPROJECTION_MAX_MEAN_ABS_ERROR = 40.0
REPROJECTION_MIN_COMPARED_PIXELS = 25


class ValidationError(Exception):
    """Raised by `validate_in_memory` when a SpatialPhotoResult fails validation."""


@dataclass
class ValidationReport:
    """All validation failures found, if any."""

    ok: bool
    failures: list[str] = field(default_factory=list)


def _check_shapes(result: SpatialPhotoResult) -> list[str]:
    failures = []
    if result.rgb.ndim != 4 or result.rgb.shape[-1] != 3:
        failures.append(f"rgb must have shape [N,H,W,3], got {result.rgb.shape}")
        return failures  # can't infer N reliably from a malformed rgb array

    num_views = result.rgb.shape[0]
    for name, arr in (("depth", result.depth), ("mask", result.mask)):
        if arr.ndim != 3 or arr.shape[0] != num_views:
            failures.append(f"{name} must have shape [{num_views},H,W], got {arr.shape}")
    for name, arr in (("K", result.K), ("R", result.R)):
        if arr.shape != (num_views, 3, 3):
            failures.append(f"{name} must have shape [{num_views},3,3], got {arr.shape}")
    if result.C.shape != (num_views, 3):
        failures.append(f"C must have shape [{num_views},3], got {result.C.shape}")
    return failures


def _check_metadata(metadata: dict) -> list[str]:
    return [
        f"missing required metadata key: {key}"
        for key in REQUIRED_METADATA_KEYS
        if key not in metadata
    ]


def _check_camera_rotations(R: np.ndarray) -> list[str]:
    failures = []
    identity = np.eye(3)
    for i, r_i in enumerate(R):
        orth_error = float(np.max(np.abs(r_i.T @ r_i - identity)))
        det = float(np.linalg.det(r_i))
        if orth_error >= ROTATION_TOLERANCE:
            failures.append(f"view {i}: R^T R deviates from identity by {orth_error:.2e}")
        if abs(det - 1.0) >= ROTATION_TOLERANCE:
            failures.append(f"view {i}: det(R)={det:.6f}, expected ~1.0")
    return failures


def _check_depth(depth: np.ndarray, mask: np.ndarray) -> list[str]:
    if not np.all(np.isfinite(depth)):
        return ["depth contains NaN or Inf values"]

    failures = []
    valid = mask == 1
    if valid.any() and np.any(depth[valid] <= 0):
        failures.append("depth has non-positive values at pixels marked valid by the mask")
    invalid = mask == 0
    if invalid.any() and np.any(depth[invalid] != 0.0):
        failures.append("depth is non-zero at pixels marked invalid by the mask")
    return failures


def _reproject(
    depth_src: np.ndarray, K_src: np.ndarray, R_src: np.ndarray, C_src: np.ndarray,
    K_dst: np.ndarray, R_dst: np.ndarray, C_dst: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproject every pixel of the source view into the destination camera.

    Returns (u_dst, v_dst, in_front), each shape (H, W).
    """
    height, width = depth_src.shape
    v_grid, u_grid = np.mgrid[0:height, 0:width].astype(np.float32)

    fx, fy, cx, cy = K_src[0, 0], K_src[1, 1], K_src[0, 2], K_src[1, 2]
    x_cam = (u_grid - cx) / fx * depth_src
    y_cam = (v_grid - cy) / fy * depth_src
    z_cam = depth_src

    points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    points_world = points_cam @ R_src + C_src  # == R_src.T @ points_cam + C_src

    points_dst_cam = (points_world - C_dst) @ R_dst.T  # == R_dst @ (points_world - C_dst)
    in_front = points_dst_cam[..., 2] > 1e-6

    fx_d, fy_d, cx_d, cy_d = K_dst[0, 0], K_dst[1, 1], K_dst[0, 2], K_dst[1, 2]
    z_safe = np.where(in_front, points_dst_cam[..., 2], 1.0)
    u_dst = fx_d * points_dst_cam[..., 0] / z_safe + cx_d
    v_dst = fy_d * points_dst_cam[..., 1] / z_safe + cy_d
    return u_dst, v_dst, in_front


def _check_reprojection(result: SpatialPhotoResult, src_idx: int, dst_idx: int) -> list[str]:
    height, width = result.depth.shape[1:3]
    u_dst, v_dst, in_front = _reproject(
        result.depth[src_idx], result.K[src_idx], result.R[src_idx], result.C[src_idx],
        result.K[dst_idx], result.R[dst_idx], result.C[dst_idx],
    )
    u_round = np.round(u_dst).astype(np.int64)
    v_round = np.round(v_dst).astype(np.int64)
    in_bounds = (u_round >= 0) & (u_round < width) & (v_round >= 0) & (v_round < height)

    comparable = (result.mask[src_idx] == 1) & in_front & in_bounds
    if comparable.sum() < REPROJECTION_MIN_COMPARED_PIXELS:
        return [
            f"reprojection view {src_idx}->{dst_idx}: only {int(comparable.sum())} pixels had "
            f"an in-bounds, in-front, valid correspondence "
            f"(need >= {REPROJECTION_MIN_COMPARED_PIXELS})"
        ]

    u_hit, v_hit = u_round[comparable], v_round[comparable]
    dst_valid = result.mask[dst_idx][v_hit, u_hit] == 1
    if not dst_valid.any():
        return [f"reprojection view {src_idx}->{dst_idx}: no comparable pixels land on valid target pixels"]

    src_colors = result.rgb[src_idx][comparable][dst_valid].astype(np.float32)
    dst_colors = result.rgb[dst_idx][v_hit[dst_valid], u_hit[dst_valid]].astype(np.float32)
    mean_abs_error = float(np.mean(np.abs(src_colors - dst_colors)))
    if mean_abs_error > REPROJECTION_MAX_MEAN_ABS_ERROR:
        return [
            f"reprojection view {src_idx}->{dst_idx}: mean abs color error {mean_abs_error:.1f} "
            f"exceeds threshold {REPROJECTION_MAX_MEAN_ABS_ERROR}"
        ]
    return []


def _validate(result: SpatialPhotoResult) -> ValidationReport:
    failures: list[str] = []
    shape_failures = _check_shapes(result)
    failures += shape_failures
    failures += _check_metadata(result.metadata)

    if not shape_failures:
        failures += _check_camera_rotations(result.R)
        failures += _check_depth(result.depth, result.mask)
        num_views = result.rgb.shape[0]
        if num_views >= 6:
            failures += _check_reprojection(result, 3, 5)

    return ValidationReport(ok=len(failures) == 0, failures=failures)


def validate_in_memory(result: SpatialPhotoResult) -> None:
    """Validate `result`; raise ValidationError listing every failure found."""
    report = _validate(result)
    if not report.ok:
        raise ValidationError("; ".join(report.failures))


def validate_file(path: str | Path) -> ValidationReport:
    """Load `path` and validate it, without raising."""
    result = hdf5_io.load(path)
    return _validate(result)
