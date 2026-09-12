"""Stage 1 (OpenCV) cameras -> TMIV/MIV camera parameters, and back.

TMIV's camera frame is x forward, y left, z up: its perspective projection is
`u = cx - fx*y/x`, `v = cy - fy*z/x`, and depth is the x component
(TMIV `Renderer/Projector.h`). Its pose is camera->world: a position plus an
orientation given as `Rotation = [yaw, pitch, roll]` degrees with
`R_c2w = Rz(yaw) @ Ry(pitch) @ Rx(roll)` (`Common/Quaternion.h: euler2quat`,
and `Renderer/AffineTransform.cpp` confirms the direction).

Stage 1 uses OpenCV axes (x right, y down, z forward) with `R` world->camera
and `C` the camera centre. With `P` mapping OpenCV axes to TMIV axes, the same
physical world is expressed in TMIV coordinates as `x_tmiv = P @ x_opencv`, so::

    Position = P @ C
    R_c2w    = P @ R.T @ P.T

TMIV depth (camera x) then equals Stage 1 depth (camera z) exactly.
"""

from __future__ import annotations

import numpy as np

OPENCV_TO_TMIV = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
)


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def to_tmiv_pose(R: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV (R world->camera, C centre) -> TMIV (Position, Rotation[yaw,pitch,roll] deg)."""
    P = OPENCV_TO_TMIV
    position = P @ np.asarray(C, dtype=np.float64)
    r_c2w = P @ np.asarray(R, dtype=np.float64).T @ P.T
    yaw = np.arctan2(r_c2w[1, 0], r_c2w[0, 0])
    pitch = np.arcsin(np.clip(-r_c2w[2, 0], -1.0, 1.0))
    roll = np.arctan2(r_c2w[2, 1], r_c2w[2, 2])
    return position, np.degrees(np.array([yaw, pitch, roll]))


def from_tmiv_pose(
    position: np.ndarray, rotation_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `to_tmiv_pose`: returns (R world->camera, C) as float32, OpenCV axes."""
    P = OPENCV_TO_TMIV
    yaw, pitch, roll = np.radians(np.asarray(rotation_deg, dtype=np.float64))
    r_c2w = _rz(yaw) @ _ry(pitch) @ _rx(roll)
    R = P.T @ r_c2w.T @ P
    C = P.T @ np.asarray(position, dtype=np.float64)
    return R.astype(np.float32), C.astype(np.float32)


def tmiv_view_json(
    name: str,
    K: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    width: int,
    height: int,
    depth_range: tuple[float, float],
    bit_depth_color: int = 10,
    bit_depth_depth: int = 16,
) -> dict:
    """One camera entry of a TMIV sequence configuration ("cameras" list item)."""
    position, rotation = to_tmiv_pose(R, C)
    return {
        "Name": name,
        "Projection": "Perspective",
        "Resolution": [int(width), int(height)],
        "Focal": [float(K[0, 0]), float(K[1, 1])],
        "Principle_point": [float(K[0, 2]), float(K[1, 2])],
        "Position": [float(x) for x in position],
        "Rotation": [float(x) for x in rotation],
        "Depth_range": [float(depth_range[0]), float(depth_range[1])],
        "HasInvalidDepth": True,
        "BitDepthColor": int(bit_depth_color),
        "BitDepthDepth": int(bit_depth_depth),
        "ColorSpace": "YUV420",
        "DepthColorSpace": "YUV420",
        "Depthmap": 1,
        "Background": 0,
    }


def camera_from_tmiv_json(camera: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(K, R, C) in Stage 1 conventions from a TMIV camera entry."""
    fx, fy = camera["Focal"]
    cx, cy = camera["Principle_point"]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    R, C = from_tmiv_pose(np.asarray(camera["Position"]), np.asarray(camera["Rotation"]))
    return K, R, C
