"""World <-> camera transforms for x_cam = R x_world + t."""

from __future__ import annotations

import numpy as np

from ..model.camera import Camera


def camera_to_world_many(camera: Camera, p_camera: np.ndarray) -> np.ndarray:
    """P_world = R^T (P_camera - t), for row-vector points."""
    return (np.asarray(p_camera, np.float64) - camera.translation) @ camera.rotation


def world_to_camera_many(camera: Camera, p_world: np.ndarray) -> np.ndarray:
    return np.asarray(p_world, np.float64) @ camera.rotation.T + camera.translation


def camera_to_world(camera: Camera, p_camera) -> np.ndarray:
    return camera_to_world_many(camera, np.asarray(p_camera, np.float64).reshape(1, 3))[0]


def world_to_camera(camera: Camera, p_world) -> np.ndarray:
    return world_to_camera_many(camera, np.asarray(p_world, np.float64).reshape(1, 3))[0]


def translated_camera(camera: Camera, offset_in_camera_axes) -> Camera:
    """Same orientation/intrinsics, centre moved by `offset` expressed in the camera's own axes."""
    from .camera_math import camera_center

    offset = np.asarray(offset_in_camera_axes, np.float64).reshape(3)
    new_center = camera_center(camera) + camera.rotation.T @ offset
    return Camera(
        camera.projection_type, camera.width, camera.height, camera.fx, camera.fy,
        camera.cx, camera.cy, camera.skew, camera.rotation.copy(),
        -camera.rotation @ new_center, camera.coordinate_convention,
    )
