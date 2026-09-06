"""Public Python API.

`generate_spatial_photo` ties inference, camera rig construction, rendering,
HDF5 packaging, and validation into one call.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import cameras, inference, render, validation
from .hdf5_io import SpatialPhotoResult, load

DEFAULT_ANGLE_DEG = 10.0

ProgressCallback = Callable[[str, float], None]


def _build_metadata(
    image_path: Path,
    scene: inference.SceneBundle,
    angle_deg: float,
    output_width: int,
    output_height: int,
) -> dict:
    return {
        "format_version": "1.0",
        "model_name": "SHARP",
        "model_version": scene.model_version,
        "source_width": scene.width,
        "source_height": scene.height,
        "source_filename": image_path.name,
        "output_width": output_width,
        "output_height": output_height,
        "num_views": 9,
        "view_layout": "3x3",
        "horizontal_angle": float(angle_deg),
        "vertical_angle": float(angle_deg),
        "coordinate_system": "OpenCV",
        "pose_convention": (
            "R is world->camera; C is the camera's world position; "
            "x_cam = R @ (x_world - C)"
        ),
        "depth_definition": "camera_z",
        "depth_unit": "meter",
    }


def generate_spatial_photo(
    image_path: str | Path,
    angle_deg: float = DEFAULT_ANGLE_DEG,
    output_width: int | None = None,
    output_height: int | None = None,
    checkpoint_path: str | Path | None = None,
    device: str = "default",
    precision: str = "fp32",
    cache_predictor: bool = False,
    on_progress: ProgressCallback | None = None,
) -> SpatialPhotoResult:
    """Convert one RGB image into a 9-view SpatialPhotoResult.

    Raises `validation.ValidationError` if the generated scene fails
    validation -- nothing is saved to disk in that case.

    `precision` and `cache_predictor` are passed through to `inference.infer`.
    `on_progress`, if given, is called as `on_progress(stage, fraction)` at each
    stage boundary, where `fraction` is in [0, 1]; it is for reporting only and
    must not raise.
    """
    report = on_progress if on_progress is not None else lambda stage, fraction: None

    image_path = Path(image_path)
    report("inference", 0.05)
    scene = inference.infer(
        image_path,
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        device=device,
        precision=precision,
        cache_predictor=cache_predictor,
    )

    width = output_width if output_width is not None else scene.width
    height = output_height if output_height is not None else scene.height

    report("rendering", 0.55)
    rig = cameras.build_camera_rig(
        scene.gaussians.mean_vectors, scene.f_px, scene.width, scene.height,
        angle_deg, width, height,
    )
    rgb, depth, mask = render.render_views(scene.gaussians, rig, width, height)

    result = SpatialPhotoResult(
        rgb=rgb, depth=depth, mask=mask,
        K=np.stack([pose.K for pose in rig]),
        R=np.stack([pose.R for pose in rig]),
        C=np.stack([pose.C for pose in rig]),
        metadata=_build_metadata(image_path, scene, angle_deg, width, height),
    )
    report("validating", 0.9)
    validation.validate_in_memory(result)
    report("done", 1.0)
    return result


def load_spatial_photo(path: str | Path) -> SpatialPhotoResult:
    """Load a previously saved spatial_photo.h5 file. Does not require SHARP or CUDA."""
    return load(path)
