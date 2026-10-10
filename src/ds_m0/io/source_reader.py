"""Source dataset reader: existing spatial_photo.h5 (9-view RGBD) -> SourceDataset.

Hides the physical layout and normalises every camera into the internal
convention x_cam = R x_world + t. Uses h5py directly so that neither SHARP,
torch nor the sharp_spatialize package is needed.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from ..errors import InputError
from ..geometry import camera_math
from ..model.camera import Camera
from ..model.source_dataset import SourceDataset, SourceView


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def normalize_source_camera(K: np.ndarray, R: np.ndarray, C: np.ndarray, width: int, height: int) -> Camera:
    """Upstream (K, R, C) with x_cam = R (x_world - C)  ->  internal (R, t = -R C)."""
    if K.shape != (3, 3) or R.shape != (3, 3) or C.shape != (3,):
        raise InputError("camera arrays must be K(3,3), R(3,3), C(3,)")
    if not (np.isfinite(K).all() and np.isfinite(R).all() and np.isfinite(C).all()):
        raise InputError("camera parameters are not finite")
    return Camera.from_K_R_C(K, R, C, width, height)


def read_source_dataset(path: str | Path) -> SourceDataset:
    path = Path(path)
    if not path.is_file():
        raise InputError(f"source dataset not found: {path}")
    try:
        with h5py.File(path, "r") as f:
            for key in ("rgb", "depth", "camera/K", "camera/R", "camera/C"):
                if key not in f:
                    raise InputError(f"source dataset is missing '{key}'")
            rgb, depth = f["rgb"][:], f["depth"][:].astype(np.float32)
            mask = f["mask"][:] if "mask" in f else np.ones(depth.shape, np.uint8)
            K, R, C = f["camera/K"][:], f["camera/R"][:], f["camera/C"][:]
            attrs = {k: v for k, v in f.attrs.items()}
    except OSError as exc:
        raise InputError(f"cannot open source dataset {path}: {exc}") from exc

    if _text(attrs.get("coordinate_system", "OpenCV")).lower() != "opencv":
        raise InputError(f"camera convention {attrs['coordinate_system']!r} cannot be converted")
    if _text(attrs.get("depth_definition", "camera_z")) != "camera_z":
        raise InputError(f"depth definition {attrs['depth_definition']!r} is not camera-space Z")
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise InputError(f"rgb must be [N,H,W,3], got {rgb.shape}")
    n, h, w = rgb.shape[:3]
    if depth.shape != (n, h, w) or mask.shape != (n, h, w):
        raise InputError(f"RGB and depth dimensions do not match: {rgb.shape} vs {depth.shape}")
    if K.shape != (n, 3, 3) or R.shape != (n, 3, 3) or C.shape != (n, 3):
        raise InputError("camera arrays do not match the number of views")

    views = []
    for i in range(n):
        cam = normalize_source_camera(K[i], R[i], C[i], w, h)
        valid = (mask[i] > 0) & camera_math.valid_depth(depth[i])
        views.append(SourceView(i, np.ascontiguousarray(rgb[i], np.uint8), depth[i], valid, cam))
    return SourceDataset(w, h, views, _text(attrs.get("depth_unit", "unspecified")))
