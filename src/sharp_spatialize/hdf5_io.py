"""Reads and writes spatial_photo.h5 files.

No SHARP or torch import here on purpose: this is the module a downstream
consumer with no SHARP install would vendor/reference to read a spatial
photo, per the spec's requirement that the HDF5 file be usable without
SHARP or the 3DGS representation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass
class SpatialPhotoResult:
    """In-memory contents of one spatial_photo.h5 file."""

    rgb: np.ndarray  # [N, H, W, 3] uint8
    depth: np.ndarray  # [N, H, W] float32
    mask: np.ndarray  # [N, H, W] uint8
    K: np.ndarray  # [N, 3, 3] float32
    R: np.ndarray  # [N, 3, 3] float32
    C: np.ndarray  # [N, 3] float32
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Write this result to `path` as an HDF5 spatial photo file."""
        save(path, self)


def save(path: str | Path, result: SpatialPhotoResult) -> None:
    """Write `result` to `path`, atomically (via a temp file + rename).

    Creates `path`'s parent directory if it doesn't exist. If writing fails
    partway through, the partially-written temp file is deleted before the
    exception propagates -- no stale `.tmp` file is left behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with h5py.File(tmp_path, "w") as f:
            f.create_dataset("rgb", data=result.rgb.astype(np.uint8), compression="gzip")
            f.create_dataset("depth", data=result.depth.astype(np.float32), compression="gzip")
            f.create_dataset("mask", data=result.mask.astype(np.uint8), compression="gzip")
            camera_group = f.create_group("camera")
            camera_group.create_dataset("K", data=result.K.astype(np.float32))
            camera_group.create_dataset("R", data=result.R.astype(np.float32))
            camera_group.create_dataset("C", data=result.C.astype(np.float32))
            for key, value in result.metadata.items():
                f.attrs[key] = value
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    os.replace(tmp_path, path)


def load(path: str | Path) -> SpatialPhotoResult:
    """Read a spatial_photo.h5 file back into a SpatialPhotoResult."""
    with h5py.File(path, "r") as f:
        return SpatialPhotoResult(
            rgb=f["rgb"][:],
            depth=f["depth"][:],
            mask=f["mask"][:],
            K=f["camera/K"][:],
            R=f["camera/R"][:],
            C=f["camera/C"][:],
            metadata=dict(f.attrs),
        )
