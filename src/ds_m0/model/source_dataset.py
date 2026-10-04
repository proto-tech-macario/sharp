"""Source (9-view RGBD) data model. Deliberately separate from the DS Asset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import InputError
from ..geometry import camera_math
from .camera import Camera


@dataclass
class SourceView:
    view_id: int
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32
    depth_valid: np.ndarray  # (H, W) bool
    camera: Camera

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])


@dataclass
class SourceDataset:
    width: int
    height: int
    views: list[SourceView]
    depth_unit: str = "unspecified"

    def view_count(self) -> int:
        return len(self.views)

    def view(self, view_id: int) -> SourceView:
        for v in self.views:
            if v.view_id == view_id:
                return v
        raise InputError(f"source view {view_id} is missing")

    def base_view(self, view_id: int) -> SourceView:
        return self.view(view_id)

    def validate(self, expected_view_count: int | None = 9) -> None:
        if expected_view_count is not None and self.view_count() != expected_view_count:
            raise InputError(f"expected {expected_view_count} views, got {self.view_count()}")
        if self.view_count() == 0:
            raise InputError("dataset has no views")
        ids = [v.view_id for v in self.views]
        if len(set(ids)) != len(ids):
            raise InputError(f"duplicate view ids: {ids}")
        for v in self.views:
            if v.rgb.ndim != 3 or v.rgb.shape[2] != 3 or v.rgb.dtype != np.uint8:
                raise InputError(f"view {v.view_id}: rgb must be uint8 (H,W,3)")
            if (v.width, v.height) != (self.width, self.height):
                raise InputError(f"view {v.view_id}: {v.width}x{v.height} != {self.width}x{self.height}")
            if v.depth.shape != (self.height, self.width) or v.depth_valid.shape != v.depth.shape:
                raise InputError(f"view {v.view_id}: RGB and depth dimensions do not match")
            if v.camera is None:
                raise InputError(f"view {v.view_id}: camera parameters are missing")
            try:
                camera_math.validate(v.camera)
            except Exception as exc:  # noqa: BLE001 -- re-raised as an input error
                raise InputError(f"view {v.view_id}: invalid camera: {exc}") from exc
            if not np.isfinite(v.camera.translation).all():
                raise InputError(f"view {v.view_id}: non-finite camera parameters")
        if not any(v.depth_valid.any() for v in self.views):
            raise InputError("depth contains no usable samples")
