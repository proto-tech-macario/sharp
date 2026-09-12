"""The spatial-sequence contract shared by Stage 2 components (spec §16-§21).

A `SpatialFrame` is one timestamp plus exactly 9 `SpatialView`s, each carrying
RGB, depth, validity mask, and K/R/C in Stage 1's conventions: OpenCV axes,
`x_cam = R @ (x_world - C)`, depth = camera-space Z in metres, 0.0 where the
mask is 0. This module depends on numpy only, so the MIV encoder can consume
it without SHARP or 3DGS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sharp_spatialize.hdf5_io import SpatialPhotoResult

NUM_VIEWS = 9
FORMAT_VERSION = "2.0"
CAMERA_CONVENTION = (
    "R is world->camera; C is the camera's world position; x_cam = R @ (x_world - C)"
)


@dataclass
class SpatialView:
    """One view of a spatial frame, pixel-aligned RGB/depth/mask plus its camera."""

    rgb: np.ndarray  # [H, W, 3] uint8
    depth: np.ndarray  # [H, W] float32, metres, 0.0 where invalid
    mask: np.ndarray  # [H, W] uint8, 1 = valid
    K: np.ndarray  # [3, 3] float32
    R: np.ndarray  # [3, 3] float32, world -> camera
    C: np.ndarray  # [3] float32, camera position in world


@dataclass
class SpatialFrame:
    """The 9 views generated from one input video frame, sharing one timestamp."""

    timestamp: float  # seconds, from the source presentation timestamp
    views: list[SpatialView]
    pts: int | None = None  # source PTS in the stream's time base
    source_frame_index: int | None = None  # index in presentation order in the source
    metadata: dict[str, Any] = field(default_factory=dict)  # Stage 1 per-frame metadata

    def __post_init__(self) -> None:
        if len(self.views) != NUM_VIEWS:
            raise ValueError(f"a spatial frame must have exactly 9 views, got {len(self.views)}")

    @property
    def height(self) -> int:
        """View height in pixels."""
        return int(self.views[0].rgb.shape[0])

    @property
    def width(self) -> int:
        """View width in pixels."""
        return int(self.views[0].rgb.shape[1])

    def stacked(self) -> dict[str, np.ndarray]:
        """The views as Stage 1-shaped arrays: rgb [9,H,W,3], depth [9,H,W], ..."""
        return {
            name: np.stack([getattr(view, name) for view in self.views])
            for name in ("rgb", "depth", "mask", "K", "R", "C")
        }

    @classmethod
    def from_stage1(
        cls,
        result: SpatialPhotoResult,
        timestamp: float,
        pts: int | None = None,
        source_frame_index: int | None = None,
    ) -> SpatialFrame:
        """Wrap one Stage 1 `SpatialPhotoResult` as a timestamped spatial frame."""
        views = [
            SpatialView(
                rgb=result.rgb[v], depth=result.depth[v], mask=result.mask[v],
                K=result.K[v], R=result.R[v], C=result.C[v],
            )
            for v in range(result.rgb.shape[0])
        ]
        return cls(
            timestamp=float(timestamp), views=views, pts=pts,
            source_frame_index=source_frame_index, metadata=dict(result.metadata),
        )

    def to_stage1(self) -> SpatialPhotoResult:
        """This frame as a Stage 1 `SpatialPhotoResult` (for Stage 1 validation/saving)."""
        from sharp_spatialize.hdf5_io import SpatialPhotoResult

        return SpatialPhotoResult(**self.stacked(), metadata=dict(self.metadata))


@dataclass
class SequenceInfo:
    """Sequence-level metadata of a temporal spatial sequence (spec §15 /metadata)."""

    width: int
    height: int
    fps: float
    view_count: int = NUM_VIEWS
    time_base: str | None = None
    source_filename: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_attrs(self, frame_count: int = 0) -> dict[str, Any]:
        """HDF5 attributes for the /metadata group."""
        attrs: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "width": int(self.width),
            "height": int(self.height),
            "fps": float(self.fps),
            "frame_count": int(frame_count),
            "view_count": int(self.view_count),
            "view_layout": "3x3",
            "depth_unit": "meter",
            "depth_definition": "camera_z",
            "invalid_depth_value": 0.0,
            "coordinate_system": "OpenCV",
            "camera_convention": CAMERA_CONVENTION,
            "time_base": self.time_base or "",
            "source_filename": self.source_filename,
        }
        attrs.update(self.extra)
        return attrs

    @classmethod
    def from_attrs(cls, attrs: dict[str, Any]) -> SequenceInfo:
        """Inverse of `to_attrs` (frame_count and fixed convention fields are dropped)."""
        known = {
            "format_version", "width", "height", "fps", "frame_count", "view_count",
            "view_layout", "depth_unit", "depth_definition", "invalid_depth_value",
            "coordinate_system", "camera_convention", "time_base", "source_filename",
        }
        return cls(
            width=int(attrs["width"]),
            height=int(attrs["height"]),
            fps=float(attrs["fps"]),
            view_count=int(attrs["view_count"]),
            time_base=str(attrs.get("time_base", "")) or None,
            source_filename=str(attrs.get("source_filename", "")),
            extra={k: v for k, v in attrs.items() if k not in known},
        )
