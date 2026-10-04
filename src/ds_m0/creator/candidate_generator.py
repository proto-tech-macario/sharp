"""Layer-1 candidate generation (spec Part II sections 10-13)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import camera_math, projection, transformation, visibility
from ..model.camera import Camera
from ..model.layer import Layer
from ..model.source_dataset import SourceView


@dataclass
class CandidateGenerationConfig:
    relative_depth_threshold: float = 0.02
    absolute_depth_threshold: float = 0.0
    rounding_mode: str = "half_up"


@dataclass
class Layer1Candidates:
    """Struct-of-arrays form of a list of Layer1Candidate. Internal to the creator."""

    target_x: np.ndarray  # int64
    target_y: np.ndarray  # int64
    depth_base: np.ndarray  # float64
    rgb: np.ndarray  # (N, 3) uint8
    source_view_id: np.ndarray  # int64
    source_view_angle: np.ndarray  # float64
    source_index: np.ndarray  # int64 raster index in the source view (final, unique tie-break)

    def __len__(self) -> int:
        return len(self.target_x)

    @staticmethod
    def empty() -> "Layer1Candidates":
        z = np.empty(0, np.int64)
        return Layer1Candidates(z, z, np.empty(0), np.empty((0, 3), np.uint8), z, np.empty(0), z)

    @staticmethod
    def concat(items: list["Layer1Candidates"]) -> "Layer1Candidates":
        items = [c for c in items if len(c)]
        if not items:
            return Layer1Candidates.empty()
        return Layer1Candidates(*(np.concatenate([getattr(c, f) for c in items]) for f in (
            "target_x", "target_y", "depth_base", "rgb", "source_view_id", "source_view_angle", "source_index",
        )))


def generate_candidates(
    source_view: SourceView, layer0: Layer, base_camera: Camera, config: CandidateGenerationConfig
) -> Layer1Candidates:
    """Hidden-surface candidates from one non-base view. Does not resolve conflicts."""
    valid = source_view.depth_valid & camera_math.valid_depth(source_view.depth)
    ys, xs = np.nonzero(valid)  # raster order
    if len(xs) == 0:
        return Layer1Candidates.empty()
    z_src = source_view.depth[ys, xs].astype(np.float64)
    p_src = camera_math.backproject_many(source_view.camera, xs, ys, z_src)
    p_world = transformation.camera_to_world_many(source_view.camera, p_src)
    p_base = transformation.world_to_camera_many(base_camera, p_world)
    u, v, z_base, ok = projection.project_many(base_camera, p_base)
    safe_u = np.where(ok, u, -1.0)
    safe_v = np.where(ok, v, -1.0)
    tx, ty, inside = projection.to_pixels(safe_u, safe_v, layer0.width, layer0.height, config.rounding_mode)
    keep = ok & inside
    keep[keep] &= layer0.valid[ty[keep], tx[keep]]
    z0 = np.zeros(len(xs))
    z0[keep] = layer0.depth[ty[keep], tx[keep]].astype(np.float64)
    keep &= visibility.is_hidden(z_base, z0, config.relative_depth_threshold, config.absolute_depth_threshold)
    n = int(keep.sum())
    angle = camera_math.view_angle(base_camera, source_view.camera)
    return Layer1Candidates(
        target_x=tx[keep],
        target_y=ty[keep],
        depth_base=z_base[keep],
        rgb=source_view.rgb[ys[keep], xs[keep]],
        source_view_id=np.full(n, source_view.view_id, np.int64),
        source_view_angle=np.full(n, angle, np.float64),
        source_index=(ys[keep] * source_view.width + xs[keep]).astype(np.int64),
    )
