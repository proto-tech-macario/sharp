"""DS Presenter: DS Asset + target camera -> RGB + coverage (forward point splat).

Depends only on the model and geometry packages -- never on the creator, the
file I/O stack, the sparse codec, the source dataset, 3DGS or MIV.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..config.m0_config import PresenterConfig
from ..errors import ConfigurationError
from ..geometry import camera_math, projection, transformation
from ..model.asset import DSAsset
from ..model.camera import Camera
from .depth_buffer import DepthBuffer

log = logging.getLogger("ds_m0.presenter")


@dataclass
class RenderResult:
    rgb: np.ndarray  # (H, W, 3) uint8; undefined (zero) where coverage == 0
    coverage: np.ndarray  # (H, W) bool
    depth: np.ndarray | None = None  # optional debug output, float32, inf where uncovered


def render(asset: DSAsset, target_camera: Camera, config: PresenterConfig | None = None) -> RenderResult:
    config = config or PresenterConfig()
    config.validate()
    asset.validate()  # structurally invalid geometry must never render silently
    camera_math.validate(target_camera)
    width = config.output_width or target_camera.width
    height = config.output_height or target_camera.height
    if width <= 0 or height <= 0:
        raise ConfigurationError("output dimensions must be positive")

    xs, ys, zs, lids, orders, colors = [], [], [], [], [], []
    for layer_id, layer in enumerate(asset.layers):
        py, px = np.nonzero(layer.valid)  # raster order == source_order
        if len(px) == 0:
            continue
        z = layer.depth[py, px].astype(np.float64)
        p_base = camera_math.backproject_many(asset.base_camera, px, py, z)
        p_world = transformation.camera_to_world_many(asset.base_camera, p_base)
        p_tgt = transformation.world_to_camera_many(target_camera, p_world)
        u, v, z_t, ok = projection.project_many(target_camera, p_tgt)
        tx, ty, inside = projection.to_pixels(
            np.where(ok, u, -1.0), np.where(ok, v, -1.0), width, height, config.rounding_mode
        )
        keep = ok & inside
        xs.append(tx[keep]); ys.append(ty[keep]); zs.append(z_t[keep])
        lids.append(np.full(int(keep.sum()), layer_id, np.int64))
        orders.append((py * asset.spatial_width + px)[keep].astype(np.int64))
        colors.append(layer.rgb[py[keep], px[keep]])

    rgb = np.zeros((height, width, 3), np.uint8)
    coverage = np.zeros((height, width), bool)
    depth_out = np.full((height, width), np.inf, np.float32)
    if xs:
        x, y, z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
        buf = DepthBuffer(width, height, config.depth_epsilon)
        win = buf.update_many(x, y, z, np.concatenate(lids), np.concatenate(orders))
        color = np.concatenate(colors)
        rgb[y[win], x[win]] = color[win]
        coverage[y[win], x[win]] = True
        depth_out[y[win], x[win]] = z[win].astype(np.float32)
    log.info("rendered %dx%d, covered %d px", width, height, int(coverage.sum()))
    return RenderResult(rgb, coverage if config.enable_coverage_output else np.zeros_like(coverage), depth_out)
