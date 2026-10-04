"""Hand-made cameras/assets shared by the ds_m0 tests."""

from __future__ import annotations

import numpy as np
from ds_m0.model import Camera, DSAsset, Layer


def make_camera(width=8, height=8, f=8.0, R=None, C=(0.0, 0.0, 0.0), skew=0.0, fy=None, cx=None, cy=None) -> Camera:
    K = np.array([[f, skew, width / 2.0 if cx is None else cx],
                  [0, f if fy is None else fy, height / 2.0 if cy is None else cy], [0, 0, 1.0]])
    return Camera.from_K_R_C(K, np.eye(3) if R is None else np.asarray(R, float), np.asarray(C, float), width, height)


def make_asset(width=8, height=8, depth0=2.0, layer1_points=(), camera=None) -> DSAsset:
    """Tiny hand-made asset: flat Layer 0; Layer 1 samples given as (x, y, depth, (r, g, b))."""
    cam = camera or make_camera(width, height)
    l0 = Layer.empty(width, height)
    l0.valid[:] = True
    l0.depth[:] = depth0
    l0.rgb[:] = (np.arange(width * height * 3) % 251).reshape(height, width, 3).astype(np.uint8)
    l1 = Layer.empty(width, height)
    for x, y, d, rgb in layer1_points:
        l1.valid[y, x] = True
        l1.depth[y, x] = d
        l1.rgb[y, x] = rgb
    return DSAsset("1.0", width, height, cam, [l0, l1], {"base_view_id": 4})
