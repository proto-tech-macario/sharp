"""Deterministic synthetic 9-view RGBD scene (used for tests and demos).

Ray-casts a textured back wall plus two spheres through the same 3x3 camera
rig layout the real pipeline produces (base view 4 = identity pose). Only
IEEE-exact operations (+ - * / sqrt floor) are used so the scene is bit-stable
across platforms.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from ..model.camera import Camera
from ..model.source_dataset import SourceDataset, SourceView

LAYOUT = ((-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
WALL_Z = 6.0
WALL_HALF = 4.0
SPHERES = (((0.0, 0.0, 3.5), 1.0), ((-1.4, 0.6, 4.5), 0.5))
FOCUS = np.array([0.0, 0.0, 4.0])
LIGHT = np.array([-0.4, -0.6, -0.7]) / np.sqrt(0.16 + 0.36 + 0.49)


def _look_at(center: np.ndarray, target: np.ndarray) -> np.ndarray:
    z = target - center
    z = z / np.sqrt((z * z).sum())
    x = np.cross([0.0, 1.0, 0.0], z)
    x = x / np.sqrt((x * x).sum())
    y = np.cross(z, x)
    return np.stack([x, y, z])  # rows = camera axes in world; x_cam = R (x_world - C)


def rig(baseline: float = 0.35) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for hs, vs in LAYOUT:
        c = np.array([hs * baseline, vs * baseline, 0.0])
        out.append((_look_at(c, FOCUS), c))
    return out


def _render_view(K: np.ndarray, R: np.ndarray, C: np.ndarray, w: int, h: int):
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    d = d_cam @ R  # world direction with camera-z == 1, so ray parameter t == camera depth
    best = np.full((h, w), np.inf)
    rgb = np.zeros((h, w, 3), np.float64)

    t_wall = (WALL_Z - C[2]) / d[..., 2]
    p = C + t_wall[..., None] * d
    hit = (t_wall > 0) & (np.abs(p[..., 0]) < WALL_HALF) & (np.abs(p[..., 1]) < WALL_HALF)
    checker = (np.floor(p[..., 0] / 0.5) + np.floor(p[..., 1] / 0.5)) % 2
    wall_rgb = np.where(checker[..., None] == 0, [200.0, 200.0, 190.0], [60.0, 70.0, 110.0])
    best = np.where(hit, t_wall, best)
    rgb = np.where(hit[..., None], wall_rgb, rgb)

    for k, (center, radius) in enumerate(SPHERES):
        oc = C - np.asarray(center)
        a = (d * d).sum(-1)
        b = 2.0 * (d * oc).sum(-1)
        c0 = (oc * oc).sum() - radius * radius
        disc = b * b - 4.0 * a * c0
        ok = disc >= 0
        t = (-b - np.sqrt(np.where(ok, disc, 0.0))) / (2.0 * a)
        ok &= (t > 0) & (t < best)
        n = (C + t[..., None] * d - np.asarray(center)) / radius
        shade = 0.35 + 0.65 * np.clip((n * -LIGHT).sum(-1), 0, 1)
        base = np.array([220.0, 60.0, 50.0]) if k == 0 else np.array([60.0, 200.0, 90.0])
        best = np.where(ok, t, best)
        rgb = np.where(ok[..., None], base * shade[..., None], rgb)
    mask = np.isfinite(best)
    depth = np.where(mask, best, 0.0).astype(np.float32)
    return np.floor(rgb).clip(0, 255).astype(np.uint8), depth, mask.astype(np.uint8)


def make_synthetic_arrays(width: int = 96, height: int = 96, baseline: float = 0.35):
    f = 1.2 * width
    K = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])
    rgbs, depths, masks, Ks, Rs, Cs = [], [], [], [], [], []
    for R, C in rig(baseline):
        rgb, depth, mask = _render_view(K, R, C, width, height)
        rgbs.append(rgb); depths.append(depth); masks.append(mask)
        Ks.append(K.astype(np.float32)); Rs.append(R.astype(np.float32)); Cs.append(C.astype(np.float32))
    return (np.stack(rgbs), np.stack(depths), np.stack(masks), np.stack(Ks), np.stack(Rs), np.stack(Cs))


def make_synthetic_dataset(width: int = 96, height: int = 96, baseline: float = 0.35) -> SourceDataset:
    rgb, depth, mask, K, R, C = make_synthetic_arrays(width, height, baseline)
    views = [
        SourceView(i, rgb[i], depth[i], (mask[i] > 0) & (depth[i] > 0),
                   Camera.from_K_R_C(K[i], R[i], C[i], width, height))
        for i in range(9)
    ]
    return SourceDataset(width, height, views, "meter")


def write_synthetic_h5(path: str | Path, width: int = 96, height: int = 96, baseline: float = 0.35) -> Path:
    """Write the arrays in the same layout as sharp_spatialize's spatial_photo.h5."""
    rgb, depth, mask, K, R, C = make_synthetic_arrays(width, height, baseline)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("rgb", data=rgb, compression="gzip")
        f.create_dataset("depth", data=depth, compression="gzip")
        f.create_dataset("mask", data=mask, compression="gzip")
        cam = f.create_group("camera")
        cam.create_dataset("K", data=K); cam.create_dataset("R", data=R); cam.create_dataset("C", data=C)
        f.attrs.update({
            "format_version": "1.0", "num_views": 9, "view_layout": "3x3", "coordinate_system": "OpenCV",
            "pose_convention": "R is world->camera; C is the camera's world position; x_cam = R @ (x_world - C)",
            "depth_definition": "camera_z", "depth_unit": "meter",
            "output_width": width, "output_height": height, "source_filename": "ds_m0_synthetic",
        })
    return path
