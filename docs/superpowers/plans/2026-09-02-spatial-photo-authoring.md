# Stage 1 Spatial Photo Authoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `sharp_spatialize`, an installable package that converts one RGB image into a 9-view `spatial_photo.h5` (9 RGB views, 9 depth maps, 9 validity masks, per-view camera K/R/C), with a Python API, two CLIs, and an automated validation suite — implementing the Stage 1 Spatial Photo Authoring Specification v1.0.

**Architecture:** Six small, independently-testable modules under `src/sharp_spatialize/` (`cameras.py`, `hdf5_io.py`, `validation.py`, `inference.py`, `render.py`, `api.py`) plus a `cli.py`, wired together as: `inference.infer()` → `cameras.build_camera_rig()` → `render.render_views()` → `hdf5_io.save()`, with `validation.py` run automatically before any file is written. Only `render.py` requires a CUDA GPU (gsplat has no CPU/MPS kernel); every other module is fully testable on any machine.

**Tech Stack:** Python 3.13, PyTorch, `sharp.utils.camera`/`sharp.utils.gsplat`/`sharp.utils.gaussians` (this repo's existing SHARP code), `h5py`, Click, pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-spatial-photo-authoring-design.md` (and the source PDF it implements, "Stage 1 Spatial Photo Authoring Specification" v1.0)

## Global Constraints

- Exactly 9 views in a fixed 3×3 layout V0..V8: `V0=(-a,+a) V1=(0,+a) V2=(+a,+a) V3=(-a,0) V4=(0,0) V5=(+a,0) V6=(-a,-a) V7=(0,-a) V8=(+a,-a)`, where `a = angle_deg`. V4 is the reference camera.
- `angle_deg` is configurable; at minimum 5, 10, and 15 must work (any positive float is accepted).
- HDF5 datasets, exact shapes/dtypes: `/rgb` `[9,H,W,3]` uint8; `/depth` `[9,H,W]` float32; `/mask` `[9,H,W]` uint8 (0=invalid, 1=valid); `/camera/K` `[9,3,3]` float32; `/camera/R` `[9,3,3]` float32; `/camera/C` `[9,3]` float32.
- Required file-level HDF5 attrs (at least): `format_version, model_name, model_version, source_width, source_height, output_width, output_height, num_views, view_layout, horizontal_angle, vertical_angle, coordinate_system, pose_convention, depth_definition, depth_unit`.
- Coordinate convention: OpenCV (X right, Y down, Z forward). `R` is world→camera; `C` is the camera's world position; `x_cam = R @ (x_world - C)`. Depth is camera-space Z, unit = meter.
- Camera validity tolerance: `‖RᵀR − I‖_max < 1e-4` and `|det(R) − 1| < 1e-4` for every view.
- Rendering (`render.py`) requires CUDA — no CPU/MPS fallback. Every other module must work without CUDA.
- CLI: `sharp-spatialize --input image.jpg --angle 10 --output spatial_photo.h5 [--width W] [--height H] [-c checkpoint] [--device DEV]` and `sharp-spatialize-validate spatial_photo.h5`.
- Python API: `generate_spatial_photo(image_path, angle_deg=10) -> result`; `result.save("spatial_photo.h5")`; `result.rgb / .depth / .mask / .K / .R / .C`.
- The HDF5 file must be fully readable (`hdf5_io.load`) with zero SHARP/torch imports.

---

### Task 1: Camera rig geometry (`cameras.py`)

**Files:**
- Create: `src/sharp_spatialize/__init__.py` (empty for now — populated in Task 6)
- Create: `src/sharp_spatialize/cameras.py`
- Test: `tests/sharp_spatialize/test_cameras.py`

**Interfaces:**
- Produces: `CameraPose` dataclass (`K: np.ndarray[3,3] float32`, `R: np.ndarray[3,3] float32`, `C: np.ndarray[3] float32`), `DEFAULT_LAYOUT: tuple[tuple[float,float],...]` (9 `(h_sign, v_sign)` pairs, V0..V8 order), `focus_depth(mean_vectors: torch.Tensor) -> float`, `camera_at(mean_vectors, f_px, source_width, source_height, h_angle_deg, v_angle_deg, output_width, output_height) -> CameraPose`, `build_camera_rig(mean_vectors, f_px, source_width, source_height, angle_deg, output_width, output_height) -> list[CameraPose]` (always length 9, V0..V8 order).

- [ ] **Step 1: Install dev/runtime dependencies into the project venv**

Run:
```bash
/Users/macariofang/.local/bin/uv pip install --python .venv/bin/python pytest h5py
```
Expected: `Installed ... pytest h5py ...` (or "Already installed" on a re-run).

- [ ] **Step 2: Write the failing tests**

Create `tests/sharp_spatialize/test_cameras.py`:

```python
"""Tests for sharp_spatialize.cameras."""

from __future__ import annotations

import numpy as np
import torch

from sharp_spatialize.cameras import DEFAULT_LAYOUT, build_camera_rig, camera_at, focus_depth

PLANE_Z = 5.0


def _flat_mean_vectors() -> torch.Tensor:
    xs = torch.linspace(-3.0, 3.0, 20)
    ys = torch.linspace(-3.0, 3.0, 20)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    points = torch.stack(
        [grid_x.flatten(), grid_y.flatten(), torch.full_like(grid_x.flatten(), PLANE_Z)], dim=-1
    )
    return points.unsqueeze(0)


def test_focus_depth_is_near_the_flat_scenes_depth():
    depth = focus_depth(_flat_mean_vectors())
    assert depth == PLANE_Z


def test_focus_depth_raises_when_no_positive_depth_points():
    mean_vectors = torch.zeros(1, 5, 3)  # all z == 0, nothing strictly positive
    try:
        focus_depth(mean_vectors)
        raise AssertionError("expected ValueError for a scene with no positive-depth points")
    except ValueError:
        pass


def test_default_layout_matches_spec_v0_through_v8():
    assert DEFAULT_LAYOUT == (
        (-1.0, 1.0), (0.0, 1.0), (1.0, 1.0),
        (-1.0, 0.0), (0.0, 0.0), (1.0, 0.0),
        (-1.0, -1.0), (0.0, -1.0), (1.0, -1.0),
    )


def test_build_camera_rig_returns_exactly_nine_views():
    rig = build_camera_rig(_flat_mean_vectors(), 40.0, 32, 32, 10.0, 32, 32)
    assert len(rig) == 9


def test_v4_is_the_reference_camera():
    rig = build_camera_rig(_flat_mean_vectors(), 40.0, 32, 32, 10.0, 32, 32)
    v4 = rig[4]
    np.testing.assert_allclose(v4.R, np.eye(3, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(v4.C, np.zeros(3, dtype=np.float32), atol=1e-6)


def test_every_view_has_an_orthonormal_rotation_with_determinant_one():
    rig = build_camera_rig(_flat_mean_vectors(), 40.0, 32, 32, 10.0, 32, 32)
    for pose in rig:
        orth_error = np.max(np.abs(pose.R.T @ pose.R - np.eye(3)))
        assert orth_error < 1e-4
        assert abs(np.linalg.det(pose.R) - 1.0) < 1e-4


def test_camera_at_agrees_with_build_camera_rig_at_a_grid_angle():
    mean_vectors = _flat_mean_vectors()
    rig = build_camera_rig(mean_vectors, 40.0, 32, 32, 10.0, 32, 32)
    v0_direct = camera_at(mean_vectors, 40.0, 32, 32, -10.0, 10.0, 32, 32)
    np.testing.assert_allclose(v0_direct.R, rig[0].R)
    np.testing.assert_allclose(v0_direct.C, rig[0].C)
    np.testing.assert_allclose(v0_direct.K, rig[0].K)


def test_camera_at_supports_an_arbitrary_intermediate_angle():
    """Acceptance criterion #10: any angle within the configured range must render."""
    pose = camera_at(_flat_mean_vectors(), 40.0, 32, 32, 3.7, -4.2, 32, 32)
    orth_error = np.max(np.abs(pose.R.T @ pose.R - np.eye(3)))
    assert orth_error < 1e-4


def test_rescales_intrinsics_for_a_different_output_size():
    rig = build_camera_rig(_flat_mean_vectors(), 40.0, 32, 32, 10.0, 64, 16)
    v4 = rig[4]
    assert v4.K[0, 0] == 80.0  # fx doubled: output_width 64 / source_width 32
    assert v4.K[1, 1] == 20.0  # fy halved: output_height 16 / source_height 32
    assert v4.K[0, 2] == 32.0  # cx = output_width / 2
    assert v4.K[1, 2] == 8.0   # cy = output_height / 2
```

- [ ] **Step 3: Run the tests to verify they fail with ModuleNotFoundError**

Run: `.venv/bin/python -m pytest tests/sharp_spatialize/test_cameras.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sharp_spatialize'`

- [ ] **Step 4: Write `src/sharp_spatialize/__init__.py`**

```python
"""sharp_spatialize: converts one RGB image into a 9-view spatial photo."""
```

- [ ] **Step 5: Write `src/sharp_spatialize/cameras.py`**

```python
"""Builds the 3x3 virtual-camera rig for a SHARP scene.

Every camera orbits the same fixed focus point at the scene's focus depth and
looks back at it -- this produces real parallax/occlusion between views,
unlike a pure pan/tilt rotation in place. Geometry is expressed directly in
the reference camera's own coordinate frame (X right, Y down, Z forward, per
the OpenCV convention SHARP already uses), which doubles as "world" here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from sharp.utils.camera import create_camera_matrix

FOCUS_DEPTH_QUANTILE = 0.1

# (h_sign, v_sign) for V0..V8, per the spec's fixed 3x3 table:
#   V0=(-a,+a) V1=(0,+a) V2=(+a,+a)
#   V3=(-a, 0) V4=(0, 0) V5=(+a, 0)
#   V6=(-a,-a) V7=(0,-a) V8=(+a,-a)
DEFAULT_LAYOUT: tuple[tuple[float, float], ...] = (
    (-1.0, 1.0), (0.0, 1.0), (1.0, 1.0),
    (-1.0, 0.0), (0.0, 0.0), (1.0, 0.0),
    (-1.0, -1.0), (0.0, -1.0), (1.0, -1.0),
)


@dataclass
class CameraPose:
    """One virtual camera's parameters, OpenCV convention (world->camera)."""

    K: np.ndarray  # (3, 3) float32
    R: np.ndarray  # (3, 3) float32
    C: np.ndarray  # (3,) float32


def focus_depth(mean_vectors: torch.Tensor) -> float:
    """The reference-camera-space depth every view orbits around.

    Uses the 10th percentile of the scene's positive-depth points, same
    heuristic as `sharp.utils.camera.PinholeCameraModel`.
    """
    depths = mean_vectors.reshape(-1, 3)[:, 2]
    depths = depths[depths > 0]
    if depths.numel() == 0:
        raise ValueError("Scene has no positive-depth points; cannot compute a focus depth.")
    return float(torch.quantile(depths.float().cpu(), FOCUS_DEPTH_QUANTILE))


def camera_at(
    mean_vectors: torch.Tensor,
    f_px: float,
    source_width: int,
    source_height: int,
    h_angle_deg: float,
    v_angle_deg: float,
    output_width: int,
    output_height: int,
) -> CameraPose:
    """Build one camera pose orbiting the scene's focus point by the given angles.

    `(0, 0)` reduces to the reference camera's own pose (identity R, C at the
    origin). Any angle works, not just the 9 grid values -- this is what
    lets an arbitrary intermediate view be rendered for testing.
    """
    depth = focus_depth(mean_vectors)

    h_rad = np.radians(h_angle_deg)
    v_rad = np.radians(v_angle_deg)
    eye_pos = torch.tensor(
        [depth * np.tan(h_rad), depth * np.tan(v_rad), 0.0], dtype=torch.float32
    )
    look_at = torch.tensor([0.0, 0.0, depth], dtype=torch.float32)
    world_up = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32)

    extrinsics = create_camera_matrix(eye_pos, look_at, world_up, inverse=True)
    R = extrinsics[:3, :3].numpy().astype(np.float32)
    C = eye_pos.numpy().astype(np.float32)

    fx = f_px * (output_width / source_width)
    fy = f_px * (output_height / source_height)
    K = np.array(
        [[fx, 0.0, output_width / 2.0], [0.0, fy, output_height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return CameraPose(K=K, R=R, C=C)


def build_camera_rig(
    mean_vectors: torch.Tensor,
    f_px: float,
    source_width: int,
    source_height: int,
    angle_deg: float,
    output_width: int,
    output_height: int,
) -> list[CameraPose]:
    """Build the 9-camera rig V0..V8 for the given angle (applied to both axes)."""
    return [
        camera_at(
            mean_vectors, f_px, source_width, source_height,
            h_sign * angle_deg, v_sign * angle_deg, output_width, output_height,
        )
        for h_sign, v_sign in DEFAULT_LAYOUT
    ]
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/sharp_spatialize/test_cameras.py -v`
Expected: 9 passed.

- [ ] **Step 7: Commit**

```bash
git add src/sharp_spatialize/__init__.py src/sharp_spatialize/cameras.py tests/sharp_spatialize/test_cameras.py
git commit -m "sharp_spatialize: add camera rig geometry (cameras.py)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LDaau3BWz9CKHKLfpGVU2B"
```

---

### Task 2: HDF5 writer/reader (`hdf5_io.py`)

**Files:**
- Create: `src/sharp_spatialize/hdf5_io.py`
- Test: `tests/sharp_spatialize/test_hdf5_io.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `SpatialPhotoResult` dataclass (`rgb, depth, mask, K, R, C: np.ndarray`, `metadata: dict`, method `.save(path)`), `save(path, result: SpatialPhotoResult) -> None`, `load(path) -> SpatialPhotoResult`. No SHARP/torch import in this file.

- [ ] **Step 1: Add `h5py` to `pyproject.toml` and reinstall the package**

Edit `pyproject.toml`, in the `[project]` `dependencies` list, add `"h5py",` right after `"gsplat",`:

```toml
dependencies = [
  "click",
  "gsplat",
  "h5py",
  "imageio[ffmpeg]",
  "matplotlib",
  "pillow-heif",
  "plyfile",
  "scipy",
  "timm",
  "torch",
  "torchvision",
]
```

Run:
```bash
/Users/macariofang/.local/bin/uv pip install --python .venv/bin/python -e .
```
Expected: reinstalls `sharp` in editable mode and installs `h5py` (already present from Task 1's manual install, so this just confirms `pyproject.toml` now declares it correctly).

- [ ] **Step 2: Write the failing tests**

Create `tests/sharp_spatialize/test_hdf5_io.py`:

```python
"""Tests for sharp_spatialize.hdf5_io."""

from __future__ import annotations

import numpy as np

from sharp_spatialize.hdf5_io import SpatialPhotoResult, load, save


def _make_result(num_views: int = 9, height: int = 4, width: int = 6) -> SpatialPhotoResult:
    rng = np.random.default_rng(0)
    return SpatialPhotoResult(
        rgb=rng.integers(0, 255, size=(num_views, height, width, 3), dtype=np.uint8),
        depth=(rng.random((num_views, height, width)).astype(np.float32) + 0.1),
        mask=np.ones((num_views, height, width), dtype=np.uint8),
        K=np.tile(np.eye(3, dtype=np.float32), (num_views, 1, 1)),
        R=np.tile(np.eye(3, dtype=np.float32), (num_views, 1, 1)),
        C=np.zeros((num_views, 3), dtype=np.float32),
        metadata={"format_version": "1.0", "model_name": "SHARP", "num_views": num_views},
    )


def test_save_then_load_round_trips_all_datasets(tmp_path):
    result = _make_result()
    out_path = tmp_path / "spatial_photo.h5"

    save(out_path, result)
    loaded = load(out_path)

    np.testing.assert_array_equal(loaded.rgb, result.rgb)
    np.testing.assert_array_equal(loaded.depth, result.depth)
    np.testing.assert_array_equal(loaded.mask, result.mask)
    np.testing.assert_array_equal(loaded.K, result.K)
    np.testing.assert_array_equal(loaded.R, result.R)
    np.testing.assert_array_equal(loaded.C, result.C)
    assert loaded.metadata["format_version"] == "1.0"
    assert loaded.metadata["num_views"] == 9


def test_result_save_method_matches_module_level_save(tmp_path):
    result = _make_result()
    out_path = tmp_path / "spatial_photo.h5"

    result.save(out_path)

    assert out_path.exists()
    loaded = load(out_path)
    np.testing.assert_array_equal(loaded.rgb, result.rgb)


def test_save_does_not_leave_a_tmp_file_behind(tmp_path):
    result = _make_result()
    out_path = tmp_path / "spatial_photo.h5"

    save(out_path, result)

    assert list(tmp_path.iterdir()) == [out_path]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/sharp_spatialize/test_hdf5_io.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sharp_spatialize.hdf5_io'`

- [ ] **Step 4: Write `src/sharp_spatialize/hdf5_io.py`**

```python
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
    """Write `result` to `path`, atomically (via a temp file + rename)."""
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/sharp_spatialize/test_hdf5_io.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/sharp_spatialize/hdf5_io.py tests/sharp_spatialize/test_hdf5_io.py
git commit -m "sharp_spatialize: add HDF5 writer/reader (hdf5_io.py)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LDaau3BWz9CKHKLfpGVU2B"
```

---

### Task 3: Validation logic (`validation.py`) + shared test fixture

**Files:**
- Create: `src/sharp_spatialize/validation.py`
- Create: `tests/sharp_spatialize/conftest.py`
- Test: `tests/sharp_spatialize/test_validation.py`

**Interfaces:**
- Consumes: `sharp_spatialize.cameras.build_camera_rig` (Task 1, test fixture only), `sharp_spatialize.hdf5_io.SpatialPhotoResult` / `save` / `load` (Task 2).
- Produces: `ValidationError(Exception)`, `ValidationReport` dataclass (`ok: bool`, `failures: list[str]`), `validate_in_memory(result: SpatialPhotoResult) -> None` (raises `ValidationError`), `validate_file(path) -> ValidationReport` (never raises). Also, for later tasks' tests: the `consistent_spatial_photo_result` pytest fixture defined in `conftest.py`.

- [ ] **Step 1: Write the shared synthetic-scene test fixture**

Create `tests/sharp_spatialize/conftest.py`:

```python
"""Shared pytest fixtures for sharp_spatialize tests.

`consistent_spatial_photo_result` builds a small synthetic scene (a flat
checkerboard at world z=PLANE_Z) and ray-casts it through the *real* camera
rig from `cameras.py`, so tests get geometrically self-consistent K/R/C/
depth/rgb data without needing SHARP or a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sharp_spatialize.cameras import build_camera_rig
from sharp_spatialize.hdf5_io import SpatialPhotoResult

PLANE_Z = 5.0
PLANE_HALF_EXTENT = 1.8
OUTPUT_SIZE = 32


def _synthetic_mean_vectors() -> torch.Tensor:
    xs = torch.linspace(-3.0, 3.0, 20)
    ys = torch.linspace(-3.0, 3.0, 20)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    points = torch.stack(
        [grid_x.flatten(), grid_y.flatten(), torch.full_like(grid_x.flatten(), PLANE_Z)], dim=-1
    )
    return points.unsqueeze(0)


def _render_plane_view(K: np.ndarray, R: np.ndarray, C: np.ndarray, size: int):
    """Ray-cast a checkerboard plane at world z=PLANE_Z through camera (K, R, C)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    v_grid, u_grid = np.mgrid[0:size, 0:size].astype(np.float64)
    dir_cam = np.stack([(u_grid - cx) / fx, (v_grid - cy) / fy, np.ones_like(u_grid)], axis=-1)
    dir_world = dir_cam @ R  # == R.T @ dir_cam per-pixel, since x_world = R.T @ x_cam + C

    denom = dir_world[..., 2]
    valid = np.abs(denom) > 1e-8
    t = np.divide(PLANE_Z - C[2], denom, out=np.full_like(denom, -1.0), where=valid)
    valid &= t > 1e-6

    world_xy = C[:2] + t[..., None] * dir_world[..., :2]
    valid &= (np.abs(world_xy[..., 0]) <= PLANE_HALF_EXTENT) & (
        np.abs(world_xy[..., 1]) <= PLANE_HALF_EXTENT
    )

    checker = (
        np.floor(world_xy[..., 0]).astype(np.int64) + np.floor(world_xy[..., 1]).astype(np.int64)
    ) % 2
    rgb = np.where(
        checker[..., None] == 0, np.array([220, 60, 60]), np.array([60, 60, 220])
    ).astype(np.uint8)
    rgb = np.broadcast_to(rgb, (size, size, 3)).copy()
    depth = np.where(valid, t, 0.0).astype(np.float32)
    mask = valid.astype(np.uint8)
    return rgb, depth, mask


def _full_metadata() -> dict:
    return {
        "format_version": "1.0", "model_name": "SHARP", "model_version": "test",
        "source_width": OUTPUT_SIZE, "source_height": OUTPUT_SIZE,
        "output_width": OUTPUT_SIZE, "output_height": OUTPUT_SIZE,
        "num_views": 9, "view_layout": "3x3",
        "horizontal_angle": 10.0, "vertical_angle": 10.0,
        "coordinate_system": "OpenCV",
        "pose_convention": "R world->camera, C camera position",
        "depth_definition": "camera_z", "depth_unit": "meter",
    }


def make_consistent_result() -> SpatialPhotoResult:
    """A 9-view SpatialPhotoResult that is fully geometrically self-consistent."""
    mean_vectors = _synthetic_mean_vectors()
    rig = build_camera_rig(
        mean_vectors, f_px=40.0, source_width=OUTPUT_SIZE, source_height=OUTPUT_SIZE,
        angle_deg=10.0, output_width=OUTPUT_SIZE, output_height=OUTPUT_SIZE,
    )
    rgb_list, depth_list, mask_list = [], [], []
    for pose in rig:
        rgb, depth, mask = _render_plane_view(pose.K, pose.R, pose.C, OUTPUT_SIZE)
        rgb_list.append(rgb)
        depth_list.append(depth)
        mask_list.append(mask)
    return SpatialPhotoResult(
        rgb=np.stack(rgb_list), depth=np.stack(depth_list), mask=np.stack(mask_list),
        K=np.stack([p.K for p in rig]), R=np.stack([p.R for p in rig]),
        C=np.stack([p.C for p in rig]), metadata=_full_metadata(),
    )


@pytest.fixture
def consistent_spatial_photo_result() -> SpatialPhotoResult:
    """Fresh copy of a geometrically consistent 9-view result for each test."""
    return make_consistent_result()
```

- [ ] **Step 2: Write the failing tests**

Create `tests/sharp_spatialize/test_validation.py`:

```python
"""Tests for sharp_spatialize.validation."""

from __future__ import annotations

import numpy as np
import pytest

from sharp_spatialize.hdf5_io import save
from sharp_spatialize.validation import ValidationError, validate_file, validate_in_memory


def test_validate_in_memory_accepts_a_consistent_synthetic_scene(consistent_spatial_photo_result):
    validate_in_memory(consistent_spatial_photo_result)  # must not raise


def test_check_shapes_flags_wrong_rgb_shape(consistent_spatial_photo_result):
    result = consistent_spatial_photo_result
    result.rgb = result.rgb[..., :2]

    with pytest.raises(ValidationError, match="rgb must have shape"):
        validate_in_memory(result)


def test_check_metadata_flags_missing_keys(consistent_spatial_photo_result):
    result = consistent_spatial_photo_result
    del result.metadata["depth_unit"]

    with pytest.raises(ValidationError, match="missing required metadata key: depth_unit"):
        validate_in_memory(result)


def test_check_camera_rotations_flags_non_orthonormal_r(consistent_spatial_photo_result):
    result = consistent_spatial_photo_result
    result.R[0] = result.R[0] * 2.0

    with pytest.raises(ValidationError, match=r"R\^T R deviates"):
        validate_in_memory(result)


def test_check_depth_flags_nan(consistent_spatial_photo_result):
    result = consistent_spatial_photo_result
    result.depth[0, 0, 0] = np.nan

    with pytest.raises(ValidationError, match="NaN or Inf"):
        validate_in_memory(result)


def test_check_depth_flags_nonzero_depth_at_invalid_mask(consistent_spatial_photo_result):
    result = consistent_spatial_photo_result
    invalid_pixels = result.mask[4] == 0
    assert invalid_pixels.any(), "fixture must have at least one invalid pixel in view 4"
    result.depth[4][invalid_pixels] = 3.0

    with pytest.raises(ValidationError, match="non-zero at pixels marked invalid"):
        validate_in_memory(result)


def test_reprojection_fails_when_depth_is_corrupted(consistent_spatial_photo_result):
    result = consistent_spatial_photo_result
    result.depth[3] += 2.0  # break consistency between views 3 and 5

    with pytest.raises(ValidationError, match="reprojection"):
        validate_in_memory(result)


def test_validate_file_reports_all_failures_without_raising(tmp_path, consistent_spatial_photo_result):
    result = consistent_spatial_photo_result
    del result.metadata["depth_unit"]
    result.R[0] = result.R[0] * 2.0
    path = tmp_path / "broken.h5"
    save(path, result)

    report = validate_file(path)

    assert not report.ok
    assert any("depth_unit" in failure for failure in report.failures)
    assert any("R^T R" in failure for failure in report.failures)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/sharp_spatialize/test_validation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sharp_spatialize.validation'`

- [ ] **Step 4: Write `src/sharp_spatialize/validation.py`**

```python
"""Validates a SpatialPhotoResult against the Stage 1 spec's checks: file
shapes/metadata, camera rotation validity, depth sanity, and a reprojection
consistency test between two views.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import hdf5_io
from .hdf5_io import SpatialPhotoResult

REQUIRED_METADATA_KEYS = (
    "format_version", "model_name", "model_version",
    "source_width", "source_height", "output_width", "output_height",
    "num_views", "view_layout", "horizontal_angle", "vertical_angle",
    "coordinate_system", "pose_convention", "depth_definition", "depth_unit",
)

ROTATION_TOLERANCE = 1e-4
REPROJECTION_MAX_MEAN_ABS_ERROR = 40.0
REPROJECTION_MIN_COMPARED_PIXELS = 25


class ValidationError(Exception):
    """Raised by `validate_in_memory` when a SpatialPhotoResult fails validation."""


@dataclass
class ValidationReport:
    """All validation failures found, if any."""

    ok: bool
    failures: list[str] = field(default_factory=list)


def _check_shapes(result: SpatialPhotoResult) -> list[str]:
    failures = []
    if result.rgb.ndim != 4 or result.rgb.shape[-1] != 3:
        failures.append(f"rgb must have shape [N,H,W,3], got {result.rgb.shape}")
        return failures  # can't infer N reliably from a malformed rgb array

    num_views = result.rgb.shape[0]
    for name, arr in (("depth", result.depth), ("mask", result.mask)):
        if arr.ndim != 3 or arr.shape[0] != num_views:
            failures.append(f"{name} must have shape [{num_views},H,W], got {arr.shape}")
    for name, arr in (("K", result.K), ("R", result.R)):
        if arr.shape != (num_views, 3, 3):
            failures.append(f"{name} must have shape [{num_views},3,3], got {arr.shape}")
    if result.C.shape != (num_views, 3):
        failures.append(f"C must have shape [{num_views},3], got {result.C.shape}")
    return failures


def _check_metadata(metadata: dict) -> list[str]:
    return [
        f"missing required metadata key: {key}"
        for key in REQUIRED_METADATA_KEYS
        if key not in metadata
    ]


def _check_camera_rotations(R: np.ndarray) -> list[str]:
    failures = []
    identity = np.eye(3)
    for i, r_i in enumerate(R):
        orth_error = float(np.max(np.abs(r_i.T @ r_i - identity)))
        det = float(np.linalg.det(r_i))
        if orth_error >= ROTATION_TOLERANCE:
            failures.append(f"view {i}: R^T R deviates from identity by {orth_error:.2e}")
        if abs(det - 1.0) >= ROTATION_TOLERANCE:
            failures.append(f"view {i}: det(R)={det:.6f}, expected ~1.0")
    return failures


def _check_depth(depth: np.ndarray, mask: np.ndarray) -> list[str]:
    if not np.all(np.isfinite(depth)):
        return ["depth contains NaN or Inf values"]

    failures = []
    valid = mask == 1
    if valid.any() and np.any(depth[valid] <= 0):
        failures.append("depth has non-positive values at pixels marked valid by the mask")
    invalid = mask == 0
    if invalid.any() and np.any(depth[invalid] != 0.0):
        failures.append("depth is non-zero at pixels marked invalid by the mask")
    return failures


def _reproject(
    depth_src: np.ndarray, K_src: np.ndarray, R_src: np.ndarray, C_src: np.ndarray,
    K_dst: np.ndarray, R_dst: np.ndarray, C_dst: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproject every pixel of the source view into the destination camera.

    Returns (u_dst, v_dst, in_front), each shape (H, W).
    """
    height, width = depth_src.shape
    v_grid, u_grid = np.mgrid[0:height, 0:width].astype(np.float32)

    fx, fy, cx, cy = K_src[0, 0], K_src[1, 1], K_src[0, 2], K_src[1, 2]
    x_cam = (u_grid - cx) / fx * depth_src
    y_cam = (v_grid - cy) / fy * depth_src
    z_cam = depth_src

    points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    points_world = points_cam @ R_src + C_src  # == R_src.T @ points_cam + C_src

    points_dst_cam = (points_world - C_dst) @ R_dst.T  # == R_dst @ (points_world - C_dst)
    in_front = points_dst_cam[..., 2] > 1e-6

    fx_d, fy_d, cx_d, cy_d = K_dst[0, 0], K_dst[1, 1], K_dst[0, 2], K_dst[1, 2]
    z_safe = np.where(in_front, points_dst_cam[..., 2], 1.0)
    u_dst = fx_d * points_dst_cam[..., 0] / z_safe + cx_d
    v_dst = fy_d * points_dst_cam[..., 1] / z_safe + cy_d
    return u_dst, v_dst, in_front


def _check_reprojection(result: SpatialPhotoResult, src_idx: int, dst_idx: int) -> list[str]:
    height, width = result.depth.shape[1:3]
    u_dst, v_dst, in_front = _reproject(
        result.depth[src_idx], result.K[src_idx], result.R[src_idx], result.C[src_idx],
        result.K[dst_idx], result.R[dst_idx], result.C[dst_idx],
    )
    u_round = np.round(u_dst).astype(np.int64)
    v_round = np.round(v_dst).astype(np.int64)
    in_bounds = (u_round >= 0) & (u_round < width) & (v_round >= 0) & (v_round < height)

    comparable = (result.mask[src_idx] == 1) & in_front & in_bounds
    if comparable.sum() < REPROJECTION_MIN_COMPARED_PIXELS:
        return [
            f"reprojection view {src_idx}->{dst_idx}: only {int(comparable.sum())} pixels had "
            f"an in-bounds, in-front, valid correspondence "
            f"(need >= {REPROJECTION_MIN_COMPARED_PIXELS})"
        ]

    u_hit, v_hit = u_round[comparable], v_round[comparable]
    dst_valid = result.mask[dst_idx][v_hit, u_hit] == 1
    if not dst_valid.any():
        return [f"reprojection view {src_idx}->{dst_idx}: no comparable pixels land on valid target pixels"]

    src_colors = result.rgb[src_idx][comparable][dst_valid].astype(np.float32)
    dst_colors = result.rgb[dst_idx][v_hit[dst_valid], u_hit[dst_valid]].astype(np.float32)
    mean_abs_error = float(np.mean(np.abs(src_colors - dst_colors)))
    if mean_abs_error > REPROJECTION_MAX_MEAN_ABS_ERROR:
        return [
            f"reprojection view {src_idx}->{dst_idx}: mean abs color error {mean_abs_error:.1f} "
            f"exceeds threshold {REPROJECTION_MAX_MEAN_ABS_ERROR}"
        ]
    return []


def _validate(result: SpatialPhotoResult) -> ValidationReport:
    failures: list[str] = []
    shape_failures = _check_shapes(result)
    failures += shape_failures
    failures += _check_metadata(result.metadata)

    if not shape_failures:
        failures += _check_camera_rotations(result.R)
        failures += _check_depth(result.depth, result.mask)
        num_views = result.rgb.shape[0]
        if num_views >= 6:
            failures += _check_reprojection(result, 3, 5)

    return ValidationReport(ok=len(failures) == 0, failures=failures)


def validate_in_memory(result: SpatialPhotoResult) -> None:
    """Validate `result`; raise ValidationError listing every failure found."""
    report = _validate(result)
    if not report.ok:
        raise ValidationError("; ".join(report.failures))


def validate_file(path: str | Path) -> ValidationReport:
    """Load `path` and validate it, without raising."""
    result = hdf5_io.load(path)
    return _validate(result)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/sharp_spatialize/test_validation.py -v`
Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add src/sharp_spatialize/validation.py tests/sharp_spatialize/conftest.py tests/sharp_spatialize/test_validation.py
git commit -m "sharp_spatialize: add validation logic (validation.py)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LDaau3BWz9CKHKLfpGVU2B"
```

---

### Task 4: SHARP inference wrapper (`inference.py`)

**Files:**
- Create: `src/sharp_spatialize/inference.py`
- Test: `tests/sharp_spatialize/test_inference.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-3.
- Produces: `SceneBundle` dataclass (`gaussians: Gaussians3D, f_px: float, width: int, height: int, device: str, model_version: str`), `pick_device(requested: str) -> str`, `infer(image_path, checkpoint_path=None, device="default") -> SceneBundle` (`scene = sharp.infer(image)` boundary — the only place in the package that imports `sharp.models`).

- [ ] **Step 1: Write the failing tests**

Create `tests/sharp_spatialize/test_inference.py`:

```python
"""Tests for sharp_spatialize.inference.

`_run_forward_pass` is tested with a fake predictor stub so this file needs
no network access, checkpoint download, or GPU.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

import sharp_spatialize.inference as inference_module
from sharp.utils.gaussians import Gaussians3D


class _FakePredictor(torch.nn.Module):
    """Stands in for the real SHARP predictor: returns a fixed-size Gaussians3D."""

    def __init__(self, num_points: int = 10) -> None:
        super().__init__()
        self.num_points = num_points

    def forward(self, image, disparity_factor):  # noqa: ARG002 - matches predictor signature
        n = self.num_points
        mean_vectors = torch.rand(1, n, 3) * 0.5 + 0.25
        return Gaussians3D(
            mean_vectors=mean_vectors,
            singular_values=torch.ones(1, n, 3) * 0.01,
            quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(1, n, 1),
            colors=torch.rand(1, n, 3),
            opacities=torch.ones(1, n) * 0.9,
        )


def test_pick_device_passes_through_an_explicit_choice():
    assert inference_module.pick_device("cpu") == "cpu"


def test_run_forward_pass_returns_finite_gaussians_of_expected_batch_size():
    image = (np.random.default_rng(0).random((64, 96, 3)) * 255).astype(np.uint8)
    gaussians = inference_module._run_forward_pass(_FakePredictor(num_points=12), image, 80.0, "cpu")

    assert gaussians.mean_vectors.shape == (1, 12, 3)
    assert torch.isfinite(gaussians.mean_vectors).all()


def test_infer_wires_together_load_predictor_and_forward_pass(tmp_path, monkeypatch):
    image_path = tmp_path / "input.png"
    Image.fromarray(
        (np.random.default_rng(1).random((40, 50, 3)) * 255).astype(np.uint8)
    ).save(image_path)

    monkeypatch.setattr(
        inference_module, "_load_predictor",
        lambda checkpoint_path, device: (_FakePredictor(num_points=8), "fake-model-v1"),
    )

    scene = inference_module.infer(image_path, device="cpu")

    assert scene.width == 50
    assert scene.height == 40
    assert scene.device == "cpu"
    assert scene.model_version == "fake-model-v1"
    assert scene.gaussians.mean_vectors.shape[0] == 1
    assert torch.isfinite(scene.gaussians.mean_vectors).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/sharp_spatialize/test_inference.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sharp_spatialize.inference'`

- [ ] **Step 3: Write `src/sharp_spatialize/inference.py`**

```python
"""Isolates all SHARP-specific model loading and inference behind `infer()`.

No other module in sharp_spatialize imports `sharp.models` or touches the
predictor directly -- everything downstream only sees the `SceneBundle` this
module returns. This is the `scene = sharp.infer(image)` boundary the spec
calls for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F

from sharp.cli.predict import DEFAULT_MODEL_URL
from sharp.models import PredictorParams, RGBGaussianPredictor, create_predictor
from sharp.utils import io as sharp_io
from sharp.utils.gaussians import Gaussians3D, unproject_gaussians

INTERNAL_SHAPE = (1536, 1536)  # mirrors sharp.cli.predict.predict_image


@dataclass
class SceneBundle:
    """Everything downstream sharp_spatialize modules need about one SHARP scene."""

    gaussians: Gaussians3D
    f_px: float
    width: int
    height: int
    device: str
    model_version: str


def pick_device(requested: str) -> str:
    """Resolve "default" to the best available device; pass through anything else."""
    if requested != "default":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.mps.is_available():
        return "mps"
    return "cpu"


def _load_predictor(
    checkpoint_path: Path | None, device: str
) -> tuple[RGBGaussianPredictor, str]:
    """Load the SHARP predictor and return it with a model-version string."""
    if checkpoint_path is None:
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
        model_version = Path(urlparse(DEFAULT_MODEL_URL).path).name
    else:
        state_dict = torch.load(checkpoint_path, weights_only=True)
        model_version = Path(checkpoint_path).name

    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(device)
    return predictor, model_version


def _run_forward_pass(
    predictor: RGBGaussianPredictor, image: np.ndarray, f_px: float, device: str
) -> Gaussians3D:
    """Run one SHARP forward pass and unproject the result into metric world space."""
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)
    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(INTERNAL_SHAPE[1], INTERNAL_SHAPE[0]),
        mode="bilinear",
        align_corners=True,
    )
    with torch.no_grad():
        gaussians_ndc = predictor(image_resized_pt, disparity_factor)

    intrinsics = (
        torch.tensor(
            [[f_px, 0, width / 2, 0], [0, f_px, height / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= INTERNAL_SHAPE[0] / width
    intrinsics_resized[1] *= INTERNAL_SHAPE[1] / height

    return unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, INTERNAL_SHAPE
    )


def infer(
    image_path: str | Path,
    checkpoint_path: Path | None = None,
    device: str = "default",
) -> SceneBundle:
    """Run SHARP on `image_path` and return the resulting scene."""
    resolved_device = pick_device(device)
    predictor, model_version = _load_predictor(checkpoint_path, resolved_device)

    image, _icc_profile, f_px = sharp_io.load_rgb(Path(image_path))
    height, width = image.shape[:2]
    gaussians = _run_forward_pass(predictor, image, f_px, resolved_device)

    return SceneBundle(
        gaussians=gaussians,
        f_px=f_px,
        width=width,
        height=height,
        device=resolved_device,
        model_version=model_version,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/sharp_spatialize/test_inference.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sharp_spatialize/inference.py tests/sharp_spatialize/test_inference.py
git commit -m "sharp_spatialize: add SHARP inference wrapper (inference.py)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LDaau3BWz9CKHKLfpGVU2B"
```

---

### Task 5: Gaussian rendering (`render.py`) — CUDA-gated

**Files:**
- Create: `src/sharp_spatialize/render.py`
- Test: `tests/sharp_spatialize/test_render.py`

**Interfaces:**
- Consumes: `sharp_spatialize.cameras.CameraPose` (Task 1).
- Produces: `render_views(gaussians: Gaussians3D, rig: list[CameraPose], output_width: int, output_height: int, mask_alpha_threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — `(rgb [N,H,W,3] uint8, depth [N,H,W] float32, mask [N,H,W] uint8)` where `N = len(rig)`. Raises `RuntimeError` if CUDA is unavailable.

- [ ] **Step 1: Write the failing tests**

Create `tests/sharp_spatialize/test_render.py`:

```python
"""Tests for sharp_spatialize.render.

Only the "no CUDA" guard clause is meaningfully testable on a machine without
a GPU; the real rendering path is skip-marked and runs wherever CUDA is
available (see also tests/sharp_spatialize/test_e2e.py).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sharp.utils.gaussians import Gaussians3D
from sharp_spatialize.cameras import CameraPose
from sharp_spatialize.render import render_views


def _dummy_gaussians() -> Gaussians3D:
    return Gaussians3D(
        mean_vectors=torch.zeros(1, 1, 3),
        singular_values=torch.ones(1, 1, 3),
        quaternions=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        colors=torch.zeros(1, 1, 3),
        opacities=torch.ones(1, 1),
    )


def _dummy_pose() -> CameraPose:
    return CameraPose(
        K=np.eye(3, dtype=np.float32), R=np.eye(3, dtype=np.float32), C=np.zeros(3, dtype=np.float32)
    )


@pytest.mark.skipif(torch.cuda.is_available(), reason="only meaningful without a CUDA GPU")
def test_render_views_raises_a_clear_error_without_cuda():
    with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
        render_views(_dummy_gaussians(), [_dummy_pose()], output_width=8, output_height=8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_render_views_produces_expected_shapes_on_cuda():
    rgb, depth, mask = render_views(
        _dummy_gaussians(), [_dummy_pose(), _dummy_pose()], output_width=8, output_height=8
    )
    assert rgb.shape == (2, 8, 8, 3)
    assert rgb.dtype == np.uint8
    assert depth.shape == (2, 8, 8)
    assert depth.dtype == np.float32
    assert mask.shape == (2, 8, 8)
    assert mask.dtype == np.uint8
    assert np.isfinite(depth).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/sharp_spatialize/test_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sharp_spatialize.render'`

- [ ] **Step 3: Write `src/sharp_spatialize/render.py`**

```python
"""Renders RGB, depth, and validity masks for a camera rig via SHARP's
Gaussian rasterizer (`sharp.utils.gsplat`).

Requires a CUDA GPU -- gsplat has no CPU/MPS kernel, the same constraint
`sharp.cli.render` already enforces for `sharp render`.
"""

from __future__ import annotations

import numpy as np
import torch

from sharp.utils.gaussians import Gaussians3D
from sharp.utils.gsplat import GSplatRenderer

from .cameras import CameraPose

DEFAULT_MASK_ALPHA_THRESHOLD = 0.5


def render_views(
    gaussians: Gaussians3D,
    rig: list[CameraPose],
    output_width: int,
    output_height: int,
    mask_alpha_threshold: float = DEFAULT_MASK_ALPHA_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render every camera in `rig` from the same `gaussians` scene.

    Returns (rgb [N,H,W,3] uint8, depth [N,H,W] float32, mask [N,H,W] uint8),
    where N = len(rig). Depth is camera-space Z; invalid (low-alpha) pixels
    get depth forced to 0.0 and mask 0, so no output pixel is ever NaN/Inf.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "sharp-spatialize requires a CUDA GPU for view rendering "
            "(gsplat has no CPU/MPS kernel)."
        )
    device = torch.device("cuda")
    gaussians_cuda = gaussians.to(device)
    renderer = GSplatRenderer(color_space="linearRGB")

    rgb_frames, depth_frames, mask_frames = [], [], []
    for pose in rig:
        r_t = torch.from_numpy(pose.R).to(device=device, dtype=torch.float32)
        c_t = torch.from_numpy(pose.C).to(device=device, dtype=torch.float32)
        k_t = torch.from_numpy(pose.K).to(device=device, dtype=torch.float32)

        extrinsics = torch.eye(4, dtype=torch.float32, device=device)
        extrinsics[:3, :3] = r_t
        extrinsics[:3, 3] = -(r_t @ c_t)

        intrinsics = torch.eye(4, dtype=torch.float32, device=device)
        intrinsics[:3, :3] = k_t

        output = renderer(
            gaussians_cuda,
            extrinsics=extrinsics[None],
            intrinsics=intrinsics[None],
            image_width=output_width,
            image_height=output_height,
        )

        alpha = output.alpha[0, 0]
        mask = alpha > mask_alpha_threshold
        depth = torch.where(mask, output.depth[0, 0], torch.zeros_like(output.depth[0, 0]))
        rgb = (output.color[0].permute(1, 2, 0).clamp(0.0, 1.0) * 255.0).to(torch.uint8)

        rgb_frames.append(rgb.cpu().numpy())
        depth_frames.append(depth.cpu().numpy().astype(np.float32))
        mask_frames.append(mask.cpu().numpy().astype(np.uint8))

    return np.stack(rgb_frames), np.stack(depth_frames), np.stack(mask_frames)
```

- [ ] **Step 4: Run the tests to verify they pass (on this machine, the no-CUDA branch)**

Run: `.venv/bin/python -m pytest tests/sharp_spatialize/test_render.py -v`
Expected: 1 passed, 1 skipped (the CUDA-shapes test is skipped here; it will run on a GPU machine).

- [ ] **Step 5: Commit**

```bash
git add src/sharp_spatialize/render.py tests/sharp_spatialize/test_render.py
git commit -m "sharp_spatialize: add gsplat-based view rendering (render.py)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LDaau3BWz9CKHKLfpGVU2B"
```

---

### Task 6: Public Python API (`api.py` + `__init__.py`)

**Files:**
- Modify: `src/sharp_spatialize/__init__.py`
- Create: `src/sharp_spatialize/api.py`
- Test: `tests/sharp_spatialize/test_api.py`

**Interfaces:**
- Consumes: `inference.infer`, `inference.SceneBundle` (Task 4); `cameras.build_camera_rig` (Task 1); `render.render_views` (Task 5); `hdf5_io.SpatialPhotoResult`, `hdf5_io.load` (Task 2); `validation.validate_in_memory`, `validation.ValidationError` (Task 3).
- Produces: `generate_spatial_photo(image_path, angle_deg=10.0, output_width=None, output_height=None, checkpoint_path=None, device="default") -> SpatialPhotoResult`, `load_spatial_photo(path) -> SpatialPhotoResult`. Package-level exports: `sharp_spatialize.generate_spatial_photo`, `sharp_spatialize.load_spatial_photo`, `sharp_spatialize.SpatialPhotoResult`, `sharp_spatialize.ValidationError`.

- [ ] **Step 1: Write the failing tests**

Create `tests/sharp_spatialize/test_api.py`:

```python
"""Tests for sharp_spatialize.api. inference.infer and render.render_views
are monkeypatched so this file needs no SHARP checkpoint, network, or GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import sharp_spatialize.api as api_module
from sharp.utils.gaussians import Gaussians3D
from sharp_spatialize.inference import SceneBundle
from sharp_spatialize.validation import ValidationError


def _fake_scene(width: int = 32, height: int = 32) -> SceneBundle:
    xs = torch.linspace(-2.0, 2.0, 10)
    ys = torch.linspace(-2.0, 2.0, 10)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    xy = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    mean_vectors = torch.cat([xy, torch.full((xy.shape[0], 1), 5.0)], dim=-1).unsqueeze(0)
    n = mean_vectors.shape[1]
    gaussians = Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=torch.ones(1, n, 3) * 0.01,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(1, n, 1),
        colors=torch.rand(1, n, 3),
        opacities=torch.ones(1, n),
    )
    return SceneBundle(
        gaussians=gaussians, f_px=40.0, width=width, height=height,
        device="cpu", model_version="fake-v1",
    )


def _fake_render_views(gaussians, rig, output_width, output_height, mask_alpha_threshold=0.5):
    n = len(rig)
    rgb = np.zeros((n, output_height, output_width, 3), dtype=np.uint8)
    depth = np.full((n, output_height, output_width), 5.0, dtype=np.float32)
    mask = np.ones((n, output_height, output_width), dtype=np.uint8)
    return rgb, depth, mask


def test_generate_spatial_photo_wires_the_pipeline_and_returns_a_valid_result(tmp_path, monkeypatch):
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"not a real jpeg -- inference.infer is mocked")

    monkeypatch.setattr(api_module.inference, "infer", lambda *a, **k: _fake_scene())
    monkeypatch.setattr(api_module.render, "render_views", _fake_render_views)

    result = api_module.generate_spatial_photo(image_path, angle_deg=10.0)

    assert result.rgb.shape == (9, 32, 32, 3)
    assert result.depth.shape == (9, 32, 32)
    assert result.mask.shape == (9, 32, 32)
    assert result.K.shape == (9, 3, 3)
    assert result.R.shape == (9, 3, 3)
    assert result.C.shape == (9, 3)
    assert result.metadata["num_views"] == 9
    assert result.metadata["horizontal_angle"] == 10.0
    assert result.metadata["vertical_angle"] == 10.0
    assert result.metadata["source_filename"] == "input.jpg"
    assert result.metadata["depth_unit"] == "meter"

    out_path = tmp_path / "spatial_photo.h5"
    result.save(out_path)
    assert out_path.exists()


def test_generate_spatial_photo_raises_when_validation_fails(tmp_path, monkeypatch):
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"not a real jpeg -- inference.infer is mocked")

    def _broken_render_views(gaussians, rig, output_width, output_height, mask_alpha_threshold=0.5):
        rgb, depth, mask = _fake_render_views(gaussians, rig, output_width, output_height)
        depth[0, 0, 0] = np.nan
        return rgb, depth, mask

    monkeypatch.setattr(api_module.inference, "infer", lambda *a, **k: _fake_scene())
    monkeypatch.setattr(api_module.render, "render_views", _broken_render_views)

    with pytest.raises(ValidationError):
        api_module.generate_spatial_photo(image_path)


def test_load_spatial_photo_round_trips_a_saved_result(tmp_path, monkeypatch):
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"not a real jpeg -- inference.infer is mocked")
    monkeypatch.setattr(api_module.inference, "infer", lambda *a, **k: _fake_scene())
    monkeypatch.setattr(api_module.render, "render_views", _fake_render_views)

    result = api_module.generate_spatial_photo(image_path)
    out_path = tmp_path / "spatial_photo.h5"
    result.save(out_path)

    reloaded = api_module.load_spatial_photo(out_path)
    np.testing.assert_array_equal(reloaded.rgb, result.rgb)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/sharp_spatialize/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sharp_spatialize.api'`

- [ ] **Step 3: Write `src/sharp_spatialize/api.py`**

```python
"""Public Python API: `generate_spatial_photo` ties inference, camera rig
construction, rendering, HDF5 packaging, and validation into one call.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import cameras, inference, render, validation
from .hdf5_io import SpatialPhotoResult, load

DEFAULT_ANGLE_DEG = 10.0


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
        "pose_convention": "R is world->camera; C is the camera's world position; x_cam = R @ (x_world - C)",
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
) -> SpatialPhotoResult:
    """Convert one RGB image into a 9-view SpatialPhotoResult.

    Raises `validation.ValidationError` if the generated scene fails
    validation -- nothing is saved to disk in that case.
    """
    image_path = Path(image_path)
    scene = inference.infer(
        image_path,
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        device=device,
    )

    width = output_width or scene.width
    height = output_height or scene.height

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
    validation.validate_in_memory(result)
    return result


def load_spatial_photo(path: str | Path) -> SpatialPhotoResult:
    """Load a previously saved spatial_photo.h5 file. Does not require SHARP or CUDA."""
    return load(path)
```

- [ ] **Step 4: Update `src/sharp_spatialize/__init__.py`**

```python
"""sharp_spatialize: converts one RGB image into a 9-view spatial photo."""

from .api import generate_spatial_photo, load_spatial_photo
from .hdf5_io import SpatialPhotoResult
from .validation import ValidationError

__all__ = ["generate_spatial_photo", "load_spatial_photo", "SpatialPhotoResult", "ValidationError"]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/sharp_spatialize/test_api.py -v`
Expected: 3 passed.

- [ ] **Step 6: Run the full test suite so far**

Run: `.venv/bin/python -m pytest tests/sharp_spatialize/ -v`
Expected: all passing (render's CUDA-shapes test still skipped on this machine).

- [ ] **Step 7: Commit**

```bash
git add src/sharp_spatialize/__init__.py src/sharp_spatialize/api.py tests/sharp_spatialize/test_api.py
git commit -m "sharp_spatialize: add public Python API (api.py)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LDaau3BWz9CKHKLfpGVU2B"
```

---

### Task 7: CLI (`cli.py`) + console-script entry points

**Files:**
- Create: `src/sharp_spatialize/cli.py`
- Test: `tests/sharp_spatialize/test_cli.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `api.generate_spatial_photo` (Task 6); `validation.validate_file`, `validation.ValidationError` (Task 3); `tests/sharp_spatialize/conftest.py`'s `consistent_spatial_photo_result` fixture (Task 3).
- Produces: Click commands `generate_cli`, `validate_cli`; console scripts `sharp-spatialize` and `sharp-spatialize-validate`.

- [ ] **Step 1: Write the failing tests**

Create `tests/sharp_spatialize/test_cli.py`:

```python
"""Tests for sharp_spatialize.cli that don't require CUDA or a real model."""

from __future__ import annotations

from click.testing import CliRunner

from sharp_spatialize.cli import generate_cli, validate_cli
from sharp_spatialize.hdf5_io import save


def test_validate_cli_passes_on_a_consistent_file(tmp_path, consistent_spatial_photo_result):
    path = tmp_path / "spatial_photo.h5"
    save(path, consistent_spatial_photo_result)

    result = CliRunner().invoke(validate_cli, [str(path)])

    assert result.exit_code == 0
    assert "PASS" in result.output


def test_validate_cli_fails_on_a_broken_file(tmp_path, consistent_spatial_photo_result):
    broken = consistent_spatial_photo_result
    del broken.metadata["depth_unit"]
    path = tmp_path / "broken.h5"
    save(path, broken)

    result = CliRunner().invoke(validate_cli, [str(path)])

    assert result.exit_code == 1
    assert "FAIL" in result.output
    assert "depth_unit" in result.output


def test_generate_cli_reports_a_clear_error_for_a_missing_input_file(tmp_path):
    result = CliRunner().invoke(
        generate_cli,
        ["--input", str(tmp_path / "does_not_exist.jpg"), "--output", str(tmp_path / "out.h5")],
    )

    assert result.exit_code != 0
    assert "does_not_exist.jpg" in result.output
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/sharp_spatialize/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sharp_spatialize.cli'`

- [ ] **Step 3: Write `src/sharp_spatialize/cli.py`**

```python
"""Command-line entry points: `sharp-spatialize` and `sharp-spatialize-validate`."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from .api import generate_spatial_photo
from .validation import ValidationError, validate_file


@click.command()
@click.option(
    "-i", "--input", "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True, help="Input JPEG/PNG image.",
)
@click.option(
    "-o", "--output", "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True, help="Output spatial_photo.h5 path.",
)
@click.option(
    "--angle", "angle_deg", type=float, default=10.0, show_default=True,
    help="Horizontal/vertical camera angle in degrees. Any positive float works; "
    "5, 10, and 15 are the values this tool is validated against.",
)
@click.option("--width", "output_width", type=int, default=None, help="Output width in pixels (defaults to the source image width).")
@click.option("--height", "output_height", type=int, default=None, help="Output height in pixels (defaults to the source image height).")
@click.option(
    "-c", "--checkpoint", "checkpoint_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
    help="SHARP checkpoint path (downloads the default model if omitted).",
)
@click.option(
    "--device", default="default",
    help="Device for SHARP inference: cpu, mps, cuda, or default (auto-detect). "
    "Rendering always requires CUDA regardless of this setting.",
)
def generate_cli(
    input_path: Path,
    output_path: Path,
    angle_deg: float,
    output_width: int | None,
    output_height: int | None,
    checkpoint_path: Path | None,
    device: str,
) -> None:
    """Convert one RGB image into a 9-view spatial_photo.h5 file."""
    try:
        result = generate_spatial_photo(
            input_path, angle_deg=angle_deg, output_width=output_width,
            output_height=output_height, checkpoint_path=checkpoint_path, device=device,
        )
    except ValidationError as exc:
        click.echo(f"ERROR: generated scene failed validation: {exc}", err=True)
        sys.exit(1)

    result.save(output_path)
    click.echo(
        f"Wrote {output_path} "
        f"({result.rgb.shape[0]} views, {result.rgb.shape[2]}x{result.rgb.shape[1]})."
    )


@click.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def validate_cli(input_path: Path) -> None:
    """Validate an existing spatial_photo.h5 file. Does not require SHARP or CUDA."""
    report = validate_file(input_path)
    if report.ok:
        click.echo(f"PASS: {input_path} is a valid spatial photo.")
        return
    click.echo(f"FAIL: {input_path} failed validation:", err=True)
    for failure in report.failures:
        click.echo(f"  - {failure}", err=True)
    sys.exit(1)
```

- [ ] **Step 4: Register the console scripts in `pyproject.toml`**

Edit the `[project.scripts]` section:

```toml
[project.scripts]
sharp = "sharp.cli:main_cli"
sharp-spatialize = "sharp_spatialize.cli:generate_cli"
sharp-spatialize-validate = "sharp_spatialize.cli:validate_cli"
```

Run:
```bash
/Users/macariofang/.local/bin/uv pip install --python .venv/bin/python -e .
ls .venv/bin/sharp-spatialize .venv/bin/sharp-spatialize-validate
```
Expected: both files listed (the console scripts are now installed).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/sharp_spatialize/test_cli.py -v`
Expected: 3 passed.

- [ ] **Step 6: Smoke-test the installed console scripts**

Run: `.venv/bin/sharp-spatialize --help` and `.venv/bin/sharp-spatialize-validate --help`
Expected: both print usage text without error.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/sharp_spatialize/cli.py tests/sharp_spatialize/test_cli.py
git commit -m "sharp_spatialize: add CLI and console-script entry points

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LDaau3BWz9CKHKLfpGVU2B"
```

---

### Task 8: End-to-end CUDA tests

**Files:**
- Create: `tests/sharp_spatialize/test_e2e.py`

**Interfaces:**
- Consumes: `api.generate_spatial_photo` (Task 6), `cameras.camera_at` (Task 1), `render.render_views` (Task 5), `hdf5_io.load` (Task 2), `validation.validate_file` (Task 3), `inference.infer` (Task 4). Uses the repo's existing `data/teaser.jpg` as the sample input.
- Produces: nothing consumed by later tasks — this is the top of the dependency graph. Entirely skipped on non-CUDA machines; runs unmodified on a CUDA machine.

- [ ] **Step 1: Write the tests**

Create `tests/sharp_spatialize/test_e2e.py`:

```python
"""End-to-end tests requiring a real CUDA GPU: SHARP inference + gsplat
rendering + HDF5 packaging + validation, all for real, using the repo's
bundled sample image. Skipped everywhere else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sharp_spatialize.api import generate_spatial_photo
from sharp_spatialize.cameras import camera_at
from sharp_spatialize.hdf5_io import load
from sharp_spatialize.inference import infer
from sharp_spatialize.render import render_views
from sharp_spatialize.validation import validate_file

SAMPLE_IMAGE = Path(__file__).resolve().parents[2] / "data" / "teaser.jpg"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


def test_generate_spatial_photo_end_to_end(tmp_path):
    result = generate_spatial_photo(SAMPLE_IMAGE, angle_deg=10.0, device="cuda")

    assert result.rgb.shape[0] == 9
    assert result.rgb.dtype == np.uint8
    assert result.depth.shape[0] == 9

    out_path = tmp_path / "spatial_photo.h5"
    result.save(out_path)

    report = validate_file(out_path)
    assert report.ok, report.failures

    reloaded = load(out_path)
    assert reloaded.rgb.shape == result.rgb.shape


def test_arbitrary_intermediate_camera_angle_renders():
    """Acceptance criterion #10: an arbitrary angle inside the configured range renders."""
    scene = infer(SAMPLE_IMAGE, device="cuda")
    pose = camera_at(
        scene.gaussians.mean_vectors, scene.f_px, scene.width, scene.height,
        h_angle_deg=3.7, v_angle_deg=-4.2, output_width=scene.width, output_height=scene.height,
    )
    rgb, depth, mask = render_views(scene.gaussians, [pose], scene.width, scene.height)

    assert rgb.shape == (1, scene.height, scene.width, 3)
    assert np.isfinite(depth).all()
```

- [ ] **Step 2: Run on this machine to confirm the skip works**

Run: `.venv/bin/python -m pytest tests/sharp_spatialize/test_e2e.py -v`
Expected: 2 skipped (no CUDA here).

- [ ] **Step 3: Commit**

```bash
git add tests/sharp_spatialize/test_e2e.py
git commit -m "sharp_spatialize: add CUDA end-to-end tests (skip-marked here)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LDaau3BWz9CKHKLfpGVU2B"
```

- [ ] **Step 4: Note for later — run for real on a CUDA machine**

On a CUDA-equipped machine, after checking out this branch: `uv pip install --python .venv/bin/python -e . && .venv/bin/python -m pytest tests/sharp_spatialize/test_e2e.py -v` should show 2 passed. This is required before Stage 1 acceptance criteria #1-4, #9, and #10 can be considered verified (see spec §15 in the design doc).

---

### Task 9: HDF5 format documentation + examples

**Files:**
- Create: `docs/spatial_photo_format.md`
- Create: `examples/README.md`

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: the "HDF5 format documentation" and "Example input and output files" deliverables from spec §14.

- [ ] **Step 1: Write `docs/spatial_photo_format.md`**

```markdown
# spatial_photo.h5 Format

A spatial photo is a 9-view HDF5 representation of one RGB image, produced by
`sharp-spatialize` (see `src/sharp_spatialize/`). It is readable with plain
`h5py` and `numpy` -- no SHARP install, no GPU, no 3D Gaussian representation.

## Reading a file

```python
import h5py

with h5py.File("spatial_photo.h5", "r") as f:
    rgb = f["rgb"][:]          # [9, H, W, 3] uint8
    depth = f["depth"][:]      # [9, H, W] float32, meters
    mask = f["mask"][:]        # [9, H, W] uint8, 1 = valid
    K = f["camera/K"][:]       # [9, 3, 3] float32
    R = f["camera/R"][:]       # [9, 3, 3] float32
    C = f["camera/C"][:]       # [9, 3] float32
    metadata = dict(f.attrs)
```

Or, with `sharp_spatialize` installed: `sharp_spatialize.load_spatial_photo(path)`.

## Datasets

| Path | Shape | Dtype | Meaning |
|---|---|---|---|
| `/rgb` | `[9, H, W, 3]` | `uint8` | RGB image per view |
| `/depth` | `[9, H, W]` | `float32` | Camera-space Z, in meters. `0.0` at invalid pixels. |
| `/mask` | `[9, H, W]` | `uint8` | `0` = invalid, `1` = valid |
| `/camera/K` | `[9, 3, 3]` | `float32` | Intrinsic matrix per view |
| `/camera/R` | `[9, 3, 3]` | `float32` | World->camera rotation per view |
| `/camera/C` | `[9, 3]` | `float32` | Camera position in world coordinates per view |

## Camera convention

OpenCV convention: X right, Y down, Z forward. For a world point `x_world`,
its camera-space coordinates are `x_cam = R @ (x_world - C)`. Depth is the Z
component of `x_cam` (`depth_definition = "camera_z"`).

## View layout

9 views in a fixed 3x3 grid, indexed 0..8 in row-major order (`view_layout =
"3x3"`). `V4` is the reference camera (the original input viewpoint):

```
V0 V1 V2      (-a,+a) (0,+a) (+a,+a)
V3 V4 V5   =  (-a, 0) (0, 0) (+a, 0)
V6 V7 V8      (-a,-a) (0,-a) (+a,-a)
```

where `a` is the file's `horizontal_angle`/`vertical_angle` attribute (in
degrees) and each pair is `(horizontal_angle, vertical_angle)`. Every camera
orbits the same fixed focus point and looks back at it, so adjacent views
have real parallax/occlusion, not just a re-oriented crop of the same pixels.

## File-level attributes (`f.attrs`)

`format_version, model_name, model_version, source_width, source_height,
source_filename, output_width, output_height, num_views, view_layout,
horizontal_angle, vertical_angle, coordinate_system, pose_convention,
depth_definition, depth_unit`.

## Validation

`sharp-spatialize-validate spatial_photo.h5` checks dataset shapes/dtypes,
required metadata, camera rotation validity (`RᵀR ≈ I`, `det(R) ≈ 1`), depth
sanity (no NaN/Inf, positive where valid, zero where invalid), and reprojects
one view into another to confirm the camera/depth conventions are internally
consistent. See `src/sharp_spatialize/validation.py`.
```

- [ ] **Step 2: Write `examples/README.md`**

```markdown
# Examples

`../data/teaser.jpg` is the bundled sample input (also used by `webui/`).

Generating `spatial_photo.h5` requires a CUDA GPU (the renderer uses SHARP's
gsplat-based rasterizer, which has no CPU/MPS kernel), so this repo does not
ship a pre-generated output file. To produce one on a CUDA machine:

```bash
sharp-spatialize --input ../data/teaser.jpg --angle 10 --output spatial_photo.h5
sharp-spatialize-validate spatial_photo.h5
```

See `docs/spatial_photo_format.md` for the file format.
```

- [ ] **Step 3: Commit**

```bash
mkdir -p examples
git add docs/spatial_photo_format.md examples/README.md
git commit -m "sharp_spatialize: add HDF5 format docs and examples README

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LDaau3BWz9CKHKLfpGVU2B"
```

---

### Task 10: Final verification

**Files:** none (verification only).

**Interfaces:** none — this task only runs and reports on everything built in Tasks 1-9.

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all tests pass or skip (CUDA-only tests skip on this machine); zero failures or errors.

- [ ] **Step 2: Lint the new code**

Run: `.venv/bin/python -m pip show ruff >/dev/null 2>&1 || /Users/macariofang/.local/bin/uv pip install --python .venv/bin/python ruff`

Run: `.venv/bin/python -m ruff check src/sharp_spatialize tests/sharp_spatialize`
Expected: no errors. Fix anything reported (matching the repo's existing `ruff` config in `pyproject.toml`: line length 100, Google-style docstrings) before moving on.

- [ ] **Step 3: Confirm the console scripts still work end-to-end (argument parsing only)**

Run:
```bash
.venv/bin/sharp-spatialize --help
.venv/bin/sharp-spatialize-validate --help
```
Expected: both print usage text without error.

- [ ] **Step 4: Write a short status note for the branch**

Append to `docs/superpowers/plans/2026-09-02-spatial-photo-authoring.md` (this file), at the very end, a "## Status" section:

```markdown
## Status

All 10 tasks implemented and committed on branch `spatial-photo-authoring`.
Everything not requiring CUDA has been run and passes on this machine
(camera geometry, HDF5 I/O, validation logic, inference plumbing, API
wiring, CLI argument handling). `render.py`'s real-rendering path and
`tests/sharp_spatialize/test_e2e.py` are skip-marked here and still need to
be run on a CUDA-equipped machine (see Task 8, Step 4) before Stage 1
acceptance criteria #1-4, #9, and #10 (spec §15) can be marked verified.
```

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-09-02-spatial-photo-authoring.md
git commit -m "sharp_spatialize: record verification status

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LDaau3BWz9CKHKLfpGVU2B"
```

- [ ] **Step 6: Hand off**

Report to the user: what was implemented, that the full non-CUDA test suite passes, and that end-to-end CUDA verification (Task 8) is the one remaining step before this can be called done against the spec's acceptance criteria. Do not merge to `main` or open a PR without being asked.
