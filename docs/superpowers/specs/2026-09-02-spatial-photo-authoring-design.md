# Stage 1 Spatial Photo Authoring — Design

Date: 2026-09-02
Source requirements: "Stage 1 Spatial Photo Authoring Specification" v1.0 (PDF, provided by user)

## 1. Objective

Build `sharp_spatialize`, a new package on top of this repo's SHARP model, that
converts one RGB image into a 9-view spatial representation (`spatial_photo.h5`):
9 RGB views, 9 depth maps, 9 validity masks, and per-view camera metadata (K, R, C).
The output file must be readable by downstream consumers without SHARP or the
3D Gaussian representation installed.

This mirrors the source spec 1:1 (functional scope, camera layout, HDF5 schema,
metadata, validation, deliverables, acceptance criteria) — see section 10 for the
full traceability table. This document additionally records the concrete
architecture, module boundaries, and geometry decisions made to implement it in
*this* repository.

## 2. Key decisions (resolved during brainstorming)

1. **Render backend: CUDA/gsplat only, no CPU/MPS fallback.** The repo's existing
   Gaussian rasterizer (`sharp.utils.gsplat.GSplatRenderer`, used by `sharp render`)
   requires CUDA — `cli/render.py` already hard-fails without it. This design keeps
   that constraint rather than building a second rasterizer. Development and
   camera/HDF5/validation testing happen on this machine (no CUDA); real end-to-end
   rendering runs wherever a CUDA GPU is available.
2. **Camera rig geometry: orbit around a fixed focus point**, not pure pan/tilt
   rotation-in-place. Each of the 9 cameras sits on a small arc at the scene's focus
   depth and looks back at the same focus point the reference camera faces. This
   produces genuine parallax/occlusion between views (consistent with "depth =
   camera-space Z" meaning something different per view), and reuses the geometry
   already implicit in `sharp.utils.camera.PinholeCameraModel` / eye trajectories.
3. **Package location: new top-level `src/sharp_spatialize/`**, not folded into
   `webui/` or `src/sharp/`. It depends on `sharp` (imports its predictor, Gaussian
   utils, `io`, `camera`, `gsplat`) but ships its own console-script entry points
   (`sharp-spatialize`, `sharp-spatialize-validate`) and Python API, matching the
   spec's literal CLI example and keeping "isolated behind an interface" (spec §4)
   honest — nothing outside `inference.py` touches `Gaussians3D` or the predictor.

## 3. Architecture

```
input.jpg
   │
   ▼
inference.py    scene = infer(image)                         [only module touching SHARP internals]
   │
   ▼
cameras.py      rig = build_camera_rig(scene, angle_deg)      → 9× (K, R, C), OpenCV convention
   │
   ▼
render.py       rgb[9,H,W,3], depth[9,H,W], mask[9,H,W]       [gsplat, requires CUDA]
   │              = render_views(scene, rig)
   ▼
hdf5_io.py      save(path, rgb, depth, mask, K, R, C, meta)   [no SHARP/torch import]
   │
   ▼
validation.py   validate(...)  — runs before declaring success; also standalone
   │
   ▼
spatial_photo.h5
```

Each module has one job and can be understood/tested without the others:

| Module | Depends on | CUDA required | Testable here (no CUDA) |
|---|---|---|---|
| `inference.py` | `sharp.models`, `sharp.utils.io` | no (cpu/mps/cuda) | only with a checkpoint download |
| `cameras.py` | `sharp.utils.camera`, `sharp.utils.gaussians.Gaussians3D` (type only) | no | yes, fully |
| `render.py` | `sharp.utils.gsplat` | **yes** | no (skip-marked) |
| `hdf5_io.py` | `h5py`, `numpy` | no | yes, fully |
| `validation.py` | `hdf5_io`, `numpy` | no | yes, fully |
| `api.py` | all of the above | yes (end-to-end) | partially (mock render) |
| `cli.py` | `api`, `validation` | yes (for `sharp-spatialize`) | `sharp-spatialize-validate` yes |

## 4. Module details

### 4.1 `inference.py`

```python
@dataclass
class SceneBundle:
    gaussians: Gaussians3D
    f_px: float
    width: int
    height: int
    device: str

def infer(image_path: Path, checkpoint_path: Path | None = None, device: str = "default") -> SceneBundle:
    ...
```

Steps: `sharp.utils.io.load_rgb` → build/load predictor (same pattern as
`webui/sharp_runner.py::_load_predictor` / `cli/predict.py`) → one forward pass →
`unproject_gaussians` into metric world space. Records `original width`, `original
height`, and the source filename go into metadata one level up, in `api.py`, since
`inference.py` only knows about pixel data, not the caller's path.

### 4.2 `cameras.py`

```python
DEFAULT_LAYOUT: tuple[tuple[float, float], ...]  # (h_deg, v_deg) for V0..V8, per spec table

@dataclass
class CameraPose:
    K: np.ndarray  # 3x3
    R: np.ndarray  # 3x3, world->camera, OpenCV convention
    C: np.ndarray  # (3,), world position

def camera_at(scene: SceneBundle, h_angle_deg: float, v_angle_deg: float,
              output_width: int, output_height: int) -> CameraPose: ...

def build_camera_rig(scene: SceneBundle, angle_deg: float,
                      output_width: int, output_height: int) -> list[CameraPose]:
    """Returns [V0..V8] using angle_deg for both axes, per the spec's fixed table:
    V0=(-a,+a) V1=(0,+a) V2=(+a,+a)
    V3=(-a, 0) V4=(0, 0) V5=(+a, 0)
    V6=(-a,-a) V7=(0,-a) V8=(+a,-a)
    """
```

Geometry inside `camera_at`:

1. `focus_depth` = 10th-percentile depth of `scene.gaussians.mean_vectors` in the
   reference camera frame — the same heuristic `PinholeCameraModel` uses today.
2. `eye_pos = (focus_depth * tan(h_angle), focus_depth * tan(v_angle), 0)` in
   reference-camera coordinates (X=right, Y=down per spec §6).
3. `look_at = (0, 0, focus_depth)` — every view orbits the same point.
4. `R, C = sharp.utils.camera.create_camera_matrix(eye_pos, look_at, world_up, inverse=True)`
   (`R` = the 3x3 rotation block, `C = eye_pos`).
5. `K` = reference intrinsics, rescaled by `output_width/height` if they differ
   from `scene.width/height`.

`(0°, 0°)` (V4) collapses to the reference camera's own pose, satisfying "V4 is
the reference camera." `angle_deg` is a single configurable number applied to
both axes per the spec's table (the initial implementation supports at least
5°, 10°, 15°, enforced as a CLI choice **plus** any custom float accepted by
the Python API, so `--angle` isn't artificially restrictive beyond the spec's
minimum bar).

`camera_at` is reused directly for acceptance criterion #10 ("an arbitrary
intermediate camera position within the configured viewing range can be
rendered for testing") — any `(h, v)` pair within `[-angle_deg, +angle_deg]`
works, not just the 9 grid points.

### 4.3 `render.py`

```python
def render_views(scene: SceneBundle, rig: list[CameraPose],
                  output_width: int, output_height: int,
                  mask_alpha_threshold: float = 0.5,
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (rgb[9,H,W,3] uint8, depth[9,H,W] float32, mask[9,H,W] uint8)."""
```

- Raises `RuntimeError` immediately if `not torch.cuda.is_available()`, before
  constructing `GSplatRenderer` — mirrors `cli/render.py`'s existing guard, so
  failure is a clear message, not a stack trace from inside gsplat.
- One `GSplatRenderer(color_space=...)` instance, called once per view (batch
  size 1, matching existing usage in `cli/render.py::render_gaussians`) — the
  Gaussians are only ever forward-passed once through SHARP (in `inference.py`);
  this stage is pure rendering of the same fixed scene from 9 poses, so "the
  system shall not run SHARP separately for each view" is inherently satisfied.
- `depth` = gsplat's alpha-normalized depth output (`RenderingOutputs.depth`),
  which is already camera-space Z.
- `mask` = `alpha > mask_alpha_threshold`. Where invalid, depth is forced to
  `0.0` (never NaN/Inf) — satisfies validation rule "no unexpected NaN/Inf" and
  "invalid pixels correctly represented by the mask" unambiguously.

### 4.4 `hdf5_io.py`

No SHARP or torch import — pure `numpy` + `h5py`. This is the file a downstream
consumer with no SHARP install would vendor/reference.

```python
@dataclass
class SpatialPhotoResult:
    rgb: np.ndarray     # [9,H,W,3] uint8
    depth: np.ndarray   # [9,H,W] float32
    mask: np.ndarray    # [9,H,W] uint8
    K: np.ndarray       # [9,3,3] float32
    R: np.ndarray       # [9,3,3] float32
    C: np.ndarray       # [9,3] float32
    metadata: dict

    def save(self, path: Path) -> None: ...

def save(path: Path, result: SpatialPhotoResult) -> None: ...
def load(path: Path) -> SpatialPhotoResult: ...
```

Writes exactly the schema in spec §10/§11: `/rgb`, `/depth`, `/mask`,
`/camera/K`, `/camera/R`, `/camera/C` datasets with the specified shapes/dtypes,
and file-level attrs: `format_version, model_name, model_version, source_width,
source_height, output_width, output_height, num_views, view_layout,
horizontal_angle, vertical_angle, coordinate_system, pose_convention,
depth_definition, depth_unit`. Values: `model_name="SHARP"`, `depth_unit="meter"`
(SHARP's Gaussians are metric per the repo's own README), `depth_definition=
"camera_z"`, `coordinate_system="OpenCV"`, `pose_convention` documents that
`R` is world→camera and `C` is the camera's world position (i.e. `x_cam = R @
(x_world - C)`).

Written to `<path>.tmp` then `os.replace`d into place — a crash mid-write never
leaves a corrupt `spatial_photo.h5` on disk.

### 4.5 `api.py`

```python
def generate_spatial_photo(image_path: str | Path, angle_deg: float = 10,
                            output_width: int | None = None,
                            output_height: int | None = None,
                            checkpoint_path: Path | None = None,
                            device: str = "default") -> SpatialPhotoResult:
    scene = inference.infer(image_path, checkpoint_path, device)
    w = output_width or scene.width
    h = output_height or scene.height
    rig = cameras.build_camera_rig(scene, angle_deg, w, h)
    rgb, depth, mask = render.render_views(scene, rig, w, h)
    K, R, C = _stack(rig)
    metadata = _build_metadata(image_path, scene, angle_deg, w, h)
    result = SpatialPhotoResult(rgb, depth, mask, K, R, C, metadata)
    validation.validate_in_memory(result)   # raises on failure — nothing bad gets saved
    return result
```

`result.save("spatial_photo.h5")` is the exact call shape from spec §12.
`result.rgb/.depth/.mask/.K/.R/.C` are the exact attribute names required.

### 4.6 `cli.py`

Two Click commands, both `[project.scripts]` entries:

- `sharp-spatialize --input image.jpg --angle 10 --output spatial_photo.h5
  [--width W] [--height H] [--checkpoint PATH] [--device DEV]` — the spec's
  literal example invocation.
- `sharp-spatialize-validate spatial_photo.h5` — standalone validator, no SHARP
  import needed at all (only `hdf5_io` + `validation` + `numpy`/`h5py`), for
  downstream teams or CI to sanity-check a file without a GPU or SHARP install.

### 4.7 `validation.py`

```python
def validate_in_memory(result: SpatialPhotoResult) -> None: ...   # raises ValidationError
def validate_file(path: Path) -> ValidationReport: ...             # collects all failures, doesn't raise
```

Checks (spec §13):
- **File**: all required datasets exist with correct shapes; all required attrs present.
- **Camera**: for every view, `‖RᵀR − I‖_max < 1e-4` and `|det(R) − 1| < 1e-4`.
- **Depth**: no NaN/Inf anywhere; `depth[mask == 1] > 0`; every `mask == 0` pixel's
  depth is exactly `0.0` (round-trips render.py's convention).
- **Reprojection**: pick a source/target pair (e.g. V3→V5), unproject the source
  view's valid pixels to 3D using its `depth/K/R/C`, reproject into the target
  camera, and compare against the target's actual RGB in the overlapping region
  (mean absolute error under a threshold on pixels where both masks are valid).
  This is the "automated test that reproject a source view into another camera"
  requirement, runnable both as a library function and as a pytest test.

`validate_in_memory` raises (used inside `generate_spatial_photo`, so a failing
scene never gets saved); `validate_file`/`sharp-spatialize-validate` report all
failures at once for diagnosability.

## 5. Error handling

- Corrupt/unsupported image → `inference.infer` surfaces `sharp.utils.io.load_rgb`'s
  error directly, no partial state created.
- No CUDA at render time → explicit `RuntimeError("sharp-spatialize requires a
  CUDA GPU for view rendering (gsplat)")`, raised before constructing the
  renderer.
- Degenerate geometry (no positive-depth Gaussians in the reference view, so the
  focus-depth quantile is undefined) → explicit `ValueError` from `cameras.py`,
  not a silent NaN propagating into the rig.
- `generate_spatial_photo`: validation failure raises before `save()` is ever
  reachable — an invalid scene is never persisted.
- `hdf5_io.save`: atomic write via temp file + rename.

## 6. Testing strategy

Split along the CUDA boundary:

**Runs anywhere (this machine included), no CUDA/model needed:**
- `cameras.py`: angle grid matches the spec's V0..V8 table exactly; every `R` is
  orthonormal with `det=1` by construction; V4 == reference camera pose;
  `camera_at` agrees with `build_camera_rig`'s per-slot output at grid angles.
- `hdf5_io.py`: round-trip save→load on synthetic arrays, byte-for-byte
  comparison, all required datasets/attrs present.
- `validation.py`: unit tests against hand-built good/bad synthetic
  `SpatialPhotoResult`s (bad `R`, NaN depth, inconsistent mask, mismatched
  shapes) to prove each check actually fires and only fires when it should.

**Requires CUDA — `@pytest.mark.skipif(not torch.cuda.is_available())`,
lives in this repo, runs unmodified wherever a GPU is available:**
- End-to-end: real image → `generate_spatial_photo` → 9 real rendered views →
  `validate_in_memory` passes → reprojection MAE under threshold (spec §13's
  "at least one automated test").
- Acceptance criterion #10: `camera_at` at an arbitrary intermediate angle
  (not one of the 9 grid points) renders without error.

## 7. Dependencies / packaging changes

- Add `h5py` to `[project.dependencies]` in `pyproject.toml`.
- Add a new `sharp_spatialize` entry under `[tool.setuptools.packages.find]`
  (already covered by `where = ["src"]` since it's a sibling of `sharp/`).
- Add to `[project.scripts]`:
  ```
  sharp-spatialize = "sharp_spatialize.cli:generate_cli"
  sharp-spatialize-validate = "sharp_spatialize.cli:validate_cli"
  ```

## 8. Deliverables mapping (spec §14)

| Spec item | This design |
|---|---|
| Source code | `src/sharp_spatialize/` |
| Executable/CLI | `sharp-spatialize`, `sharp-spatialize-validate` |
| Python API | `generate_spatial_photo`, `SpatialPhotoResult`, `load_spatial_photo` |
| SHARP integration | `inference.py` |
| 9-view generator | `cameras.py` |
| RGB/depth renderer | `render.py` |
| HDF5 writer/reader | `hdf5_io.py` |
| Validation script | `validation.py` + `sharp-spatialize-validate` |
| HDF5 format documentation | `docs/spatial_photo_format.md` |
| Example input and output files | `examples/` — sample input shipped now; the generated `.h5` requires a CUDA run to produce, so a placeholder + the exact command to (re)generate it ships instead of a fabricated file |

## 9. Acceptance criteria mapping (spec §15)

| # | Criterion | Verified by |
|---|---|---|
| 1 | JPEG/PNG processed successfully | `inference.infer`, exercised by e2e test |
| 2 | One SHARP 3D representation generated | `inference.infer` (single forward pass) |
| 3 | Exactly 9 views | `cameras.build_camera_rig` returns a fixed 9-tuple; unit-tested |
| 4 | Each view has RGB + depth | `render.render_views` shape contract; unit-tested (shapes), e2e (real values) |
| 5 | Each view has valid K, R, C | `validation.py` R-orthonormality/det check |
| 6 | Common coordinate system across views | single `focus_depth`/reference frame in `cameras.py`; all poses expressed relative to it |
| 7 | 9-view result in one HDF5 file | `hdf5_io.save` |
| 8 | HDF5 readable without SHARP/3DGS | `hdf5_io.py` has zero SHARP/torch imports; `sharp-spatialize-validate` proves this in practice |
| 9 | Camera/depth reprojection validation passes | `validation.py` reprojection test, e2e |
| 10 | Arbitrary intermediate camera position renders | `cameras.camera_at` + e2e test at a non-grid angle |

## 10. Traceability to source spec

This design implements the source PDF spec section-for-section:
§1 Objective → §3 here. §2 Functional scope → §4 module list. §3 Input → handled
in `inference.py`/`api.py` (records width/height/filename in metadata). §4 SHARP
Processing → §4.1 (`inference.py`, `scene = sharp.infer(image)` boundary).
§5 Camera Generation → §4.2 (`cameras.py`, exact V0-V8 table, configurable angle,
5/10/15° supported at minimum). §6 Camera Parameters → K/R/C in
`CameraPose`/HDF5 schema, OpenCV convention (X right, Y down, Z forward),
documented in HDF5 metadata via `pose_convention`. §7 RGB Rendering, §8 Depth
Rendering, §9 Validity Mask → §4.3 (`render.py`). §10 HDF5 Output, §11 HDF5
Metadata → §4.4 (`hdf5_io.py`). §12 Software Interface → §4.5/§4.6
(`api.py`/`cli.py`). §13 Validation → §4.7 (`validation.py`). §14 Deliverables →
§8 here. §15 Acceptance Criteria → §9 here. §16 Output Definition → matches the
architecture diagram in §3 here exactly.

## 11. Explicitly out of scope (per source spec §2)

No video processing, no mobile-NPU optimization, no CPU/MPS rendering fallback
(per decision 1 above).
