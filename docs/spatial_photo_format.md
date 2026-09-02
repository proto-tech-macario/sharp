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

Note: because this uses the OpenCV convention (Y down), a positive
`vertical_angle` places that camera *below* the reference viewpoint, not
above -- the grid above shows index order (V0..V8), not literal vertical
position.

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
