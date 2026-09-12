# Stage 2 temporal spatial sequence (`stage2_sequence.h5`)

A spatial sequence is the Stage 1 spatial photo extended over time: one
9-view spatial frame per selected video frame, in presentation order. It is
the intermediate/debug representation of Stage 2 and the input of the
standalone MIV encoder (`sharp-miv-encode`). It is readable with plain `h5py`
and `numpy` — no SHARP, GPU, or 3D Gaussian code. Produced by
`sharp_video_to_miv` (two-step mode, or `--dump-hdf5`); written by
`src/sharp_video/sequence_io.py`.

## Reading a file

```python
import h5py

with h5py.File("stage2_sequence.h5", "r") as f:
    meta = dict(f["metadata"].attrs)          # width, height, fps, frame_count, ...
    frame = f["frames/000000"]
    t = frame.attrs["timestamp"]              # seconds, from the source PTS
    rgb = frame["view_04/rgb"][:]             # [H, W, 3] uint8, the centre view
    depth = frame["view_04/depth"][:]         # [H, W] float32, metres
    K, R, C = (frame[f"view_04/{k}"][:] for k in ("K", "R", "C"))
```

Or, with `sharp_video` installed:

```python
from sharp_video.sequence_io import SequenceReader

reader = SequenceReader("stage2_sequence.h5")   # lazy: one frame in memory at a time
for frame in reader:                            # SpatialFrame, in presentation order
    print(frame.timestamp, frame.views[4].depth.max())
```

## Layout

```
/metadata                 group, attributes only
/frames/000000            one group per spatial frame, numbered 0..T-1 in output order
    attrs: timestamp, pts, source_frame_index, + Stage 1 per-frame metadata
    /view_00 ... /view_08
        rgb    [H, W, 3]  uint8
        depth  [H, W]     float32   camera-space Z, metres, 0.0 where invalid
        mask   [H, W]     uint8     1 = valid, 0 = invalid
        K      [3, 3]     float32   intrinsics
        R      [3, 3]     float32   world -> camera rotation
        C      [3]        float32   camera position in world coordinates
/frames/000001
...
```

Every per-view array, and the meaning of every value in it, is exactly the
Stage 1 spatial photo's (`docs/spatial_photo_format.md`), split per view.

### `/metadata` attributes

| Attribute | Meaning |
|---|---|
| `format_version` | `"2.0"` |
| `width`, `height` | view size in pixels (identical for every view and frame) |
| `fps` | source average frame rate |
| `frame_count` | number of spatial frames (`T`) |
| `view_count` | `9` |
| `view_layout` | `"3x3"`, V0..V8 row-major, V4 = centre/original viewpoint |
| `depth_unit` | `"meter"` |
| `depth_definition` | `"camera_z"` |
| `invalid_depth_value` | `0.0` (always paired with `mask == 0`) |
| `coordinate_system` | `"OpenCV"` (X right, Y down, Z forward) |
| `camera_convention` | `R` world→camera, `C` camera centre, `x_cam = R @ (x_world - C)` |
| `time_base` | source stream time base, e.g. `"1/30000"` |
| `source_filename` | input video file name |
| `horizontal_angle`, `vertical_angle` | virtual-camera angle in degrees (Stage 1 `angle_deg`) |
| `source_width`, `source_height`, `source_codec` | input video properties |

### Per-frame attributes

| Attribute | Meaning |
|---|---|
| `timestamp` | presentation time in seconds (`pts × time_base`), retained from the source — a clip starting at frame 30 starts at ~1 s |
| `pts` | presentation timestamp in `time_base` units (`-1` if unknown) |
| `source_frame_index` | index of the source frame in presentation order (`-1` if unknown) |
| Stage 1 metadata | `model_name`, `model_version`, `output_width`, ... as written by Stage 1 |

All 9 views of a frame share its timestamp (spec §27). Timestamps are strictly
increasing; the writer refuses anything else.

## World frame

Each frame's world frame is Stage 1's: the centre (V4) camera of *that* frame.
Frames are spatialized independently (spec §28), so there is no shared world
frame across time, and the 8 outer cameras move from frame to frame as Stage 1's
focus depth follows the scene. `sharp-video-validate` measures that motion.

## Tools

```
sharp-video-validate stage2_sequence.h5 [--temporal-json temporal.json]
sharp-video-inspect  stage2_sequence.h5 --frame 0 --frame 50 --out inspect/
sharp-miv-encode     stage2_sequence.h5 -o output.miv
```

`sharp-video-validate` runs the Stage 1 validator on every frame (shapes,
metadata, rotation validity, depth sanity, reprojection) plus the sequence
checks (frame count, increasing timestamps, constant intrinsics), and reports
temporal metrics: RGB flicker, depth flicker, camera translation/rotation
between frames, and centre-view depth change.
