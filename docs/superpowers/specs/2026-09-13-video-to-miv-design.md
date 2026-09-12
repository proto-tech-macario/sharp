# Stage 2 Video → 9-View Spatial Sequence → MIV — Design

Date: 2026-09-13
Source requirements: "IRU Spatial Media — Stage 2 Engineering Design Specification" (PDF, provided by user)
Builds on: `docs/superpowers/specs/2026-09-02-spatial-photo-authoring-design.md` (Stage 1)

## 1. Objective

Extend the validated Stage 1 pipeline (`sharp_spatialize`: one image → 9-view
`spatial_photo.h5`) to a video clip: decode frames in presentation order, run
the *unchanged* Stage 1 pipeline on every selected frame, store the temporally
ordered 9-view sequence in HDF5, and package it as an MIV (ISO/IEC 23090-12)
bitstream that an independent MIV decoder can parse and decode.

Engineering rule (spec §40): **do not change Stage 1**. Stage 2 wraps it
(frames in) and follows it (MIV out). No file under `src/sharp_spatialize/` or
`src/sharp/` is modified.

## 2. Key decisions

1. **MIV encoder = TMIV v24.0** (MPEG's reference MIV software, BSD-3), driven
   through its `scripts/encode.py` (TmivEncoder → VVenC per atlas component →
   TmivMultiplexer). The final `.miv` file is TMIV's multiplexed V3C sample
   stream: MIV metadata plus in-band VVC video sub-bitstreams. The independent
   decoder is `TmivDecoder` (integrated VVdeC). Hand-writing an MIV bitstream
   writer was rejected: it could not credibly satisfy "an independent MIV
   decoder can parse/decode the file".
2. **Per-frame cameras are preserved** (user decision). Stage 1 recomputes the
   orbit focus depth per frame, so R/C of the 8 outer views change every frame
   (spec §17: `R[T,9,3,3]`, `C[T,9,3]`). Stock TMIV reads cameras once per
   sequence in three places (`Encoder_main.cpp` loads the sequence config at
   frame 0 only; `Encoder.cpp` calls `prepareSequence` once; 
   `ViewOptimizerStage.cpp` caches the frame-0 view list). A small patch
   (`third_party/tmiv/sharp-per-frame-cameras.patch`) makes all three carry each
   frame's poses; TMIV's existing `EncodeMiv` then emits standard
   `miv_view_params_update_extrinsics` in each frame's common atlas frame. The
   encoder config uses `interPeriod = 1` so every frame gets its own common atlas
   frame. Intrinsics and depth quantization must stay constant (enforced by the
   patch and by the adapter).
3. **All 9 views are coded in full**: `NoViewOptimizer` + `NoPruner`
   (as TMIV's `one_view` test config), so the MIV file carries the complete
   9-view RGB+depth representation (spec §38 "intended multi-view/depth
   representation") and the decoder reconstructs exactly `v0..v8`.
4. **Video decoding = PyAV** (`av`), which exposes PTS/time base, frame count,
   pixel format and colour metadata, decodes in presentation order, and ships
   H.264/H.265 decoders in its wheels. Supported codecs live in one registry
   set so formats can be added later (spec §8).
5. **Frames reach Stage 1 as lossless PNGs** in a work directory and go through
   `sharp_spatialize.api.generate_spatial_photo` — the exact Stage 1 entry
   point, with `cache_predictor=True` so the checkpoint loads once per run.
   PNG frames carry no EXIF, so every frame gets Stage 1's default focal length
   and K is constant across the clip.
6. **Resume cache = Stage 1 files.** Each finished frame is saved as a Stage 1
   `spatial_photo.h5` under `<work>/frames/NNNNNN.h5` (Stage 2 fields added as
   extra attrs). A JSON journal records done/failed frames and errors
   (spec §32). Resuming skips frames whose cache file loads and validates.
7. **Development split.** Everything except the Stage 1 render (CUDA) and TMIV
   (C++ build) is unit-tested anywhere with synthetic data. TMIV integration
   tests run where `SHARP_TMIV_DIR` points at a patched TMIV build; CUDA e2e
   tests are skip-marked, as in Stage 1.

## 3. Architecture

```
input.mp4
  │ video_io.probe_video / iter_frames        (PyAV; presentation order; PTS kept)
  ▼
scheduler.FrameSelection                     (--start-frame/--end-frame/--max-frames)
  │
  ▼
runner.process_frames ── spatializer.Stage1Spatializer ──► sharp_spatialize (unchanged)
  │   workers, journal, per-frame Stage 1 cache, reorder buffer → presentation order
  ▼
SpatialFrame[t]  (contract.py: timestamp + 9 × SpatialView{rgb,depth,mask,K,R,C})
  ├──► sequence_io.SequenceWriter → stage2_sequence.h5   (Mode A; or --dump-hdf5)
  ▼
miv.MIVEncoder.begin / encode_frame / finalize          (no SHARP / torch imports)
  │   camera.py  OpenCV K,R,C → TMIV Focal/Principle_point/Position/Rotation
  │   depth.py   metres → 16-bit normalized disparity, 0 = invalid
  │   yuv.py     RGB8 → yuv420p10le (BT.709 limited)
  │   tmiv_config.py  per-frame sequence JSON + encoder/mux/QP configs
  │   tmiv.py    encode.py → TmivEncoder → vvencFFapp → TmivMultiplexer
  ▼
output.miv  (+ output.miv.json manifest, output.report.json/.md)
```

Mode A (two-step, default, recommended for debugging): frames → HDF5 → encoder
reads the HDF5 back frame-by-frame. Mode B (streaming): frames → encoder
directly (HDF5 only if `--dump-hdf5`). Both feed `MIVEncoder.encode_frame` the
same `SpatialFrame`s, so encoder inputs are byte-identical (tested).

## 4. Modules (`src/sharp_video/`)

| Module | Responsibility | Imports SHARP/torch |
|---|---|---|
| `contract.py` | `SpatialView`, `SpatialFrame`, `SequenceInfo`; Stage 1 ↔ contract conversion | no |
| `sequence_io.py` | temporal HDF5 writer (streaming, atomic) and lazy reader | no |
| `video_io.py` | probe + decode to RGB with PTS | no |
| `scheduler.py` | frame selection | no |
| `spatializer.py` | frame → Stage 1 → `SpatialFrame` + timings | yes (lazily) |
| `journal.py` | resume / failure journal | no |
| `runner.py` | workers, cache, reorder, failure policy | no (takes a callable) |
| `metrics.py` | temporal quality (flicker, instability, geometry change) | no |
| `validation.py` | sequence validation (Stage 1 checks per frame + temporal) | no |
| `inspect.py` | contact sheets, depth maps, camera tables | no |
| `report.py` | performance report (JSON + Markdown) | no |
| `pipeline.py` | Mode A / Mode B orchestration | no (spatializer injected) |
| `cli.py` | `sharp_video_to_miv`, `sharp-video-validate`, `sharp-video-inspect`, `sharp-miv-encode`, `sharp-miv-validate` | lazily |
| `miv/*` | standalone MIV encoder, decoder driver, MIV validation | **no** (tested) |

## 5. Temporal HDF5 (spec §15)

```
stage2_sequence.h5
/metadata            (group; attrs) format_version="2.0", width, height, fps,
                     frame_count, view_count=9, view_layout="3x3",
                     depth_unit="meter", depth_definition="camera_z",
                     invalid_depth_value=0.0, coordinate_system="OpenCV",
                     camera_convention="R is world->camera; C is the camera's world
                     position; x_cam = R @ (x_world - C)", time_base, source_filename,
                     horizontal_angle, vertical_angle, model_name, model_version
/frames/NNNNNN       (group; attrs) timestamp [s], pts, source_frame_index
    /view_00 … /view_08   rgb [H,W,3] u8, depth [H,W] f32, mask [H,W] u8,
                          K [3,3] f32, R [3,3] f32, C [3] f32
```

Per-view arrays and conventions are exactly Stage 1's; `/frames/NNNNNN` is
numbered by output order (0..T-1). The writer rejects non-increasing
timestamps, wrong view count, or resolution changes; it writes to `.tmp` and
renames on successful close.

## 6. MIV mapping

- **Axes.** TMIV cameras are x forward, y left, z up (`Projector.h`:
  `u = cx − fx·y/x`). With `P = [[0,0,1],[−1,0,0],[0,−1,0]]` (OpenCV → TMIV):
  `Position = P·C`, camera→world rotation `Rc2w = P·Rᵀ·Pᵀ`; TMIV `Rotation` =
  `[yaw, pitch, roll]` degrees with `Rc2w = Rz(yaw)·Ry(pitch)·Rx(roll)`
  (`euler2quat`: `qy·qp·qr`; `AffineTransform` confirms orientation is
  camera→world). World = Stage 1's reference-camera frame.
- **Intrinsics.** `Focal=[fx,fy]`, `Principle_point=[cx,cy]`, `Resolution=[W,H]`.
- **Depth.** 16-bit geometry (`BitDepthDepth=16`, `yuv420p16le`, luma used):
  `sample = round((1/z − 1/far)/(1/near − 1/far) · 65535)` clamped to
  `[1, 65535]`; invalid pixels → 0 with `HasInvalidDepth=true`. `Depth_range =
  [near, far]` is one fixed, configurable range per run (default 0.1–1000 m) so
  depth quantization never changes mid-sequence; clamped pixel counts are
  reported.
- **Texture.** `BitDepthColor=10`, `yuv420p10le`, BT.709 limited range,
  2×2 chroma averaging. Output views must have even width/height; the CLI
  defaults the output resolution to the source rounded down to a multiple of 8.
- **Timing.** MIV carries a frame rate (`Fps`), not per-frame timestamps. The
  encoder records every frame's PTS/timestamp in the `output.miv.json`
  manifest, checks the clip is constant-frame-rate within ½ frame, and flags VFR
  input in the manifest and report.

## 7. Validation (spec §33, §37.5)

- **Per frame:** Stage 1 `validation` on each frame (shapes, metadata, R
  orthonormality, depth sanity, V3→V5 reprojection).
- **Sequence:** frame count, strictly increasing timestamps, CFR check, constant
  resolution/K, 9 views per frame.
- **Temporal (measure, not fix):** per view RGB flicker (mean |Δrgb| on pixels
  valid in both frames), depth flicker (median |Δz|/z), view instability (|ΔC|,
  rotation delta angle), geometry change (median centre-view depth delta).
- **MIV:** `TmivParser` structural dump; `TmivDecoder` reconstructs v0..v8 per
  frame plus per-frame sequence configs. Checks: frame count, fps, duration,
  view names/count/resolution, per-frame K/R/C vs reference (after inverse
  axis conversion), depth correspondence (median relative error on pixels valid
  in both), RGB PSNR. File size and average bitrate.

## 8. Performance report (spec §35)

Input resolution/fps/duration/frame count; per-frame SHARP+3DGS time (Stage 1
runs SHARP and Gaussian unprojection in one `infer` call — reported together
and labelled so), 9-view render time, Stage 1 validation time, total Stage 1
time, peak GPU memory (`torch.cuda.max_memory_allocated`), MIV encode time,
MIV size, average bitrate, total time, average processing FPS, peak RSS.

## 9. Error handling

- Unsupported codec/container → `UnsupportedVideoError` before any processing.
- A frame failing Stage 1 is recorded in the journal (index, PTS, error) and
  processing continues; the run then fails with `FrameFailuresError` listing
  every failed frame — no MIV is produced with a missing frame (spec §32).
- TMIV missing/unpatched → `TmivNotFoundError` naming `SHARP_TMIV_DIR` and
  `scripts/build_tmiv.sh`; any TMIV subprocess failure surfaces its log path.
- All output files are written atomically (`.tmp` + rename).

## 10. Deliverables (spec §37)

| Item | Where |
|---|---|
| Video-to-MIV application | `sharp_video_to_miv` |
| Temporal HDF5 | `sequence_io.py`, `--dump-hdf5`, Mode A |
| Standalone MIV encoder | `sharp_video.miv.MIVEncoder`, `sharp-miv-encode` |
| TMIV patch + build | `third_party/tmiv/`, `scripts/build_tmiv.sh` |
| Validation tools | `sharp-video-validate`, `sharp-video-inspect`, `sharp-miv-validate` |
| Performance report | `<output>.report.json` / `.md` |
| MIV test files | `scripts/stage2/run_test_matrix.py` over the 5 categories of spec §34 (clips supplied by the user; a CUDA box + TMIV produce them — not fabricated here) |
| Docs | `docs/stage2_sequence_format.md`, `docs/stage2_miv.md`, README section |

## 11. Out of scope (spec §6)

MIV decoder/renderer implementation (TMIV's is used only to validate), IRU
integration, temporal optimization of any kind, SHARP changes, NPU/mobile work,
ISOBMFF (`.mp4`) packaging of the V3C stream.
