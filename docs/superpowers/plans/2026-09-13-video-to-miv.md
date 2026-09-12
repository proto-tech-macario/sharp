# Stage 2 Video → MIV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `sharp_video`: video → per-frame unchanged Stage 1 → temporal 9-view HDF5 → MIV (TMIV) with validation tools and a performance report, per the Stage 2 spec.

**Architecture:** New package `src/sharp_video/` wrapping `sharp_spatialize` (untouched). The MIV encoder (`sharp_video/miv/`) consumes only the `SpatialFrame` contract (numpy), converts to TMIV inputs (YUV + per-frame camera JSON) and drives a patched TMIV v24.0 via its `encode.py`.

**Tech Stack:** Python 3.13, numpy, h5py, PyAV (`av`), Click, pytest; TMIV v24.0 + VVenC/VVdeC (C++, built by `scripts/build_tmiv.sh`).

**Spec:** `docs/superpowers/specs/2026-09-13-video-to-miv-design.md` (implements the "IRU Spatial Media — Stage 2 Engineering Design Specification" PDF).

## Global Constraints

- No changes under `src/sharp_spatialize/` or `src/sharp/` (spec §40).
- Exactly 9 views per frame, Stage 1 layout V0..V8, V4 = centre; camera angle parameterized, default 10°.
- K/R/C, OpenCV axes, `x_cam = R @ (x_world - C)`, depth = camera Z in metres, invalid depth = 0.0 with mask 0 — identical to Stage 1.
- Frames processed independently; output strictly in presentation order; no silent frame omission.
- `sharp_video.miv`, `sharp_video.contract`, `sharp_video.sequence_io` must import with `torch`, `sharp`, `sharp_spatialize` unavailable.
- CLI: `sharp_video_to_miv --input input.mp4 --output output.miv [--dump-hdf5 X.h5] [--start-frame] [--end-frame] [--max-frames] [--worker-count] [--camera-config] [--output-resolution]`.
- MIV API shape: `MIVEncoder(config)`, `.begin(fps, width, height, num_views=9)`, `.encode_frame(timestamp, views)`, `.finalize(path)`.
- Tests: `.venv/bin/python -m pytest tests/sharp_video -q`; TMIV tests gated on `SHARP_TMIV_DIR`; CUDA tests skip without CUDA.

---

### Task 1: Contract + temporal HDF5 (`contract.py`, `sequence_io.py`)

**Files:** Create `src/sharp_video/__init__.py`, `contract.py`, `sequence_io.py`; tests `tests/sharp_video/conftest.py`, `test_contract.py`, `test_sequence_io.py`, `test_import_boundary.py`. Modify `pyproject.toml` (add `av`; scripts added in later tasks).

**Interfaces (produces):**
- `SpatialView(rgb, depth, mask, K, R, C)`; `SpatialFrame(timestamp: float, views: list[SpatialView], pts: int | None = None, source_frame_index: int | None = None, metadata: dict = {})`; `SequenceInfo(width, height, fps, view_count=9, time_base=None, source_filename="", extra: dict = {})` with `.to_attrs()`.
- `SpatialFrame.from_stage1(result, timestamp, pts=None, source_frame_index=None)`, `SpatialFrame.to_stage1() -> SpatialPhotoResult`, `SpatialFrame.stacked() -> dict[str, np.ndarray]` (`rgb [9,H,W,3]` …).
- `SequenceWriter(path, info)` (context manager) `.write_frame(frame)`, `.close()`; `SequenceReader(path)` `.info`, `__len__`, `__iter__`, `.frame(i)`, `.timestamps()`.
- Test helper `make_sequence(num_frames, size=32, fps=30.0, drift=0.0)` in `tests/sharp_video/conftest.py` built from Stage 1's `make_consistent_result` (moving cameras when `drift > 0`).

**Tests:** round trip (arrays bit-exact, attrs, schema paths `/metadata`, `/frames/000000/view_00/rgb`); frame_count written on close; non-increasing timestamp / 8 views / resolution change rejected; failed writer leaves no file; reader is lazy (iterating does not load all frames); import boundary via subprocess with a meta-path blocker for `torch`, `sharp`, `sharp_spatialize`.

- [ ] Write failing tests → run (fail) → implement → run (pass) → commit `feat(sharp_video): spatial sequence contract and temporal HDF5`.

### Task 2: Video input + frame scheduling (`video_io.py`, `scheduler.py`)

**Interfaces:** `VideoInfo` (width, height, fps, frame_count, duration, time_base, pix_fmt, codec, container, color_primaries, color_trc, colorspace, color_range, pts list optional); `probe_video(path) -> VideoInfo` (counts frames by demuxing when the header has none); `DecodedFrame(index, pts, timestamp, rgb)`; `iter_frames(path, selection=FrameSelection()) -> Iterator[DecodedFrame]`; `SUPPORTED_CODECS = {"h264", "hevc"}`, `UnsupportedVideoError`. `FrameSelection(start_frame=0, end_frame=None, max_frames=None)` with `.contains(index)`, `.done(index, emitted)`, `.validate()`.

**Tests** (fixture writes small MP4s with PyAV — `libx264`/`libx265`, 30000/1001 fps, a coloured moving square): probe fields; frame count; timestamps equal `i*1001/30000` within 1 ms; presentation order with B-frames; RGB close to source colours; selection start/end/max; mpeg4 codec → `UnsupportedVideoError`; H.265 decode.

- [ ] TDD cycle → commit `feat(sharp_video): video probing, decoding and frame selection`.

### Task 3: Stage 1 spatializer (`spatializer.py`)

**Interfaces:** `CameraConfig(angle_deg=10.0)` + `load_camera_config(path) -> CameraConfig` (JSON `{"angle_deg": …}`); `FrameTimings(sharp_and_3dgs_s, render_s, validate_s, total_s, peak_gpu_bytes)`; `Stage1Spatializer(camera=CameraConfig(), output_size=None, checkpoint_path=None, device="default", precision="fp32", work_dir)`, `__call__(frame: DecodedFrame) -> tuple[SpatialFrame, FrameTimings]`. Writes `work_dir/png/NNNNNN.png`, calls `sharp_spatialize.api.generate_spatial_photo(..., cache_predictor=True, on_progress=…)`, deletes the PNG.

**Tests:** monkeypatched `generate_spatial_photo` returning `make_consistent_result()` — args forwarded (angle, size, precision, cache_predictor), timings monotone, PNG content equals frame RGB, PNG removed; CUDA e2e test (skip) on a 3-frame clip.

- [ ] TDD cycle → commit `feat(sharp_video): per-frame Stage 1 spatializer`.

### Task 4: Journal + frame runner (`journal.py`, `runner.py`)

**Interfaces:** `Journal(path)` `.mark_done(i, pts)`, `.mark_failed(i, pts, error)`, `.is_done(i)`, `.failures`, `.last_completed`; atomic JSON. `FrameFailuresError(failures)`. `process_frames(frames, spatialize, work_dir, worker_count=1, resume=False) -> Iterator[tuple[SpatialFrame, FrameTimings | None]]` — caches each result as a Stage 1 file `work_dir/frames/NNNNNN.h5` (extra attrs `stage2_timestamp`, `stage2_pts`, `stage2_source_frame_index`), yields in presentation order via a bounded reorder window (`2 × worker_count`), records failures, keeps going, raises `FrameFailuresError` at the end if any frame failed. On resume, valid cache files are loaded instead of recomputed (timings `None`).

**Tests:** order preserved with random worker delays (4 workers); failure of frame 2 of 5 → frames 0,1 yielded, all others processed, error raised listing frame 2, journal has error text; resume after failure recomputes only frame 2; corrupt cache file is recomputed; memory bound: never more than window frames pending.

- [ ] TDD cycle → commit `feat(sharp_video): frame runner with journal, resume and ordered output`.

### Task 5: Sequence validation + temporal metrics + inspection (`validation.py`, `metrics.py`, `inspect.py`)

**Interfaces:** `validate_sequence(path, check_frames=True) -> SequenceReport(ok, failures, frame_count, fps, cfr_ok)`; `temporal_metrics(reader) -> TemporalMetrics` (per-pair arrays + summary dict: rgb_flicker, depth_flicker, camera_translation_delta, camera_rotation_delta_deg, center_depth_delta); `write_inspection(reader, frame_index, out_dir) -> list[Path]` (rgb 3×3 sheet PNG, depth 3×3 sheet PNG, `cameras.json`), `timestamp_table(reader) -> list[dict]`.

**Tests:** consistent synthetic sequence passes; broken R in frame 1 → failure mentions frame 1; duplicated timestamp flagged; static sequence → zero flicker/instability; drifting cameras → positive translation delta; inspection files exist and have expected image size.

- [ ] TDD cycle → commit `feat(sharp_video): sequence validation, temporal metrics, inspection`.

### Task 6: MIV adapter maths (`miv/camera.py`, `miv/depth.py`, `miv/yuv.py`)

**Interfaces:** `OPENCV_TO_TMIV` (3×3); `to_tmiv_pose(R, C) -> (position[3], rotation_deg[3])`, `from_tmiv_pose(position, rotation_deg) -> (R, C)`; `tmiv_view_json(name, K, R, C, width, height, depth_range) -> dict`; `depth_to_geometry(depth, mask, near, far, bit_depth=16) -> (samples uint16, clamped_count)`, `geometry_to_depth(samples, near, far, bit_depth) -> (depth, mask)`; `rgb_to_yuv420(rgb, bit_depth=10) -> (Y, U, V)`, `yuv420_to_rgb(Y, U, V, bit_depth)`, `yuv_frame_bytes`, `read_yuv420_frame(fh, w, h, bit_depth)`.

**Tests:** pose round trip on the Stage 1 rig (all 9 views, drifted too); a 3-D point projected with TMIV's formulas (`u = cx − fx·y/x`, `v = cy − fy·z/x`, world→cam via inverse of `Rz·Ry·Rx` and position) equals OpenCV projection; identity → zero angles; depth round trip relative error < 1e-4 inside range, invalid → 0, clamping counted; RGB→YUV→RGB mean abs error < 2 on smooth images; odd size rejected.

- [ ] TDD cycle → commit `feat(sharp_video.miv): camera, depth and colour conversion for TMIV`.

### Task 7: TMIV patch + build script (`third_party/tmiv/`, `scripts/build_tmiv.sh`, `miv/tmiv.py`)

**Patch (against v24.0):**
1. `source/Encoder/app/Encoder_main.cpp` `run()`: for `i > 0`, `if (auto sc = IO::tryLoadSequenceConfig(json(), m_placeholders, i)) { m_sequenceConfig = std::move(*sc); supportExperimentsThatUseASubsetOfTheCameras(); }` before `m_assessor.encode(...)`.
2. `source/ViewOptimizer/src/ViewOptimizerStage.cpp` `encode()`: copy `m_params->viewParamsList`, overwrite each view's `pose` with the same-named view from `unit.viewParamsList`.
3. `source/Encoder/src/Encoder.cpp` `process()`: `else updateViewPoses(buffer.front());` — new `updateViewPoses` copies poses into `m_transportViewParams` and `m_params.viewParamsList`, throwing if `ci` or `dq` changed.

Produce the patch with `git diff` in a v24.0 checkout. `scripts/build_tmiv.sh [--prefix DIR]`: clone v24.0 into `DIR/src`, `git apply` the patch, on macOS use a dependency file with fmt 11.2.0 and `CXXFLAGS=-Wno-register -Wno-nontrivial-memcall` plus an Apple-clang user preset; run `scripts/install.py <preset> --skip-tests -i DIR/install`; write `DIR/sharp_tmiv.json` (`{"version": "v24.0", "patch": "sharp-per-frame-cameras", "source": …, "install": …}`).

**Interfaces:** `TmivInstall(root)` `.source_dir`, `.bin_dir`, `.exe(name)`, `.encode_script`, `.patched`; `find_tmiv(explicit=None) -> TmivInstall` (arg → `SHARP_TMIV_DIR` → `./.tmiv`), `TmivNotFoundError`; `run_logged(cmd, log_path, cwd=None)` raising `TmivError(log_path)` on non-zero exit.

**Tests:** `find_tmiv` resolution order and error message (fake dirs); `run_logged` writes log and raises; patch file applies cleanly to a v24.0 checkout (gated on network/`SHARP_TMIV_SRC`).

- [ ] Build, write tests, commit `feat(tmiv): per-frame camera patch and build script`.

### Task 8: TMIV configuration generation (`miv/tmiv_config.py`)

**Interfaces:** `CONTENT_ID = "S2"`, `VIEW_NAMES = ("v0",…,"v8")`; `sequence_config(views: list[dict], fps, frame_count, content_name) -> dict` (Version "4.0", BoundingBox_center, Fps, Frames_number, lengthsInMeters, sourceCameraNames, cameras with BitDepthColor 10 / BitDepthDepth 16 / ColorSpace+DepthColorSpace YUV420 / HasInvalidDepth true / Projection Perspective); `encoder_config(width, height, intra_period, max_atlases, max_luma_picture_size) -> dict` (NoPruner, NoViewOptimizer, interPeriod 1, `inputSequenceConfigPathFmt "{1}/seq/{3:06}.json"`, VVC Main10, MIV 2, Rec Unconstrained); `multiplexer_config()`, `decoder_config(bitstream_path)`; `qp_csv(qp_texture, qp_geometry) -> str`; `atlas_budget(width, height, views=9) -> (max_atlases, max_luma_picture_size)`.

**Tests:** required keys present, per-frame fmt contains `{3`, JSON serialisable, NoPruner/NoViewOptimizer selected, atlas budget covers 9 full views, QP CSV parses with TMIV's `QuantizationParameters` rules (header `component_id,rates,RP1`).

- [ ] TDD cycle → commit `feat(sharp_video.miv): TMIV configuration generation`.

### Task 9: Streaming MIV encoder (`miv/encoder.py`)

**Interfaces:** `MIVEncoderConfig(work_dir, tmiv=None, depth_near=0.1, depth_far=1000.0, qp_texture=22, qp_geometry=12, intra_period=32, threads=4, keep_work_dir=True)`; `MIVEncoder(config, runner=run_logged)`; `.begin(fps, width, height, num_views=9)`; `.encode_frame(timestamp, views, pts=None, source_frame_index=None)` appends to `input/S2/vN_texture_WxH_yuv420p10le.yuv` / `vN_depth_WxH_yuv420p16le.yuv` and writes `input/S2/seq/NNNNNN.json`; enforces 9 views, equal sizes, constant K, increasing timestamps; `.finalize(output_path) -> MIVEncodeResult(path, size_bytes, frame_count, fps, duration_s, bitrate_bps, encode_time_s, manifest_path, cfr, clamped_depth_pixels)`. Finalize writes configs, runs `python <src>/scripts/encode.py -i input -o output -s S2 -n T -c enc.json -r RP0 RP1 -v VVenC -C <src>/config/test/vvenc.cfg -m mux.json -q qps.csv -t <install> --config-dir <input> -j threads`, copies `output/S2/RP1/TMIV_S2_RP1.bit` atomically to `output_path`, writes manifest `output_path + ".json"`. `encode_sequence(reader, config, output_path)` = Mode A helper. Memory: one frame at a time.

**Tests:** fake runner captures command and creates the expected `.bit`; YUV file sizes = T × frame bytes; per-frame JSON poses match the frames; manifest timestamps; VFR detection; errors for 8 views / K change / non-increasing timestamps / encode before begin; TMIV integration (gated): 4-frame 64×64 drifting synthetic sequence encodes to a non-empty `.miv`.

- [ ] TDD cycle → commit `feat(sharp_video.miv): streaming MIV encoder over TMIV`.

### Task 10: MIV decode + validation (`miv/decode.py`, `miv/validate.py`)

**Interfaces:** `decode_miv(miv_path, frame_count, work_dir, tmiv) -> DecodedMiv` (runs `TmivDecoder -n T -N T -s S2 -r RP1` with multiview texture/geometry and per-frame sequence-config outputs, plus `TmivParser`); `DecodedMiv.frames()` yields per-frame `{name: (rgb, depth, mask, K, R, C)}` (sequence config carried forward between change points); `validate_miv(miv_path, reference: SequenceReader, tmiv, work_dir, thresholds) -> MivReport(ok, failures, stats)` checking frame count, fps, duration, 9 named views, resolution, K/R/C tolerance (K 1e-3 rel, C 1e-4 m, R 1e-4), depth median relative error ≤ 5%, RGB PSNR ≥ 25 dB, bitrate.

**Tests:** comparison logic unit-tested with a fake `DecodedMiv` (perfect → ok; shifted camera → failure names frame/view; missing frame → failure); TMIV integration (gated): encode then validate the synthetic sequence end to end, including moving cameras reappearing per frame.

- [ ] TDD cycle → commit `feat(sharp_video.miv): MIV decoding and correctness validation`.

### Task 11: Pipeline, report, CLI (`pipeline.py`, `report.py`, `cli.py`)

**Interfaces:** `PipelineOptions(input, output, mode="two-step"|"streaming", dump_hdf5=None, work_dir=None, selection, worker_count=1, camera, output_size, resume, checkpoint, device, precision, miv: MIVEncoderConfig kwargs)`; `run_video_to_miv(options, spatialize=None, encoder_factory=MIVEncoder) -> PipelineResult(miv, sequence_path, report_paths, stats)`; `build_report(...) -> dict`, `write_report(report, base_path)` (JSON + Markdown). CLI scripts in `pyproject.toml`: `sharp_video_to_miv`, `sharp-video-validate`, `sharp-video-inspect`, `sharp-miv-encode`, `sharp-miv-validate`. `--output-resolution WxH` (default: source rounded down to multiple of 8).

**Tests:** fake spatializer + fake runner: both modes produce byte-identical encoder inputs; `--dump-hdf5` written in streaming mode; report contains every spec §35 field; failure propagates non-zero exit with failed frames listed; `CliRunner` for each command's help and a full fake run.

- [ ] TDD cycle → commit `feat(sharp_video): video-to-MIV pipeline, report and CLI`.

### Task 12: Docs, test matrix, CUDA e2e

**Files:** `docs/stage2_sequence_format.md`, `docs/stage2_miv.md`, README section, `scripts/stage2/run_test_matrix.py` (runs `sharp_video_to_miv` + validators for clips named `static_*`, `camera_motion_*`, `moving_object_*`, `camera_object_*`, `difficult_*`, aggregates reports into `matrix_report.md`), `tests/sharp_video/test_e2e.py` (CUDA + TMIV gated: 3-frame real clip → MIV → `validate_miv` ok).

- [ ] Write, run gated tests where possible, commit `docs(sharp_video): Stage 2 formats, MIV mapping, test matrix`.

### Task 13: Final verification

- [ ] Full test run, ruff on new code, spec §38 acceptance checklist mapped to evidence in `docs/stage2_miv.md`.
