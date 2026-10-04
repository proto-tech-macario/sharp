# Stage 2: spatial sequence → MIV

How `sharp_video` packages a 9-view spatial sequence as MPEG Immersive Video
(MIV, ISO/IEC 23090-12), how that was verified, and what it does not do.
Design: `docs/superpowers/specs/2026-09-13-video-to-miv-design.md`.

## Pipeline

```
input.mp4 ─► PyAV decode (presentation order, PTS kept)
          ─► Stage 1 per frame (sharp_spatialize, unchanged; CUDA)
          ─► SpatialFrame[t] ──► stage2_sequence.h5        (two-step mode / --dump-hdf5)
          ─► MIVEncoder.encode_frame                       (sharp_video.miv: no SHARP/torch)
               RGB   → yuv420p10le (BT.709, limited range)
               depth → 16-bit normalized disparity, 0 = invalid
               K,R,C → TMIV camera JSON, one file per frame
          ─► MIVEncoder.finalize: TMIV encode.py
               TmivEncoder → vvencFFapp (texture, geometry atlases) → TmivMultiplexer
          ─► output.miv  +  output.miv.json (manifest)  +  output.miv.report.{json,md}
```

`output.miv` is TMIV's multiplexed V3C sample stream: MIV metadata plus in-band
VVC Main10 video sub-bitstreams. It is packaged as a raw V3C sample stream, not ISOBMFF.

## Running it

```
scripts/build_tmiv.sh                                  # once: patched TMIV into ./.tmiv (CPU only)
sharp_video_to_miv -i input.mp4 -o output.miv          # needs the CUDA Stage 1 environment
sharp_video_to_miv -i input.mp4 -o output.miv --dump-hdf5 seq.h5 --validate
sharp-miv-encode seq.h5 -o output.miv                  # standalone encoder, no GPU
sharp-miv-validate output.miv --reference seq.h5       # TmivDecoder + checks, no GPU
sharp-video-webui ~/out                                # the same pipeline, in a browser
```

`sharp-video-webui` runs both halves of this document from one page: drop in an
mp4 and it queues `run_video_to_miv` on a worker thread (`webui/conversions.py`)
with the frame range, view height, camera angle and QPs the page asks for, then
previews the .miv it wrote by decoding it with TmivDecoder (`webui/previews.py`).
The pipeline itself is untouched by the web UI: progress is counted off the
per-frame cache `runner.process_frames` already writes, and a run is stopped by
raising out of the per-frame Stage 1 call, which is why `ConversionCancelled` is
a `BaseException` -- `process_frames` journals a failing *frame* and carries on.

MIV encoding and decoding need no GPU. Only Stage 1 (SHARP + gsplat) needs CUDA.

**Disk.** A run writes ~200 MB of intermediates per frame at 1280x720 views:
~45 MB in the Stage 1 frame cache, ~50 MB of raw views for TMIV, and the raw
atlases plus VVenC's reconstruction (~75 MB). The pipeline estimates this up
front (`diskspace.py`) and refuses a run that will not fit (`--no-disk-check`
starts anyway). On WSL2 the free space that counts is the Windows drive under
the growable `ext4.vhdx`, not what `df` reports inside Linux. When that drive
fills, the whole VM stops. The web UI deletes TMIV's raw YUV files as soon as
TMIV is done (`keep_intermediate=False`), and the SHARP predictor is released
before TMIV starts, which frees ~6 GB of GPU memory.

`scripts/build_tmiv.sh` builds on Linux (GCC) and macOS (Apple clang). On macOS
it bumps fmt to 11.2.0 and silences three warnings that TMIV's dependencies turn
into errors on newer clang.

## Mapping Stage 1 onto MIV

| Stage 1 | MIV / TMIV | Where |
|---|---|---|
| 9 views V0..V8, V4 centre | 9 source views `v0..v8`, all basic views, coded in full (`NoViewOptimizer`, `NoPruner`) | `tmiv_config.encoder_config` |
| OpenCV axes (x right, y down, z forward) | TMIV/OMAF axes (x forward, y left, z up): `x_tmiv = P x_cv`, `P = [[0,0,1],[-1,0,0],[0,-1,0]]` | `miv/camera.py` |
| `R` world→camera, `C` centre | `Position = P C`, camera→world `Rz(yaw) Ry(pitch) Rx(roll) = P Rᵀ Pᵀ` | `miv/camera.py` |
| `K` | `Focal = [fx, fy]`, `Principle_point = [cx, cy]` | `miv/camera.py` |
| depth = camera Z, metres, 0 invalid | geometry = normalized disparity over `Depth_range = [near, far]`, 16-bit input, `HasInvalidDepth`, sample 0 = unoccupied | `miv/depth.py` |
| RGB uint8 | texture `yuv420p10le`, BT.709 limited range | `miv/yuv.py` |
| per-frame R, C (Stage 1's focus depth follows the scene) | per-frame camera files; MIV view parameter updates (`miv_view_params_update_extrinsics`) in every frame's common atlas frame (`interPeriod = 1`) | `third_party/tmiv/` |
| timestamps | MIV `Fps`; per-frame timestamps and PTS in `output.miv.json` | `miv/encoder.py` |

`test_tmiv_projection_equals_opencv_projection` checks the camera conversion
against TMIV's own projection equations (`Renderer/Projector.h`) for all 9 views.

### The TMIV patch (`third_party/tmiv/sharp-per-frame-cameras.patch`)

TMIV v24.0 reads cameras once per sequence, in three places:

1. `Encoder_main.cpp` loads the sequence configuration for frame 0 only.
2. `Encoder::Impl::process` calls `prepareSequence` only once, fixing the view list.
3. `ViewOptimizerStage` caches frame 0's optimized view list and overwrites each frame's with it.

The patch reloads the configuration per frame (the input path format has a
frame-index placeholder), carries each frame's poses through the view optimizer
by view name, and updates the encoder's transported poses per access unit. It
throws if intrinsics or depth quantization change between frames. TMIV's
`EncodeMiv` then codes the new poses as standard MIV view parameter updates,
so any conformant MIV decoder reads them.

The patch also makes frame rates exact. TMIV v24.0 writes the MIV timing
information with `num_units_in_tick = 1`, so it accepts only integer rates and
aborts on 29.97 fps video. The patch codes NTSC-family rates as
`time_scale / num_units_in_tick = 30000 / 1001` (integer rates keep a tick of 1).
It also has `encode.py` pass VVenC the same fraction (`-fr 30000 --FrameScale 1001`)
instead of a float.

Finally, `encode.py` decodes tool output with `errors="replace"`. When a plane
is lossless in every frame (a constant geometry atlas, e.g. a static or empty
scene), VVenC 1.12's summary row overflows its 256-byte format buffer and the
retry reuses a consumed `va_list`, printing garbage bytes; stock `encode.py`
then aborts with `UnicodeDecodeError` after every bitstream was written. In
total: 76 added lines in 4 files (`Encoder_main.cpp`, `Encoder.cpp`,
`ViewOptimizerStage.cpp`, `scripts/encode.py`).

### Encoder settings

| Setting | Value | Why |
|---|---|---|
| `intraPeriod` | 32 (16 allowed) | VVenC random access supports only GOP 16/32 with GOP = intra period |
| `interPeriod` | 1 | one common atlas frame per frame, which carries the camera update |
| geometry | full resolution, 10-bit video, occupancy embedded, dynamic depth range | depth precision |
| atlases | the 9 views stacked in one column, split evenly over as few atlases as `maxLumaPictureSize` allows (1280x720 → one 1280x6480 atlas) | no empty atlas area for VVenC to encode |
| QP | texture 22, geometry 8 | `--qp-texture`, `--qp-geometry` |
| depth range | 0.1–1000 m (fixed per run) | `--depth-near/--depth-far`; clamped pixels are counted in the manifest |

## Verification

All of the following ran on macOS (Apple M3) with the patched TMIV built by
`scripts/build_tmiv.sh`'s macOS path. Synthetic sequences are consistent 9-view
renderings of a tilted, textured plane through the real Stage 1 camera rig,
with the plane moving so the cameras move every frame.

- `test_real_tmiv_round_trip_preserves_views_cameras_and_timing`: encode → TmivDecoder → compare.
- `test_real_tmiv_pipeline_on_h264_clip[two-step|streaming]`: H.264 clip → pipeline (Stage 1 faked) → TMIV → TmivDecoder → compare.
- `test_real_tmiv_encodes_moving_camera_sequence`: encoder smoke test.

Measured on 8 frames of 9 × 128×128 views (QP 22/8, intra period 16):

| Check | Result |
|---|---|
| frames decoded / view count | 8 / 9 |
| camera update frames seen by the decoder | 0, 1, 2, …, 7 (every frame) |
| max camera error after decode | K 0, R 1.2e-7, C 3.5e-6 m |
| depth median relative error (max over views and frames) | 0.07 % |
| validity-mask agreement (min) | 99.9 % |
| luma PSNR on valid pixels | mean 37.5 dB, min 27.2 dB |
| MIV size / bitrate | 23.9 kB / 718 kbps |

The luma minimum falls on higher VVC temporal layers (random access codes them at a
higher QP) of a sliding hard-edged checker, a deliberately harsh texture.
`sharp-miv-validate` gates luma PSNR at 25 dB. That confirms texture/depth
correspondence: a misaligned or missing texture scores 6–15 dB. It is not a
quality target, and quality is reported separately.

**Not yet run:** the full pipeline with real SHARP inference on real video.
That needs the CUDA machine. The command is the first one under *Running it*, and
`scripts/stage2/run_test_matrix.py` runs it over the five spec §34 categories.

## Limitations

- **Timing.** MIV carries a frame rate, not timestamps. Constant-frame-rate
  input is preserved exactly (frame *i* at `t0 + i/fps`), for integer rates and
  for N/1001 rates (23.976, 29.97, 59.94). For variable-frame-rate input the
  manifest keeps every timestamp and the report and manifest say
  `constant_frame_rate: false`. Other fractional rates (e.g. 12.5 fps) are
  rejected by the encoder with a clear error.
- **Constant-depth views.** If every valid pixel of a view has exactly the same
  depth (a synthetic fronto-parallel plane), TMIV's dynamic depth range
  collapses and the view decodes as unoccupied. `sharp-miv-validate` reports it
  as a mask-agreement failure. Real SHARP depth does not do this.
- **Intrinsics** must be constant over the clip. That holds for Stage 1 on video
  (same size, same default focal length for every PNG frame).
- **View names** are not part of MIV. Views are identified by index, and
  TmivDecoder calls them `pv00`…`pv08`, in v0…v8 order.
- **Packaging.** Raw V3C sample stream (`.miv`), not ISOBMFF/MP4.

## Acceptance criteria (spec §38)

| Criterion | Evidence |
|---|---|
| Video decoded into ordered RGB frames | `test_video_io.py` (H.264 with B-frames, H.265): indices, strictly increasing PTS, pixels |
| Frame timestamps preserved | `test_frames_decode_in_presentation_order_with_timestamps`, `test_selection_start_end`; sequence attrs; manifest |
| Every selected frame runs through Stage 1 | `test_video_spatializer.py` (Stage 1 API with exact arguments), `test_video_runner.py` (no silent omission, resume); `test_video_e2e.py` on CUDA |
| 9 RGB views / 9 depth maps per frame | `SpatialFrame` enforces 9; Stage 1 validation per frame (`sharp-video-validate`) |
| Stage 1-compatible K, R, C | same arrays, same conventions; Stage 1 rotation/reprojection checks per frame |
| All frames stored in the temporal HDF5 | `test_video_sequence_io.py`; `docs/stage2_sequence_format.md` |
| MIV encoder accepts the SpatialSequence contract | `MIVEncoder.begin/encode_frame/finalize`; import-boundary test (no SHARP/torch) |
| A valid MIV file is generated | real-TMIV tests above; `TmivParser` dump |
| MIV preserves frame count and timing | decoded frame count and fps, duration and manifest timestamps checked |
| MIV contains the multi-view/depth representation | 9 decoded views with texture, geometry, occupancy, cameras compared to source |
| An independent MIV decoder parses/decodes it | TmivDecoder (MPEG reference decoder, integrated VVdeC) |
| Performance and output-size statistics | `<output>.report.json/.md` (spec §35 fields), `test_report_has_every_performance_field` |
