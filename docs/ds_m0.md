# DS-Image M0 prototype (`ds_m0`)

Implements the *DS-Image M0 Prototype* specification (Parts I-IV): the chain

    9-view RGBD (spatial_photo.h5)
      -> DS Asset Builder -> DS Asset (2 layers)
      -> DS-Image Writer  -> DS-Image (JPEG + JUMBF)
      -> DS-Image Reader  -> DS Asset
      -> DS Presenter     -> RGB + coverage mask

No AI, MIV, 3DGS or SHARP code is involved; only `numpy`, `Pillow` and `h5py` are needed.

## Quick start

```bash
pip install -e .                      # or: PYTHONPATH=src python -m ds_m0 ...
ds_m0 synth --output scene.h5         # deterministic synthetic 9-view scene (or use a real spatial_photo.h5)
ds_m0 evaluate --input scene.h5 --output-dir out        # reports A-E, renders, summary.txt
ds_m0 build --input scene.h5 --output asset/            # DS Asset (debug directory, not a DS-Image)
ds_m0 write --asset asset/ --output scene.ds.jpg        # DS-Image
ds_m0 inspect --input scene.ds.jpg
ds_m0 render --asset scene.ds.jpg --translate 0.2 0 0 --output renders/
ds_m0 roundtrip --input scene.h5 --output-dir rt/
ds_m0 export-config --output m0.json                    # every algorithmic choice, human readable
```

Exit codes: 0 ok, 1 failed check, 2 configuration, 3 input, 4 geometry, 5 format, 6 runtime, 7 validation.

## Layout (Part III section 2)

| package | contents |
|---|---|
| `model` | `Camera`, `Layer`, `DSAsset`, `SourceView/SourceDataset`, metadata (de)serialiser |
| `geometry` | the only camera maths: back-projection, transforms, projection, rounding, hidden test, order-independent winner selection |
| `creator` | candidate generator, candidate merger, asset builder (knows nothing about files) |
| `io` | source reader, JPEG codec, depth codecs, sparse Layer-1 codec, JUMBF container, writer, reader, debug-asset directory |
| `presenter` | depth buffer + point-splat renderer (depends only on `model`, `geometry`, `config`) |
| `evaluation` | file / sparsity / coverage / round-trip / MIV-baseline metrics, full reference run + reports |
| `config`, `tools` | `M0Config`, CLI, synthetic dataset |

The dependency rules of section 62 are enforced by AST-based tests in
`tests/ds_m0/regression/test_ds_determinism_and_architecture.py`.

## Decisions the specification left open

* **Input**: the existing `spatial_photo.h5` (`rgb`, `depth`, `mask`, `camera/{K,R,C}`). The pose
  `x_cam = R (x_world - C)` is normalised to `x_cam = R x + t`, `t = -R C`; `coordinate_system` must be
  `OpenCV` and `depth_definition` `camera_z`, otherwise the load is rejected.
* **Pixel convention / rounding**: integer `(x, y)` is the sample position; continuous coordinates round
  with `floor(v + 0.5)` (`half_up`), the same rule in Builder and Presenter. `half_even` is selectable.
* **Source pixels whose base pixel has an invalid Layer-0 depth** produce no candidate (there is nothing to
  be "behind").
* **Tie rules**: Layer-1 merge = nearest hidden depth (within `depth_epsilon`), smaller view angle, lower view
  id, then lower source raster index (only to make ties total). Presenter = smaller depth, Layer 0 before
  Layer 1, lower raster index. Both use one shared selection routine, so results do not depend on input order.
* **Metadata**: sorted-key compact UTF-8 JSON (deterministic, versioned `format_version = "1.0"`).
* **Depth codecs** (selected by name, parameters in metadata): `float32_zlib` (lossless, default) and
  `uint16_quant_zlib` (min/max quantisation, error <= `(max-min)/65535/2`, reported in the file).
* **Layer-0 validity** is stored as its own 1-bit mask payload (`layer0_mask`). The spec lists no Layer-0
  validity payload, but the Reader must reproduce Layer0.Valid exactly, even where an invalid pixel carries a
  positive depth. It is counted in the file-size report.
* **Layer-1 RGB / mask streams** are zlib-compressed losslessly; the depth stream uses the depth codec.
* **Integrity**: every payload has a length and CRC-32 in the metadata; the Reader rejects mismatches,
  truncated packet sequences, non-zero mask padding, stream/mask count mismatches and bad dimensions.
* **JUMBF**: one ISO 19566-5 `jumb` superbox (`ds-image`) with one child superbox per payload
  (`header`, `metadata` (json box), `layer0_depth`, `layer0_mask`, `layer1_mask`, `layer1_rgb`,
  `layer1_depth` (bidb boxes)), carried in JPEG APP11 packets (`JP`, instance 1, sequence numbers, repeated
  box header) right after the JFIF segment. All of this lives in `io/jumbf_container.py`. It is a minimal
  in-tree implementation; validate it against an external JUMBF parser before treating the files as
  interchange-grade.
* **Presenter size override**: `PresenterConfig.output_width/height` replace the target camera's dimensions
  (intrinsics are used as given).
* **Config**: JSON (no extra dependency). `base_view_id` lives in `BuilderConfig` only.
* **Round-trip RGB tolerance**: mean absolute Layer-0 error <= `rgb_mean_abs_tolerance` (12 by default, a
  sanity bound for JPEG, not a DS requirement); everything else is exact for the lossless depth codec.

## Known limitations (Part IV section 49)

1. Layer 1 comes from existing multi-view RGBD, not a single image. 2. No AI completion of RGB or depth.
3. Two layers only. 4. Presenter is a one-pixel point splat. 5. Uncovered regions are not inpainted.
6. JPEG/depth encodings are prototype choices. 7. The MIV comparison is storage only (`--miv-file`).
8. No IRU hardware. Also: parallel candidate generation is not implemented (single-threaded reference), so the
optional multi-thread determinism test (section 16) is deferred.

## Tests

`pytest tests/ds_m0` covers T01-T13 and T16 (T14 is exercised with a stand-in MIV file; T15 timings are
recorded in `run_info.json`). `tests/ds_m0/golden/` holds the golden artifacts for the synthetic reference dataset: `reference_scene.h5`,
`golden_asset/`, `golden.ds.jpg`, `golden_renders/` (RGB + coverage PNGs) and `metrics.json`.
`generate_golden.py` regenerates them (bump `DATASET_ID` when you do). `run_info.json` also records peak memory.
