"""Tests for sharp_spatialize.webui.jobs.

`run_job` takes its pipeline callable by injection, so nothing here needs
SHARP, a checkpoint, or a GPU.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image
from sharp_spatialize.webui import jobs as jobs_module
from sharp_spatialize.webui.jobs import (
    INVALID_DEPTH_COLOR,
    Job,
    JobParams,
    JobStore,
    depth_to_rgb,
    encode_jpeg,
    ensure_h5,
    fit_within,
    pipeline_reporter,
    prepare_source_image,
    run_job,
    turbo,
)


def _jpeg_bytes(width: int, height: int, orientation: int | None = None) -> bytes:
    """A solid-color JPEG, optionally carrying an EXIF orientation tag."""
    image = Image.new("RGB", (width, height), (200, 120, 40))
    buffer = io.BytesIO()
    if orientation is None:
        image.save(buffer, format="JPEG")
    else:
        exif = Image.Exif()
        exif[jobs_module.EXIF_ORIENTATION_TAG] = orientation
        image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


# ---------------------------------------------------------------- colorizing


def test_turbo_spans_the_colormap_and_clamps_out_of_range_values():
    """Turbo maps 0/1 to the colormap ends and clamps anything outside [0, 1]."""
    colors = turbo(np.array([0.0, 1.0, -5.0, 5.0], dtype=np.float32))

    assert colors.dtype == np.uint8
    assert colors.shape == (4, 3)
    np.testing.assert_array_equal(colors[0], colors[2])  # -5 clamps to 0
    np.testing.assert_array_equal(colors[1], colors[3])  # +5 clamps to 1
    assert not np.array_equal(colors[0], colors[1])


def test_depth_to_rgb_colors_valid_pixels_and_flattens_invalid_ones():
    """Invalid/zero-depth pixels get the flat placeholder color, valid ones a ramp."""
    depth = np.array([[[1.0, 2.0, 4.0, 0.0]]], dtype=np.float32)
    mask = np.array([[[1, 1, 1, 0]]], dtype=np.uint8)

    colored = depth_to_rgb(depth, mask)

    assert colored.shape == (1, 1, 4, 3)
    np.testing.assert_array_equal(colored[0, 0, 3], INVALID_DEPTH_COLOR)
    # Near and far map to different ends of the colormap.
    assert not np.array_equal(colored[0, 0, 0], colored[0, 0, 2])


def test_depth_to_rgb_returns_a_flat_image_when_nothing_is_valid():
    """An all-invalid mask must not divide by zero; every pixel is the placeholder."""
    depth = np.zeros((2, 3, 3), dtype=np.float32)
    mask = np.zeros((2, 3, 3), dtype=np.uint8)

    colored = depth_to_rgb(depth, mask)

    assert (colored == INVALID_DEPTH_COLOR).all()


def test_depth_to_rgb_handles_a_single_valid_depth_without_dividing_by_zero():
    """One valid pixel makes the 2nd and 98th percentiles equal; that must not blow up."""
    depth = np.array([[[3.0, 0.0]]], dtype=np.float32)
    mask = np.array([[[1, 0]]], dtype=np.uint8)

    colored = depth_to_rgb(depth, mask)

    assert np.isfinite(colored).all()
    np.testing.assert_array_equal(colored[0, 0, 1], INVALID_DEPTH_COLOR)


def test_encode_jpeg_round_trips_through_pillow():
    """encode_jpeg produces bytes Pillow reads back at the same size."""
    rgb = np.random.default_rng(0).integers(0, 255, (12, 20, 3), dtype=np.uint8)

    decoded = Image.open(io.BytesIO(encode_jpeg(rgb)))

    assert decoded.format == "JPEG"
    assert decoded.size == (20, 12)


# ------------------------------------------------------------ source images


@pytest.mark.parametrize(
    ("width", "height", "max_size", "expected"),
    [
        (100, 50, 200, (100, 50)),  # already small enough
        (200, 100, 200, (200, 100)),  # exactly at the cap
        (400, 200, 200, (200, 100)),  # landscape
        (200, 400, 200, (100, 200)),  # portrait scales on the long edge
    ],
)
def test_fit_within_scales_only_on_the_long_edge(width, height, max_size, expected):
    """fit_within caps the long edge and preserves aspect ratio."""
    assert fit_within(width, height, max_size) == expected


def test_fit_within_never_returns_a_zero_dimension():
    """An extreme aspect ratio must still produce a renderable size."""
    assert fit_within(2000, 3, 100) == (100, 1)


def test_prepare_source_image_applies_exif_orientation_and_then_clears_it():
    """Orientation is baked into pixels once, and the tag reset so io.load_rgb won't redo it."""
    # Orientation 6 means "rotate 90° CW to display", so 40x20 becomes 20x40.
    data = _jpeg_bytes(40, 20, orientation=6)

    image, info = prepare_source_image(data, max_size=2048)

    assert image.size == (20, 40)
    assert image.getexif()[jobs_module.EXIF_ORIENTATION_TAG] == 1
    assert (info["original_width"], info["original_height"]) == (20, 40)
    assert info["downscaled"] is False


def test_prepare_source_image_downscales_and_reports_both_sizes():
    """A large upload is capped, and the info records what it was and what it became."""
    image, info = prepare_source_image(_jpeg_bytes(1600, 800), max_size=640)

    assert image.size == (640, 320)
    assert info == {
        "original_width": 1600,
        "original_height": 800,
        "width": 640,
        "height": 320,
        "downscaled": True,
    }


def test_prepare_source_image_converts_exotic_modes_to_rgb():
    """A palette PNG (mode "P") is converted so the pipeline always sees RGB."""
    buffer = io.BytesIO()
    Image.new("P", (24, 24)).save(buffer, format="PNG")

    image, _info = prepare_source_image(buffer.getvalue(), max_size=2048)

    assert image.mode == "RGB"


# ------------------------------------------------------------------- params


def test_job_params_accepts_defaults():
    """The default params are themselves valid."""
    assert JobParams().validated() is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"angle_deg": 0.0},
        {"angle_deg": -3.0},
        {"angle_deg": jobs_module.MAX_ANGLE_DEG + 0.1},
        {"max_size": 999},
        {"precision": "int8"},
    ],
)
def test_job_params_rejects_out_of_range_values(kwargs):
    """Bad angles, resolutions, and precisions are rejected rather than clamped."""
    with pytest.raises(ValueError):
        JobParams(**kwargs).validated()


# -------------------------------------------------------------------- store


def test_job_store_evicts_old_jobs_and_releases_arrays_from_merely_stale_ones():
    """The store keeps N jobs, but full arrays for only the newest few."""
    store = JobStore(max_jobs=3, max_jobs_with_arrays=1)
    created = []
    for _ in range(4):
        job = store.create(JobParams())
        job.result = object()
        job.views = [b"jpeg"]
        created.append(job)

    assert store.get(created[0].job_id) is None, "oldest job should be evicted entirely"
    for job in created[1:]:
        assert store.get(job.job_id) is not None

    assert created[-1].result is not None, "newest job keeps its arrays"
    assert created[1].result is None and created[2].result is None
    assert created[1].views == [b"jpeg"], "stale jobs keep their encoded previews"


def test_job_store_clear_cleans_up_every_job(tmp_path):
    """clear() drops all jobs and removes their temp directories."""
    store = JobStore()
    job = store.create(JobParams())
    job.views = [b"jpeg"]
    h5_dir = tmp_path / "job-temp"
    h5_dir.mkdir()
    job.h5_path = h5_dir / "spatial_photo.h5"
    job.h5_path.write_bytes(b"stub")

    store.clear()

    assert store.get(job.job_id) is None
    assert not h5_dir.exists()
    assert job.views == []


def test_job_status_is_json_serializable_and_reports_progress():
    """status() reflects the latest report() and stays JSON-friendly."""
    job = Job(job_id="abc", params=JobParams(angle_deg=7.5))
    job.report("rendering", 0.123456)

    status = job.status()

    assert status["stage"] == "rendering"
    assert status["progress"] == 0.123, "progress is rounded to keep the poll payload small"
    assert status["angle_deg"] == 7.5
    assert status["has_arrays"] is False
    assert json.loads(json.dumps(status)) == status


# ------------------------------------------------------------------ running


def _fake_generate(result):
    """A stand-in for generate_spatial_photo that reports progress like the real one."""

    def generate(input_path, *, on_progress=None, **kwargs):
        generate.calls.append({"input_path": input_path, **kwargs})
        if on_progress is not None:
            on_progress("inference", 0.05)
            on_progress("rendering", 0.55)
            on_progress("validating", 0.9)
            on_progress("done", 1.0)
        return result

    generate.calls = []
    return generate


def test_pipeline_reporter_never_reaches_full_before_encoding():
    """The pipeline's own "done" must not fill the bar while previews still encode."""
    job = Job(job_id="abc", params=JobParams())
    report = pipeline_reporter(job)

    report("rendering", 0.55)
    assert job.progress == pytest.approx(0.55 * jobs_module.PIPELINE_PROGRESS_SHARE)

    report("done", 1.0)
    assert job.stage == "rendering", "the pipeline's 'done' is dropped"
    assert job.progress < 1.0


def test_run_job_fills_in_previews_metadata_and_timings(consistent_spatial_photo_result):
    """A successful run encodes 9 color + 9 depth previews and ends in state "done"."""
    result = consistent_spatial_photo_result
    job = Job(job_id="abc", params=JobParams(max_size=640, filename="photo.jpg"))
    generate = _fake_generate(result)

    run_job(job, _jpeg_bytes(64, 48), generate=generate)

    assert job.state == "done"
    assert job.error is None
    assert job.progress == 1.0
    assert len(job.views) == 9
    assert len(job.depths) == 9
    assert Image.open(io.BytesIO(job.views[0])).size == (32, 32)
    assert job.source_jpeg is not None
    assert job.source_info["filename"] == "photo.jpg"
    assert job.metadata["num_views"] == 9
    assert "pipeline" in job.timings and "encode" in job.timings
    job.cleanup()


def test_run_job_passes_the_user_choices_through_to_the_pipeline(consistent_spatial_photo_result):
    """angle/precision from the params, plus the web UI's own caching choice, reach generate."""
    job = Job(job_id="abc", params=JobParams(angle_deg=6.5, precision="bf16"))
    generate = _fake_generate(consistent_spatial_photo_result)

    run_job(job, _jpeg_bytes(64, 48), device="cuda", generate=generate)

    (call,) = generate.calls
    assert call["angle_deg"] == 6.5
    assert call["precision"] == "bf16"
    assert call["device"] == "cuda"
    assert call["cache_predictor"] is True, "a long-lived server must not reload the checkpoint"
    job.cleanup()


def test_run_job_records_a_failure_instead_of_raising():
    """A pipeline exception becomes state="error" with a readable message."""
    job = Job(job_id="abc", params=JobParams())

    def _boom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    run_job(job, _jpeg_bytes(64, 48), generate=_boom)

    assert job.state == "error"
    assert "RuntimeError" in job.error
    assert "CUDA out of memory" in job.error
    assert job.views == []


def test_run_job_records_a_failure_for_an_undecodable_upload(consistent_spatial_photo_result):
    """Garbage bytes fail in decoding, before the pipeline is ever called."""
    job = Job(job_id="abc", params=JobParams())
    generate = _fake_generate(consistent_spatial_photo_result)

    run_job(job, b"this is not an image", generate=generate)

    assert job.state == "error"
    assert generate.calls == []


def test_run_job_cleans_up_its_temp_dir_when_it_fails():
    """A failed run must not leave its scratch directory behind."""
    job = Job(job_id="abc", params=JobParams())

    def _boom(*args, **kwargs):
        raise RuntimeError("nope")

    run_job(job, _jpeg_bytes(64, 48), generate=_boom)

    assert job.h5_path is None


# ---------------------------------------------------------------------- h5


def test_ensure_h5_writes_once_and_reuses_the_file(consistent_spatial_photo_result):
    """The .h5 is written lazily on first download and then served from disk."""
    job = Job(job_id="abc", params=JobParams())
    run_job(job, _jpeg_bytes(64, 48), generate=_fake_generate(consistent_spatial_photo_result))

    path = ensure_h5(job)
    assert path.exists()
    written_at = path.stat().st_mtime_ns

    job.release_arrays()
    assert ensure_h5(job) == path, "an already-written file survives releasing the arrays"
    assert path.stat().st_mtime_ns == written_at

    job.cleanup()
    assert not path.exists()


def test_ensure_h5_reports_a_clear_error_once_the_arrays_are_gone(consistent_spatial_photo_result):
    """Asking for the .h5 of an evicted job explains what happened."""
    job = Job(job_id="abc", params=JobParams())
    run_job(job, _jpeg_bytes(64, 48), generate=_fake_generate(consistent_spatial_photo_result))
    job.release_arrays()
    job.h5_path.unlink(missing_ok=True)

    with pytest.raises(FileNotFoundError, match="released"):
        ensure_h5(job)

    job.cleanup()
