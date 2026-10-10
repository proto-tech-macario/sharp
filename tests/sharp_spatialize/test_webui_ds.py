"""Tests for the web UI's DS-Image analysis (webui.ds) and its HTTP routes.

Use the synthetic 9-view checkerboard from conftest, so the DS chain runs on
self-consistent geometry without SHARP or a GPU.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("ds_m0")

import test_webui_server as server_tests  # noqa: E402
from sharp_spatialize.hdf5_io import SpatialPhotoResult  # noqa: E402
from sharp_spatialize.webui import ds as ds_module  # noqa: E402
from sharp_spatialize.webui.jobs import Job, JobParams, load_h5_job  # noqa: E402
from test_webui_server import POLL_TIMEOUT_S, _jpeg_bytes  # noqa: E402

client = server_tests.client  # the running test server, shared with test_webui_server

OUTER_VIEWS = [i for i in range(9) if i != ds_module.REFERENCE_VIEW]


def _h5_bytes(result: SpatialPhotoResult, tmp_path) -> bytes:
    path = tmp_path / "spatial_photo.h5"
    result.save(path)
    return path.read_bytes()


@pytest.fixture
def analysis(consistent_spatial_photo_result):
    """A finished DS analysis of the synthetic scene."""
    a = ds_module.DSAnalysis(codec="float32_zlib")
    ds_module.run_analysis(a, consistent_spatial_photo_result, h5_bytes=123456)
    yield a
    a.cleanup()


# ---------------------------------------------------------------- analysis


def test_analysis_builds_checks_and_scores_every_source_view(analysis):
    """Build, write, read back and check pass, and all 9 views get scored."""
    assert analysis.state == "done", analysis.error
    status = analysis.status()
    assert status["roundtrip"]["passed"] is True
    assert status["legacy_jpeg_ok"] is True
    assert status["h5_bytes"] == 123456
    assert [row["view"] for row in status["per_view"]] == list(range(9))
    file = status["file"]
    assert file["ds_total_bytes"] == analysis.ds_path.stat().st_size
    assert file["primary_jpeg_bytes"] > 0 and file["base_depth_bytes"] > 0


@pytest.mark.parametrize("presenter", ds_module.PRESENTERS)
def test_the_presenter_reproduces_the_reference_view_exactly(analysis, presenter):
    """At the reference camera either Presenter returns the source view pixel for pixel."""
    ref = next(row for row in analysis.per_view if row["view"] == ds_module.REFERENCE_VIEW)
    scores = ref[presenter]["presenter_only"]
    assert scores["coverage"] == pytest.approx(1.0)
    assert scores["psnr"] > 100  # identical pixels
    assert scores["depth_rel_median"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("presenter", ds_module.PRESENTERS)
def test_outer_views_land_on_the_reference_geometry(analysis, presenter):
    """At the outer cameras the rendered depth matches the source depth."""
    for row in analysis.per_view:
        if row["view"] in OUTER_VIEWS:
            assert row[presenter]["presenter_only"]["coverage"] > 0.3
            assert row[presenter]["presenter_only"]["depth_rel_median"] < 0.01


def test_the_mesh_presenter_covers_at_least_what_the_points_do(analysis):
    """Triangles fill the pinholes between projected points, so coverage never drops."""
    summary = analysis.status()["summary"]
    assert summary["mesh"]["ds_image"]["coverage"] >= summary["points"]["ds_image"]["coverage"]


def test_summaries_average_only_the_outer_views(analysis):
    """Summaries skip the trivially exact reference view; LOO stays empty until run."""
    status = analysis.status()
    assert status["presenters"] == list(ds_module.PRESENTERS)
    for presenter in ds_module.PRESENTERS:
        summary = status["summary"][presenter]["presenter_only"]
        expected = np.mean([r[presenter]["presenter_only"]["coverage"]
                            for r in analysis.per_view if r["view"] != ds_module.REFERENCE_VIEW])
        assert summary["coverage"] == pytest.approx(expected)
        assert status["summary"][presenter]["leave_one_out"]["coverage"] is None


def test_analysis_encodes_renders_errors_and_layers(analysis):
    """Every render and error map (per presenter) and layer preview is a JPEG."""
    for view in range(9):
        for kind in ("render", "error"):
            for presenter in ds_module.PRESENTERS:
                image = analysis.images[f"{kind}/{presenter}/{view}"]
                assert Image.open(io.BytesIO(image)).format == "JPEG"
    for key in ("layer0_rgb", "layer0_depth", "layer1_rgb", "layer1_depth", "layer1_mask"):
        assert key in analysis.images


def test_the_quantized_codec_is_smaller_and_still_passes(consistent_spatial_photo_result, analysis):
    """uint16 depth shrinks the depth payload and still passes the round-trip."""
    quantized = ds_module.DSAnalysis(codec="uint16_quant_zlib")
    ds_module.run_analysis(quantized, consistent_spatial_photo_result)
    try:
        assert quantized.state == "done", quantized.error
        assert quantized.roundtrip["passed"] is True
        assert (quantized.file_metrics["base_depth_bytes"]
                < analysis.file_metrics["base_depth_bytes"])
    finally:
        quantized.cleanup()


def test_leave_one_out_scores_every_outer_view(analysis):
    """Leave-one-out adds a score to each of the 8 outer views."""
    ds_module.run_leave_one_out(analysis)
    assert analysis.loo_state == "done", analysis.loo_error
    for presenter in ds_module.PRESENTERS:
        scored = [row["view"] for row in analysis.per_view if "leave_one_out" in row[presenter]]
        assert scored == OUTER_VIEWS
        assert analysis.status()["summary"][presenter]["leave_one_out"]["coverage"] is not None


def test_leave_one_out_needs_the_source_views(analysis):
    """Without the source views, leave-one-out reports why instead of crashing."""
    analysis.source = None
    ds_module.run_leave_one_out(analysis)
    assert analysis.loo_state == "error"
    assert "rebuild" in analysis.loo_error


@pytest.mark.parametrize("presenter", ds_module.PRESENTERS)
def test_free_view_renders_a_jpeg_and_reports_coverage(analysis, presenter):
    """A free-view render is a full-size JPEG with a coverage share."""
    image, coverage = ds_module.render_free_view(analysis, (0.0, 0.0, 0.0), presenter)
    assert Image.open(io.BytesIO(image)).size == (analysis.asset.spatial_width,
                                                  analysis.asset.spatial_height)
    assert 0.0 < coverage <= 1.0


def test_a_failing_analysis_is_recorded_instead_of_raised():
    """A broken input ends in state 'error', not an exception."""
    broken = ds_module.DSAnalysis(codec="float32_zlib")
    ds_module.run_analysis(broken, object())
    assert broken.state == "error"
    assert broken.error


def test_score_render_counts_holes_as_coverage_not_colour_error(consistent_spatial_photo_result):
    """Holes lower coverage but not PSNR/SSIM."""
    view = ds_module.source_from_result(consistent_spatial_photo_result).views[4]
    coverage = view.depth_valid.copy()
    coverage[: coverage.shape[0] // 2] = False  # half the picture is a hole
    score = ds_module.score_render(view.rgb, coverage, view.depth, view)
    assert score["coverage"] == pytest.approx(coverage.sum() / view.depth_valid.sum())
    assert score["psnr"] > 100 and score["ssim"] == pytest.approx(1.0)


def test_error_image_marks_holes_blue_and_errors_red(consistent_spatial_photo_result):
    """The error map paints holes blue and colour differences red."""
    view = ds_module.source_from_result(consistent_spatial_photo_result).views[4]
    rgb = view.rgb.copy()
    rgb[0, 0] = 255 - rgb[0, 0]
    coverage = view.depth_valid.copy()
    hole = tuple(np.argwhere(view.depth_valid)[-1])
    coverage[hole] = False
    out = ds_module.error_image(rgb, coverage, view)
    assert tuple(out[hole]) == tuple(ds_module.HOLE_COLOR)
    if coverage[0, 0]:
        assert out[0, 0, 0] == 255


# ---------------------------------------------------------------- .h5 jobs


def test_an_uploaded_h5_becomes_a_finished_job(consistent_spatial_photo_result, tmp_path):
    """An uploaded spatial_photo.h5 fills in a finished job without the pipeline."""
    job = Job(job_id="h5", params=JobParams(filename="scene.h5"))
    load_h5_job(job, _h5_bytes(consistent_spatial_photo_result, tmp_path))
    try:
        assert job.state == "done", job.error
        assert job.from_h5 and job.status()["from_h5"]
        assert len(job.views) == len(job.depths) == 9
        assert job.h5_path.exists() and job.result is not None
    finally:
        job.cleanup()


def test_an_h5_without_nine_views_is_rejected(consistent_spatial_photo_result, tmp_path):
    """Files that do not hold exactly 9 views are refused."""
    r = consistent_spatial_photo_result
    three = SpatialPhotoResult(r.rgb[:3], r.depth[:3], r.mask[:3], r.K[:3], r.R[:3], r.C[:3],
                               r.metadata)
    job = Job(job_id="h5", params=JobParams(filename="scene.h5"))
    load_h5_job(job, _h5_bytes(three, tmp_path))
    assert job.state == "error" and "9 views" in job.error


def test_a_corrupt_h5_is_reported_as_an_error():
    """A file that is not HDF5 ends the job in an error state."""
    job = Job(job_id="h5", params=JobParams(filename="scene.h5"))
    load_h5_job(job, b"definitely not hdf5")
    assert job.state == "error"


# ---------------------------------------------------------------- HTTP


def _wait_ds(client, job_id: str, key: str = "state") -> dict:
    """Poll the DS status until the analysis (or, with key="loo", leave-one-out) ends."""
    deadline = time.monotonic() + POLL_TIMEOUT_S * 6
    while time.monotonic() < deadline:
        _status, ds = client.json(f"/api/jobs/{job_id}/ds")
        value = ds[key] if key == "state" else ds["loo"]["state"]
        if value in ("done", "error"):
            return ds
        time.sleep(0.05)
    raise AssertionError("DS analysis did not finish")


def test_config_advertises_the_ds_features(client):
    """The startup config tells the page the DS panels are available."""
    _status, config = client.json("/api/config")
    assert config["ds"]["available"] is True
    assert config["ds"]["codecs"] == list(ds_module.DEPTH_CODECS)


def test_a_ds_image_is_built_served_and_scored_over_http(client):
    """The full DS flow works over HTTP: build, files, free view, leave-one-out."""
    job = client.run_to_completion("/api/jobs", _jpeg_bytes())
    job_id = job["job_id"]
    status, started = client.json(f"/api/jobs/{job_id}/ds?codec=float32_zlib", method="POST")
    assert status == 202 and started["state"] == "running"
    ds = _wait_ds(client, job_id)
    assert ds["state"] == "done", ds["error"]
    assert ds["roundtrip"]["passed"] and ds["h5_bytes"] > 0

    status, headers, body = client.request(f"/api/jobs/{job_id}/ds/ds_image.jpg")
    assert status == 200 and body[:2] == b"\xff\xd8"
    assert "attachment" in headers["Content-Disposition"]
    _status, headers, _ = client.request(f"/api/jobs/{job_id}/ds/ds_image.jpg?inline=1")
    assert headers["Content-Disposition"].startswith("inline")
    for path in ("render/3.jpg", "error/0.jpg", "render/3.jpg?presenter=points",
                 "error/0.jpg?presenter=mesh", "layers/layer1_rgb.jpg"):
        status, headers, body = client.request(f"/api/jobs/{job_id}/ds/{path}")
        assert status == 200 and headers["Content-Type"] == "image/jpeg", path
    for presenter in ds_module.PRESENTERS:
        query = f"x=0&y=0&z=0&presenter={presenter}"
        status, headers, _ = client.request(f"/api/jobs/{job_id}/ds/free.jpg?{query}")
        assert status == 200 and 0 < float(headers["X-Coverage"]) <= 1
    assert client.json(f"/api/jobs/{job_id}")[1]["ds"]["state"] == "done"

    status, _ = client.json(f"/api/jobs/{job_id}/ds/loo", method="POST")
    assert status == 202
    ds = _wait_ds(client, job_id, key="loo")
    assert ds["loo"]["state"] == "done"
    assert ds["summary"]["mesh"]["leave_one_out"]["coverage"] is not None


@pytest.mark.parametrize("path,code", [
    ("render/9.jpg", 404), ("layers/nope.jpg", 404),
    ("free.jpg?x=abc", 400), ("free.jpg?x=1e6", 400),
    ("render/3.jpg?presenter=nope", 400), ("free.jpg?x=0&presenter=nope", 400),
])
def test_bad_ds_requests_are_rejected(client, path, code):
    """Unknown images and bad free-view offsets get clear error codes."""
    job_id = client.run_to_completion("/api/jobs", _jpeg_bytes())["job_id"]
    client.json(f"/api/jobs/{job_id}/ds", method="POST")
    _wait_ds(client, job_id)
    assert client.request(f"/api/jobs/{job_id}/ds/{path}")[0] == code


def test_ds_routes_need_a_built_ds_image_and_a_known_codec(client):
    """DS routes refuse to run before a build or with an unknown codec."""
    job_id = client.run_to_completion("/api/jobs", _jpeg_bytes())["job_id"]
    assert client.request(f"/api/jobs/{job_id}/ds")[0] == 404
    assert client.request(f"/api/jobs/{job_id}/ds/loo", method="POST")[0] == 409
    assert client.request(f"/api/jobs/{job_id}/ds?codec=png", method="POST")[0] == 400
    assert client.request("/api/jobs/nope/ds", method="POST")[0] == 404


def test_an_h5_upload_skips_the_pipeline(client, consistent_spatial_photo_result, tmp_path):
    """Posting a .h5 finishes a job without SHARP."""
    job = client.run_to_completion("/api/jobs?filename=scene.h5",
                                   _h5_bytes(consistent_spatial_photo_result, tmp_path))
    assert job["state"] == "done", job["error"]
    assert job["from_h5"] is True and job["num_views"] == 9
