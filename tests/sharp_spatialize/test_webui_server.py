"""Tests for sharp_spatialize.webui.server.

These drive a real ThreadingHTTPServer over a real socket on an ephemeral
port, with the pipeline replaced by a fixed result, so the routing, status
polling, and download paths are exercised without SHARP or a GPU.
"""

from __future__ import annotations

import io
import json
import threading
import time
import urllib.error
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer

import pytest
from PIL import Image
from sharp_spatialize.webui import server as server_module
from sharp_spatialize.webui.jobs import JobStore, run_job
from sharp_spatialize.webui.server import ServerConfig, SpatialPhotoHandler

POLL_TIMEOUT_S = 10.0


def _jpeg_bytes(width: int = 64, height: int = 48) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 120, 40)).save(buffer, format="JPEG")
    return buffer.getvalue()


class Client:
    """Minimal HTTP client bound to one running test server."""

    def __init__(self, base_url: str) -> None:
        """Point the client at a running server's base URL."""
        self.base_url = base_url

    def request(self, path: str, method: str = "GET", body: bytes | None = None):
        """Return (status, headers, body_bytes); HTTP errors come back, not raised."""
        request = urllib.request.Request(self.base_url + path, data=body, method=method)
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers, error.read()

    def json(self, path: str, method: str = "GET", body: bytes | None = None):
        """Same as `request`, with the body parsed as JSON."""
        status, _headers, payload = self.request(path, method=method, body=body)
        return status, json.loads(payload)

    def run_to_completion(self, path: str, body: bytes) -> dict:
        """POST a job, poll until it leaves the running states, and return its status."""
        status, payload = self.json(path, method="POST", body=body)
        assert status == 202, payload
        job_id = payload["job_id"]

        deadline = time.monotonic() + POLL_TIMEOUT_S
        while time.monotonic() < deadline:
            _status, job = self.json(f"/api/jobs/{job_id}")
            if job["state"] in ("done", "error"):
                return job
            time.sleep(0.02)
        raise AssertionError(f"job {job_id} did not finish within {POLL_TIMEOUT_S}s")


@pytest.fixture
def client(consistent_spatial_photo_result, tmp_path, monkeypatch):
    """A running web UI whose pipeline returns a fixed, valid result."""

    def fake_generate(input_path, *, on_progress=None, **kwargs):
        if on_progress is not None:
            on_progress("inference", 0.5)
        return consistent_spatial_photo_result

    # The server calls run_job without a `generate`; bind the fake one here.
    monkeypatch.setattr(
        server_module, "run_job", partial(run_job, generate=fake_generate)
    )

    sample = tmp_path / "sample.jpg"
    sample.write_bytes(_jpeg_bytes(32, 32))

    store = JobStore()
    config = ServerConfig(
        store=store,
        checkpoint_path=None,
        device="cpu",
        default_precision="fp16",
        sample_image=sample,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(SpatialPhotoHandler, config=config))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield Client(f"http://127.0.0.1:{httpd.server_address[1]}")
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        store.clear()


# ------------------------------------------------------------------ serving


def test_index_and_static_assets_are_served(client):
    """The root serves the page, and its CSS/JS come back with sensible content types."""
    status, headers, body = client.request("/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"<title>" in body

    for name, prefix in (("style.css", "text/css"), ("app.js", "")):
        status, headers, body = client.request(f"/static/{name}")
        assert status == 200, name
        assert body, name
        assert "charset=utf-8" in headers["Content-Type"], name
        assert headers["Content-Type"].startswith(prefix), name


def test_static_serving_refuses_to_escape_its_directory(client):
    """A traversal path must not reach files outside static/."""
    for path in ("/static/..%2f..%2fjobs.py", "/static/../jobs.py"):
        status, _headers, _body = client.request(path)
        assert status == 404, path


def test_config_reports_defaults_and_choices(client):
    """/api/config gives the client everything it needs to build its controls."""
    status, config = client.json("/api/config")

    assert status == 200
    assert config["defaults"]["precision"] == "fp16"
    assert config["defaults"]["max_size"] in config["max_size_choices"]
    assert config["sample_available"] is True
    assert config["max_angle_deg"] > 0
    assert "cuda" in config["device"]


def test_unknown_routes_return_a_json_404(client):
    """Every miss is JSON, so the client's error handling has one shape."""
    for path, method in (("/nope", "GET"), ("/api/nope", "GET"), ("/api/nope", "POST")):
        status, payload = client.json(path, method=method, body=b"" if method == "POST" else None)
        assert status == 404, path
        assert "error" in payload, path


def test_head_returns_headers_without_a_body(client):
    """HEAD on the index reports the length but sends nothing."""
    status, headers, body = client.request("/", method="HEAD")

    assert status == 200
    assert int(headers["Content-Length"]) > 0
    assert body == b""


# --------------------------------------------------------------- job flow


def test_a_job_runs_end_to_end_and_serves_its_previews(client):
    """POST an image, poll to done, then fetch the source, views, and depth previews."""
    job = client.run_to_completion("/api/jobs?filename=photo.jpg", _jpeg_bytes())

    assert job["state"] == "done", job.get("error")
    assert job["num_views"] == 9
    assert job["progress"] == 1.0
    assert job["source"]["filename"] == "photo.jpg"
    assert job["metadata"]["num_views"] == 9
    assert job["has_arrays"] is True

    job_id = job["job_id"]
    for path in (f"/api/jobs/{job_id}/source.jpg", f"/api/jobs/{job_id}/views/0.jpg",
                 f"/api/jobs/{job_id}/depths/8.jpg"):
        status, headers, body = client.request(path)
        assert status == 200, path
        assert headers["Content-Type"] == "image/jpeg", path
        assert Image.open(io.BytesIO(body)).format == "JPEG", path


def test_the_h5_download_is_a_real_file_with_a_named_attachment(client):
    """The .h5 is written on demand and offered under a name derived from the upload."""
    job = client.run_to_completion("/api/jobs?filename=holiday.jpg", _jpeg_bytes())

    status, headers, body = client.request(f"/api/jobs/{job['job_id']}/spatial_photo.h5")

    assert status == 200
    assert headers["Content-Type"] == "application/x-hdf5"
    assert 'filename="holiday_spatial_photo.h5"' in headers["Content-Disposition"]
    assert body[:8] == b"\x89HDF\r\n\x1a\n", "should be a real HDF5 file"


def test_the_sample_image_can_be_run_without_an_upload(client):
    """?sample=1 runs the bundled photo instead of the request body."""
    job = client.run_to_completion("/api/jobs?sample=1", b"\x00")

    assert job["state"] == "done", job.get("error")
    assert job["source"]["filename"] == "sample.jpg"


def test_query_parameters_choose_the_angle_and_precision(client):
    """The angle and precision from the query end up on the job."""
    job = client.run_to_completion(
        "/api/jobs?angle=6.5&precision=bf16&max_size=640", _jpeg_bytes()
    )

    assert job["angle_deg"] == 6.5
    assert job["precision"] == "bf16"


def test_a_failing_pipeline_is_reported_as_an_error_state(client, monkeypatch):
    """A pipeline exception reaches the client as state="error", not a dead poll."""
    def _boom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(server_module, "run_job", partial(run_job, generate=_boom))

    job = client.run_to_completion("/api/jobs", _jpeg_bytes())

    assert job["state"] == "error"
    assert "CUDA out of memory" in job["error"]


# ------------------------------------------------------------ bad requests


@pytest.mark.parametrize(
    "query",
    ["?angle=0", "?angle=-4", "?angle=999", "?max_size=12345", "?precision=int8"],
)
def test_invalid_parameters_are_rejected_before_any_work_starts(client, query):
    """Out-of-range params get a 400 with an explanation, not a queued job."""
    status, payload = client.json(f"/api/jobs{query}", method="POST", body=_jpeg_bytes())

    assert status == 400
    assert "error" in payload


def test_an_empty_upload_is_rejected(client):
    """A zero-length body cannot be an image."""
    status, payload = client.json("/api/jobs", method="POST", body=b"")

    assert status == 400
    assert "empty" in payload["error"]


def test_an_oversized_upload_is_rejected_without_being_read(client, monkeypatch):
    """The size cap is enforced from Content-Length, before the body is buffered."""
    monkeypatch.setattr(server_module, "MAX_UPLOAD_BYTES", 1024)

    status, payload = client.json("/api/jobs", method="POST", body=b"x" * 4096)

    assert status == 413
    assert "larger than" in payload["error"]


def test_requests_for_unknown_or_expired_jobs_return_404(client):
    """An id the store never had, or has dropped, is a clean 404."""
    status, payload = client.json("/api/jobs/deadbeef")
    assert status == 404
    assert "error" in payload

    status, payload = client.json("/api/jobs/deadbeef/views/0.jpg")
    assert status == 404


def test_out_of_range_and_malformed_view_indices_are_rejected(client):
    """View indices are bounds-checked and must be numbers."""
    job = client.run_to_completion("/api/jobs", _jpeg_bytes())
    job_id = job["job_id"]

    status, _payload = client.json(f"/api/jobs/{job_id}/views/99.jpg")
    assert status == 404

    status, _payload = client.json(f"/api/jobs/{job_id}/views/abc.jpg")
    assert status == 400

    status, _payload = client.json(f"/api/jobs/{job_id}/nonsense")
    assert status == 404
