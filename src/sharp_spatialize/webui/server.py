"""The HTTP layer of the local spatial-photo web UI.

A `ThreadingHTTPServer` from the standard library: no framework, no install
step. Generation happens on a worker thread per job, serialized by
`jobs.GPU_LOCK`, so status polls stay responsive while the GPU is busy.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import webbrowser
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import jobs as jobs_module
from .jobs import (
    DEFAULT_ANGLE_DEG,
    DEFAULT_MAX_SIZE,
    MAX_ANGLE_DEG,
    MAX_SIZE_CHOICES,
    MAX_UPLOAD_BYTES,
    Job,
    JobParams,
    JobStore,
    ensure_h5,
    run_job,
)

STATIC_DIR = Path(__file__).parent / "static"
#: The repo's bundled sample image, when running from a source checkout.
SAMPLE_IMAGE = Path(__file__).resolve().parents[3] / "data" / "teaser.jpg"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8737


class SpatialPhotoHandler(BaseHTTPRequestHandler):
    """Serves the single-page UI and its small JSON/image API."""

    server_version = "sharp-spatialize-webui"
    protocol_version = "HTTP/1.1"

    def __init__(self, *args: Any, config: ServerConfig, **kwargs: Any) -> None:
        """Bind the shared `config` before the base class starts handling the request."""
        self.config = config
        super().__init__(*args, **kwargs)

    # -- plumbing ---------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        """Quiet the per-request log; status polling would drown out anything useful."""

    def log_error(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        """Keep real errors visible on the console."""
        super().log_message(format, *args)

    def _send(
        self, status: HTTPStatus, body: bytes, content_type: str, extra_headers: dict | None = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"error": message})

    # -- routing ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        """Route GET requests."""
        parsed = urlparse(self.path)
        segments = [segment for segment in parsed.path.split("/") if segment]
        try:
            if not segments:
                return self._serve_static("index.html")
            if segments[0] == "static" and len(segments) == 2:
                return self._serve_static(segments[1])
            if segments[0] == "api":
                return self._route_api_get(segments[1:])
        except BrokenPipeError:  # pragma: no cover - client navigated away mid-response
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, f"no route for {parsed.path}")

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib signature
        """HEAD is handled exactly like GET, minus the body."""
        self.do_GET()

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        """Route POST requests."""
        parsed = urlparse(self.path)
        segments = [segment for segment in parsed.path.split("/") if segment]
        if segments == ["api", "jobs"]:
            return self._create_job(parse_qs(parsed.query))
        self._send_error_json(HTTPStatus.NOT_FOUND, f"no route for {parsed.path}")

    def _route_api_get(self, segments: list[str]) -> None:
        if segments == ["config"]:
            return self._send_json(HTTPStatus.OK, self.config.describe())

        if segments and segments[0] == "jobs" and len(segments) >= 2:
            job = self.config.store.get(segments[1])
            if job is None:
                return self._send_error_json(HTTPStatus.NOT_FOUND, "unknown or expired job")
            return self._route_job_get(job, segments[2:])

        self._send_error_json(HTTPStatus.NOT_FOUND, "no such endpoint")

    def _route_job_get(self, job: Job, segments: list[str]) -> None:
        if not segments:
            return self._send_json(HTTPStatus.OK, job.status())

        if segments == ["source.jpg"]:
            if job.source_jpeg is None:
                return self._send_error_json(HTTPStatus.NOT_FOUND, "no source image")
            return self._send(HTTPStatus.OK, job.source_jpeg, "image/jpeg")

        if segments == ["spatial_photo.h5"]:
            return self._serve_h5(job)

        if len(segments) == 2 and segments[0] in ("views", "depths"):
            images = job.views if segments[0] == "views" else job.depths
            try:
                index = int(Path(segments[1]).stem)
            except ValueError:
                return self._send_error_json(HTTPStatus.BAD_REQUEST, "bad view index")
            if not 0 <= index < len(images):
                return self._send_error_json(HTTPStatus.NOT_FOUND, f"no view {index}")
            return self._send(HTTPStatus.OK, images[index], "image/jpeg")

        self._send_error_json(HTTPStatus.NOT_FOUND, "no such job resource")

    # -- handlers ---------------------------------------------------------

    def _serve_static(self, name: str) -> None:
        path = (STATIC_DIR / name).resolve()
        if path.parent != STATIC_DIR.resolve() or not path.is_file():
            return self._send_error_json(HTTPStatus.NOT_FOUND, f"no asset {name}")
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type.endswith("javascript"):
            content_type += "; charset=utf-8"
        self._send(HTTPStatus.OK, path.read_bytes(), content_type)

    def _serve_h5(self, job: Job) -> None:
        try:
            path = ensure_h5(job)
        except FileNotFoundError as exc:
            return self._send_error_json(HTTPStatus.GONE, str(exc))
        stem = Path(job.source_info.get("filename", "spatial_photo")).stem
        self._send(
            HTTPStatus.OK,
            path.read_bytes(),
            "application/x-hdf5",
            {"Content-Disposition": f'attachment; filename="{stem}_spatial_photo.h5"'},
        )

    def _create_job(self, query: dict[str, list[str]]) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self._send_error_json(HTTPStatus.BAD_REQUEST, "empty upload")
        if length > MAX_UPLOAD_BYTES:
            return self._send_error_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"image is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
            )

        def first(name: str, default: str) -> str:
            return query.get(name, [default])[0]

        use_sample = first("sample", "") == "1"
        if use_sample and self.config.sample_image is not None:
            image_bytes = self.config.sample_image.read_bytes()
            filename = self.config.sample_image.name
            self.rfile.read(length)  # drain the placeholder body
        else:
            image_bytes = self.rfile.read(length)
            filename = first("filename", "upload.jpg")

        try:
            params = JobParams(
                angle_deg=float(first("angle", str(DEFAULT_ANGLE_DEG))),
                max_size=int(first("max_size", str(DEFAULT_MAX_SIZE))),
                precision=first("precision", self.config.default_precision),
                filename=Path(filename).name,
            ).validated()
        except ValueError as exc:
            return self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

        job = self.config.store.create(params)
        thread = threading.Thread(
            target=run_job,
            args=(job, image_bytes),
            kwargs={
                "checkpoint_path": self.config.checkpoint_path,
                "device": self.config.device,
            },
            daemon=True,
            name=f"spatialize-{job.job_id}",
        )
        thread.start()
        print(f"  job {job.job_id}: {params.filename} @ {params.angle_deg}deg {params.precision}")
        self._send_json(HTTPStatus.ACCEPTED, {"job_id": job.job_id})


class ServerConfig:
    """Everything the handler needs that is not per-request."""

    def __init__(
        self,
        store: JobStore,
        checkpoint_path: Path | None,
        device: str,
        default_precision: str,
        sample_image: Path | None,
    ) -> None:
        """Hold the job store and the pipeline settings every request shares."""
        self.store = store
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.default_precision = default_precision
        self.sample_image = sample_image

    def describe(self) -> dict[str, Any]:
        """The client's startup payload: defaults and what this machine can do."""
        return {
            "device": describe_device(),
            "sample_available": self.sample_image is not None,
            "defaults": {
                "angle_deg": DEFAULT_ANGLE_DEG,
                "max_size": DEFAULT_MAX_SIZE,
                "precision": self.default_precision,
            },
            "max_angle_deg": MAX_ANGLE_DEG,
            "max_size_choices": list(MAX_SIZE_CHOICES),
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        }


def describe_device() -> dict[str, Any]:
    """Report the CUDA device, if any, for the status chip in the UI."""
    try:
        import torch
    except Exception:  # pragma: no cover - torch is a hard dependency in practice
        return {"cuda": False, "name": "torch unavailable"}

    if not torch.cuda.is_available():
        return {"cuda": False, "name": "no CUDA device (rendering will fail)"}
    properties = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "name": properties.name,
        "vram_gb": round(properties.total_memory / 1e9, 1),
    }


def serve(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    checkpoint_path: Path | None = None,
    device: str = "default",
    precision: str = "fp16",
    open_browser: bool = False,
) -> None:
    """Run the web UI until interrupted."""
    sample = SAMPLE_IMAGE if SAMPLE_IMAGE.is_file() else None
    store = JobStore()
    config = ServerConfig(store, checkpoint_path, device, precision, sample)
    handler = partial(SpatialPhotoHandler, config=config)

    with ThreadingHTTPServer((host, port), handler) as httpd:
        url = f"http://{host}:{port}/"
        device_info = describe_device()
        print(f"sharp-spatialize web UI -> {url}")
        print(f"  device:    {device_info['name']}")
        print(f"  precision: {precision}")
        print(f"  sample:    {sample if sample else 'not available'}")
        print("  Ctrl-C to stop.")
        if open_browser:
            threading.Timer(0.5, webbrowser.open, args=(url,)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nshutting down.")
        finally:
            store.clear()
            jobs_module._free_gpu_memory()


def main(argv: list[str] | None = None) -> None:
    """Entry point for `sharp-spatialize-webui` and `python -m sharp_spatialize.webui`."""
    parser = argparse.ArgumentParser(description="Preview spatial photos in a browser.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind address (default: %(default)s).")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port (default: %(default)s)."
    )
    parser.add_argument(
        "-c", "--checkpoint", type=Path, default=None,
        help="SHARP checkpoint (downloads the default model if omitted).",
    )
    parser.add_argument(
        "--device", default="default", help="cpu, mps, cuda, or default (auto-detect)."
    )
    parser.add_argument(
        "--precision", choices=["fp32", "fp16", "bf16"], default="fp16",
        help="Default forward-pass precision (default: %(default)s; fp16 halves peak VRAM).",
    )
    parser.add_argument("--open", action="store_true", help="Open a browser on startup.")
    args = parser.parse_args(argv)

    serve(
        host=args.host,
        port=args.port,
        checkpoint_path=args.checkpoint,
        device=args.device,
        precision=args.precision,
        open_browser=args.open,
    )
