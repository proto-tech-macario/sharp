"""Job model, image encoding, and pipeline execution for the web UI.

Kept free of HTTP concerns so it can be tested directly. Only `run_job`
touches the GPU pipeline; everything else here is pure numpy/Pillow and runs
anywhere.
"""

from __future__ import annotations

import io
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

try:  # phone photos are commonly HEIC; pillow_heif is already a SHARP dependency
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - optional decoder, never fatal
    pass

MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_SIZE_CHOICES = (640, 960, 1280, 1600, 2048)
DEFAULT_MAX_SIZE = 1280
DEFAULT_ANGLE_DEG = 10.0
MAX_ANGLE_DEG = 30.0
JPEG_QUALITY = 88

#: Share of the progress bar owned by the pipeline; encoding previews fills the rest.
PIPELINE_PROGRESS_SHARE = 0.92

#: Jobs kept in the store at all; older ones are dropped entirely.
MAX_JOBS = 6
#: Jobs that keep their full float arrays (needed to write the .h5 on demand).
MAX_JOBS_WITH_ARRAYS = 2

EXIF_ORIENTATION_TAG = 274

# Google's "turbo" colormap, sampled at 9 evenly spaced stops. Interpolating
# between these is visually indistinguishable from the real thing at 8-bit
# depth, and saves depending on matplotlib inside the server.
_TURBO_STOPS = np.array(
    [
        (48, 18, 59), (70, 107, 227), (54, 174, 248), (29, 224, 200), (109, 250, 118),
        (191, 230, 57), (248, 168, 39), (238, 91, 15), (122, 4, 3),
    ],
    dtype=np.float32,
)

INVALID_DEPTH_COLOR = np.array((24, 26, 32), dtype=np.uint8)


def turbo(values: np.ndarray) -> np.ndarray:
    """Map an array of floats in [0, 1] through the turbo colormap to uint8 RGB."""
    positions = np.clip(values, 0.0, 1.0) * (len(_TURBO_STOPS) - 1)
    lower = np.floor(positions).astype(np.int32)
    upper = np.minimum(lower + 1, len(_TURBO_STOPS) - 1)
    blend = (positions - lower)[..., None]
    colors = _TURBO_STOPS[lower] * (1.0 - blend) + _TURBO_STOPS[upper] * blend
    return np.rint(colors).astype(np.uint8)


def depth_to_rgb(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Colorize [N,H,W] metric depth as [N,H,W,3] uint8 for display.

    Colors *inverse* depth (disparity), which is what makes near-field
    structure readable in scenes with a distant background, and normalizes
    across all views at once so the nine images stay directly comparable.
    Invalid pixels get a flat dark color rather than a colormap value.
    """
    valid = mask.astype(bool) & (depth > 0)
    colored = np.broadcast_to(INVALID_DEPTH_COLOR, (*depth.shape, 3)).copy()
    if not valid.any():
        return colored

    disparity = np.zeros_like(depth, dtype=np.float32)
    disparity[valid] = 1.0 / depth[valid]
    low, high = np.percentile(disparity[valid], (2.0, 98.0))
    if high <= low:
        low, high = float(disparity[valid].min()), float(disparity[valid].max())
    if high <= low:
        high = low + 1e-6

    normalized = (disparity - low) / (high - low)
    colored[valid] = turbo(normalized)[valid]
    return colored


def encode_jpeg(rgb: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """Encode one [H,W,3] uint8 array as JPEG bytes."""
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def fit_within(width: int, height: int, max_size: int) -> tuple[int, int]:
    """Scale (width, height) down so the long edge is at most `max_size`."""
    longest = max(width, height)
    if longest <= max_size:
        return width, height
    scale = max_size / longest
    return max(1, round(width * scale)), max(1, round(height * scale))


def prepare_source_image(data: bytes, max_size: int) -> tuple[Image.Image, dict[str, Any]]:
    """Decode an upload into the upright, size-capped image the pipeline will see.

    EXIF orientation is applied here and then cleared, so `sharp.utils.io.load_rgb`
    does not rotate the image a second time. The remaining EXIF (notably focal
    length, which sets the scene's scale) is preserved.
    """
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image)
    original_width, original_height = image.size

    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    width, height = fit_within(original_width, original_height, max_size)
    if (width, height) != (original_width, original_height):
        image = image.resize((width, height), Image.LANCZOS)

    exif = image.getexif()
    exif[EXIF_ORIENTATION_TAG] = 1

    info = {
        "original_width": original_width,
        "original_height": original_height,
        "width": width,
        "height": height,
        "downscaled": (width, height) != (original_width, original_height),
    }
    return image, info


@dataclass(frozen=True)
class JobParams:
    """Everything the user can choose about one generation."""

    angle_deg: float = DEFAULT_ANGLE_DEG
    max_size: int = DEFAULT_MAX_SIZE
    precision: str = "fp16"
    filename: str = "upload"

    def validated(self) -> JobParams:
        """Return these params clamped/checked, or raise ValueError."""
        if not 0 < self.angle_deg <= MAX_ANGLE_DEG:
            raise ValueError(f"angle must be in (0, {MAX_ANGLE_DEG}], got {self.angle_deg}")
        if self.max_size not in MAX_SIZE_CHOICES:
            raise ValueError(f"max_size must be one of {MAX_SIZE_CHOICES}, got {self.max_size}")
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"unknown precision {self.precision!r}")
        return self


@dataclass
class Job:
    """One image-to-spatial-photo run and its results."""

    job_id: str
    params: JobParams
    state: str = "queued"  # queued | running | done | error
    stage: str = "queued"
    progress: float = 0.0
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    timings: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_info: dict[str, Any] = field(default_factory=dict)
    views: list[bytes] = field(default_factory=list)
    depths: list[bytes] = field(default_factory=list)
    source_jpeg: bytes | None = None
    result: Any = None  # SpatialPhotoResult, kept only for the newest few jobs
    h5_path: Path | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def report(self, stage: str, progress: float) -> None:
        """Progress callback handed to `generate_spatial_photo`."""
        with self._lock:
            self.stage = stage
            self.progress = progress

    def status(self) -> dict[str, Any]:
        """A JSON-serializable snapshot for the client to poll."""
        with self._lock:
            return {
                "job_id": self.job_id,
                "state": self.state,
                "stage": self.stage,
                "progress": round(self.progress, 3),
                "error": self.error,
                "timings": {name: round(value, 2) for name, value in self.timings.items()},
                "metadata": self.metadata,
                "source": self.source_info,
                "num_views": len(self.views),
                "angle_deg": self.params.angle_deg,
                "precision": self.params.precision,
                "has_arrays": self.result is not None,
            }

    def release_arrays(self) -> None:
        """Drop the heavy float arrays, keeping the encoded previews."""
        self.result = None

    def cleanup(self) -> None:
        """Release everything this job holds, including its temp file."""
        self.release_arrays()
        self.views = []
        self.depths = []
        self.source_jpeg = None
        if self.h5_path is not None:
            shutil.rmtree(self.h5_path.parent, ignore_errors=True)
            self.h5_path = None


class JobStore:
    """Thread-safe, bounded store of recent jobs."""

    def __init__(self, max_jobs: int = MAX_JOBS, max_jobs_with_arrays: int = MAX_JOBS_WITH_ARRAYS):
        """Keep at most `max_jobs`, with float arrays for the newest `max_jobs_with_arrays`."""
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._max_jobs = max_jobs
        self._max_jobs_with_arrays = max_jobs_with_arrays

    def create(self, params: JobParams) -> Job:
        """Register a new queued job and evict whatever no longer fits."""
        job = Job(job_id=uuid.uuid4().hex[:12], params=params)
        with self._lock:
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
            evicted = [self._jobs.pop(job_id) for job_id in self._order[: -self._max_jobs]]
            self._order = self._order[-self._max_jobs :]
            stale = [
                self._jobs[job_id]
                for job_id in self._order[: -self._max_jobs_with_arrays]
                if job_id in self._jobs
            ]
        for old in evicted:
            old.cleanup()
        for old in stale:
            old.release_arrays()
        return job

    def get(self, job_id: str) -> Job | None:
        """Look up a job by id."""
        with self._lock:
            return self._jobs.get(job_id)

    def clear(self) -> None:
        """Drop every job (used on shutdown)."""
        with self._lock:
            jobs = list(self._jobs.values())
            self._jobs.clear()
            self._order.clear()
        for job in jobs:
            job.cleanup()


#: Serializes GPU work: one 6GB card, one job at a time.
GPU_LOCK = threading.Lock()


def pipeline_reporter(job: Job) -> Callable[[str, float], None]:
    """Scale `generate_spatial_photo`'s 0..1 progress into the share it owns.

    The pipeline finishes with `("done", 1.0)`, but the previews still have to
    be encoded after it returns. Reported verbatim, that would fill the bar and
    then visibly step back, so the pipeline's own progress is compressed into
    `PIPELINE_PROGRESS_SHARE` and its "done" is dropped.
    """

    def report(stage: str, fraction: float) -> None:
        if stage == "done":
            return
        job.report(stage, fraction * PIPELINE_PROGRESS_SHARE)

    return report


def run_job(
    job: Job,
    image_bytes: bytes,
    checkpoint_path: Path | None = None,
    device: str = "default",
    generate: Callable[..., Any] | None = None,
) -> None:
    """Run the full pipeline for `job`, filling in its results in place.

    Never raises: a failure is recorded on the job as `state="error"`.
    `generate` is injectable so this can be tested without SHARP or a GPU.
    """
    temp_dir: Path | None = None
    try:
        image, source_info = prepare_source_image(image_bytes, job.params.max_size)
        job.source_info = dict(source_info, filename=job.params.filename)
        job.source_jpeg = encode_jpeg(np.asarray(image.convert("RGB")))

        temp_dir = Path(tempfile.mkdtemp(prefix="sharp-webui-"))
        input_path = temp_dir / f"input{Path(job.params.filename).suffix or '.jpg'}"
        if input_path.suffix.lower() not in (".jpg", ".jpeg"):
            input_path = temp_dir / "input.jpg"
        image.convert("RGB").save(input_path, format="JPEG", quality=95, exif=image.getexif())

        if generate is None:  # imported lazily: pulls in SHARP, torch, and gsplat
            from ..api import generate_spatial_photo as generate

        with GPU_LOCK:
            job.state = "running"
            job.report("inference", 0.05)
            started = time.perf_counter()
            result = generate(
                input_path,
                angle_deg=job.params.angle_deg,
                checkpoint_path=checkpoint_path,
                device=device,
                precision=job.params.precision,
                cache_predictor=True,
                on_progress=pipeline_reporter(job),
            )
            job.timings["pipeline"] = time.perf_counter() - started

        job.report("encoding", 0.94)
        started = time.perf_counter()
        job.views = [encode_jpeg(view) for view in result.rgb]
        job.depths = [encode_jpeg(view) for view in depth_to_rgb(result.depth, result.mask)]
        job.timings["encode"] = time.perf_counter() - started

        job.metadata = {
            key: value.item() if hasattr(value, "item") else value
            for key, value in result.metadata.items()
        }
        job.result = result
        job.h5_path = temp_dir / "spatial_photo.h5"
        temp_dir = None  # ownership passes to the job

        job.state = "done"
        job.report("done", 1.0)
    except BaseException as exc:  # noqa: BLE001 - surfaced to the client verbatim
        job.error = f"{type(exc).__name__}: {exc}"
        job.state = "error"
        job.report("error", job.progress)
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
        _free_gpu_memory()


def _free_gpu_memory() -> None:
    """Return cached CUDA blocks to the driver between jobs (best effort)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - torch missing or CUDA unavailable
        pass


def ensure_h5(job: Job) -> Path:
    """Write (once) and return the job's spatial_photo.h5 path."""
    if job.h5_path is None:
        raise FileNotFoundError("this job's results are no longer available")
    if not job.h5_path.exists():
        if job.result is None:
            raise FileNotFoundError(
                "this job's arrays have been released; re-run the image to download its .h5"
            )
        job.result.save(job.h5_path)
    return job.h5_path
