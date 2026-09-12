"""Runs the unchanged Stage 1 pipeline on one decoded video frame (spec §10).

The frame is handed to Stage 1 exactly the way a user would: as a lossless PNG
through `sharp_spatialize.api.generate_spatial_photo`. The predictor is cached
across calls (`cache_predictor=True`) so the checkpoint loads once per run, and
the per-frame 3DGS scene lives only inside that call -- it is released as soon
as the 9 views are rendered (spec §29).
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .contract import SpatialFrame
from .video_io import DecodedFrame

DEFAULT_ANGLE_DEG = 10.0  # spec §11 initial viewing envelope: +-10 degrees


@dataclass(frozen=True)
class CameraConfig:
    """The Stage 1 virtual-camera configuration (spec §11: kept parameterized)."""

    angle_deg: float = DEFAULT_ANGLE_DEG

    def __post_init__(self) -> None:
        if not self.angle_deg > 0:
            raise ValueError(f"angle_deg must be positive, got {self.angle_deg}")


def load_camera_config(path: str | Path) -> CameraConfig:
    """Load a camera config JSON file, e.g. `{"angle_deg": 10}`."""
    data = json.loads(Path(path).read_text())
    unknown = sorted(set(data) - {"angle_deg"})
    if unknown:
        raise ValueError(f"unknown camera config keys in {path}: {unknown}")
    return CameraConfig(angle_deg=float(data.get("angle_deg", DEFAULT_ANGLE_DEG)))


@dataclass
class FrameTimings:
    """Where one frame's Stage 1 time went (spec §35).

    SHARP regresses the 3D Gaussians directly and Stage 1 unprojects them in
    the same `infer` call, so SHARP and 3DGS generation are timed together.
    """

    sharp_and_3dgs_s: float
    render_s: float  # 9-view RGB + depth rendering
    validate_s: float  # Stage 1 validation of the frame
    total_s: float  # everything, including the PNG hand-off
    peak_gpu_bytes: int | None


def _torch():
    # Only look at torch if Stage 1 already imported it; never import it here.
    return sys.modules.get("torch")


def _reset_peak_gpu() -> None:
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_gpu() -> int | None:
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return None


def _stage1_generate(image_path, **kwargs):
    from sharp_spatialize.api import generate_spatial_photo

    return generate_spatial_photo(image_path, **kwargs)


class Stage1Spatializer:
    """Callable: `DecodedFrame -> (SpatialFrame, FrameTimings)` via Stage 1."""

    def __init__(
        self,
        work_dir: str | Path,
        camera: CameraConfig = CameraConfig(),
        output_size: tuple[int, int] | None = None,
        checkpoint_path: str | Path | None = None,
        device: str = "default",
        precision: str = "fp32",
        generate: Callable | None = None,
    ):
        self.png_dir = Path(work_dir) / "png"
        self.png_dir.mkdir(parents=True, exist_ok=True)
        self.camera = camera
        self.output_size = output_size
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.precision = precision
        self._generate = generate or _stage1_generate

    def __call__(self, frame: DecodedFrame) -> tuple[SpatialFrame, FrameTimings]:
        png_path = self.png_dir / f"{frame.index:06d}.png"
        marks: dict[str, float] = {}

        def on_progress(stage: str, fraction: float) -> None:
            marks[stage] = time.perf_counter()

        width, height = self.output_size if self.output_size else (None, None)
        _reset_peak_gpu()
        start = time.perf_counter()
        try:
            Image.fromarray(frame.rgb).save(png_path, compress_level=1)
            result = self._generate(
                png_path,
                angle_deg=self.camera.angle_deg,
                output_width=width,
                output_height=height,
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                precision=self.precision,
                cache_predictor=True,
                on_progress=on_progress,
            )
        finally:
            png_path.unlink(missing_ok=True)
        total = time.perf_counter() - start

        def span(begin: str, end: str) -> float:
            if begin in marks and end in marks:
                return marks[end] - marks[begin]
            return 0.0

        timings = FrameTimings(
            sharp_and_3dgs_s=span("inference", "rendering"),
            render_s=span("rendering", "validating"),
            validate_s=span("validating", "done"),
            total_s=total,
            peak_gpu_bytes=_peak_gpu(),
        )
        spatial_frame = SpatialFrame.from_stage1(
            result, timestamp=frame.timestamp, pts=frame.pts, source_frame_index=frame.index
        )
        return spatial_frame, timings
