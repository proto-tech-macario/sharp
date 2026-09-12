"""Streaming MIV encoder: SpatialFrame sequence -> MIV bitstream (spec §22-§27).

    encoder = MIVEncoder(MIVEncoderConfig(work_dir="clip.work"))
    encoder.begin(fps=fps, width=width, height=height, num_views=9)
    for frame in spatial_sequence:
        encoder.encode_frame(timestamp=frame.timestamp, views=frame.views)
    encoder.finalize("output.miv")

`encode_frame` converts one frame at a time and appends it to per-view raw
YUV files plus a per-frame TMIV camera file, so memory stays at one frame no
matter how long the clip is (spec §26). `finalize` runs TMIV (encoder -> VVenC
per atlas component -> multiplexer) through its `encode.py` and writes the
multiplexed MIV bitstream plus a JSON manifest that records every frame's
timestamp. Everything MIV-specific -- atlas construction, occupancy, depth
conversion, camera packaging -- stays inside this package (spec §23).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np

from ..contract import NUM_VIEWS, SpatialView
from .camera import tmiv_view_json
from .depth import depth_to_geometry
from .tmiv import TMIV_VERSION, PATCH_NAME, TmivInstall, find_tmiv, run_logged
from .tmiv_config import (
    CONTENT_ID,
    GEOMETRY_BIT_DEPTH,
    RATE_ID,
    TEXTURE_BIT_DEPTH,
    VIEW_NAMES,
    bitstream_output_path,
    encoder_config,
    geometry_input_path,
    multiplexer_config,
    qp_csv,
    sequence_config,
    sequence_config_path,
    texture_input_path,
)
from .yuv import geometry_planes, rgb_to_yuv420, write_yuv420

Runner = Callable[..., None]

MIV_FORMAT = (
    f"MIV (ISO/IEC 23090-12) V3C sample stream with in-band VVC video, "
    f"TMIV {TMIV_VERSION} + {PATCH_NAME}"
)
_K_TOLERANCE = 1e-6
# TMIV's encode.py runs VVenC with GOP size = intra period in random-access mode;
# VVenC's automatic GOP configuration supports that only for GOP 16 and 32.
VALID_INTRA_PERIODS = (16, 32)


@dataclass
class MIVEncoderConfig:
    """Settings for `MIVEncoder`."""

    work_dir: Path
    tmiv: TmivInstall | None = None  # default: find_tmiv() at finalize
    depth_near: float = 0.1  # metres; the geometry range is fixed for the whole clip
    depth_far: float = 1000.0
    qp_texture: int = 22
    qp_geometry: int = 8
    intra_period: int = 32
    threads: int = 4

    def __post_init__(self) -> None:
        self.work_dir = Path(self.work_dir)
        if self.intra_period not in VALID_INTRA_PERIODS:
            raise ValueError(
                f"intra_period must be one of {VALID_INTRA_PERIODS} (TMIV runs VVenC in "
                f"random access with GOP size = intra period, which VVenC supports only for "
                f"these), got {self.intra_period}"
            )
        if not 0 < self.depth_near < self.depth_far:
            raise ValueError(
                f"need 0 < depth_near < depth_far, got {self.depth_near}, {self.depth_far}"
            )


@dataclass
class MIVEncodeResult:
    """What `finalize` produced, plus the MIV encoding statistics of spec §35."""

    path: Path
    manifest_path: Path
    frame_count: int
    fps: float
    duration_s: float
    size_bytes: int
    bitrate_bps: float
    encode_time_s: float  # includes streaming conversion and the TMIV run
    tmiv_time_s: float
    cfr: bool
    clamped_depth_pixels: int
    stats: dict = field(default_factory=dict)


class MIVEncoder:
    """Accepts spatial frames one at a time and produces one MIV file."""

    def __init__(self, config: MIVEncoderConfig, runner: Runner = run_logged):
        self.config = config
        self._runner = runner
        self._started = False
        self._files: dict[str, BinaryIO] = {}
        self._frames: list[dict] = []
        self._K: np.ndarray | None = None
        self._clamped = 0
        self._convert_time = 0.0

    @property
    def input_dir(self) -> Path:
        return self.config.work_dir / "input"

    @property
    def output_dir(self) -> Path:
        return self.config.work_dir / "output"

    def begin(self, fps: float, width: int, height: int, num_views: int = NUM_VIEWS) -> None:
        """Start a sequence of `width` x `height` views at `fps`."""
        if num_views != NUM_VIEWS:
            raise ValueError(f"the spatial frame layout has exactly 9 views, got {num_views}")
        if width % 2 or height % 2:
            raise ValueError(f"view width and height must be even for 4:2:0, got {width}x{height}")
        if not fps > 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.fps, self.width, self.height = float(fps), int(width), int(height)

        shutil.rmtree(self.config.work_dir, ignore_errors=True)
        (self.input_dir / CONTENT_ID / "seq").mkdir(parents=True)
        for name in VIEW_NAMES:
            self._files[f"{name}/tex"] = open(
                self.input_dir / texture_input_path(name, width, height), "wb")
            self._files[f"{name}/geo"] = open(
                self.input_dir / geometry_input_path(name, width, height), "wb")
        self._started = True

    def encode_frame(
        self,
        timestamp: float,
        views: Sequence[SpatialView],
        pts: int | None = None,
        source_frame_index: int | None = None,
    ) -> None:
        """Append one spatial frame (all 9 views share `timestamp`, spec §27)."""
        if not self._started:
            raise RuntimeError("call begin() before encode_frame()")
        start = time.perf_counter()
        self._check_frame(timestamp, views)

        index = len(self._frames)
        cameras = []
        for name, view in zip(VIEW_NAMES, views):
            write_yuv420(self._files[f"{name}/tex"], *rgb_to_yuv420(view.rgb, TEXTURE_BIT_DEPTH),
                         bit_depth=TEXTURE_BIT_DEPTH)
            samples, clamped = depth_to_geometry(
                view.depth, view.mask, self.config.depth_near, self.config.depth_far,
                GEOMETRY_BIT_DEPTH,
            )
            self._clamped += clamped
            write_yuv420(self._files[f"{name}/geo"], *geometry_planes(samples, GEOMETRY_BIT_DEPTH),
                         bit_depth=GEOMETRY_BIT_DEPTH)
            cameras.append(tmiv_view_json(
                name, view.K, view.R, view.C, self.width, self.height,
                (self.config.depth_near, self.config.depth_far),
                TEXTURE_BIT_DEPTH, GEOMETRY_BIT_DEPTH,
            ))
        # Frames_number is not known yet while streaming; finalize() fills it in.
        self._write_json(self.input_dir / sequence_config_path(index),
                         sequence_config(cameras, self.fps, frame_count=0))
        self._frames.append({
            "index": index,
            "timestamp": float(timestamp),
            "pts": None if pts is None else int(pts),
            "source_frame_index": None if source_frame_index is None else int(source_frame_index),
        })
        self._convert_time += time.perf_counter() - start

    def _check_frame(self, timestamp: float, views: Sequence[SpatialView]) -> None:
        if len(views) != NUM_VIEWS:
            raise ValueError(f"expected {NUM_VIEWS} views, got {len(views)}")
        if self._frames and timestamp <= self._frames[-1]["timestamp"]:
            raise ValueError(
                f"timestamp {timestamp} is not after the previous frame's "
                f"{self._frames[-1]['timestamp']}; frames must be in presentation order"
            )
        for v, view in enumerate(views):
            if view.rgb.shape != (self.height, self.width, 3):
                raise ValueError(
                    f"view {v}: size {view.rgb.shape[1::-1]} differs from the sequence's "
                    f"{(self.width, self.height)}"
                )
        K = np.stack([np.asarray(view.K, dtype=np.float64) for view in views])
        if self._K is None:
            self._K = K
        elif not np.allclose(K, self._K, rtol=_K_TOLERANCE, atol=_K_TOLERANCE):
            raise ValueError(
                "camera intrinsics changed between frames; the MIV encoding carries per-frame "
                "poses but constant intrinsics"
            )

    def finalize(self, output_path: str | Path) -> MIVEncodeResult:
        """Run TMIV and write `output_path` (MIV) and `output_path.json` (manifest)."""
        if not self._started:
            raise RuntimeError("call begin() before finalize()")
        if not self._frames:
            raise ValueError("no frames were encoded")
        start = time.perf_counter()
        for handle in self._files.values():
            handle.close()
        frame_count = len(self._frames)

        for index in range(frame_count):
            path = self.input_dir / sequence_config_path(index)
            seq = json.loads(path.read_text())
            seq["Frames_number"] = frame_count
            self._write_json(path, seq)

        config_dir = self.config.work_dir / "configs"
        encoder_cfg = self._write_json(
            config_dir / "encoder.json",
            encoder_config(self.width, self.height, self.fps, self.config.intra_period),
        )
        mux_cfg = self._write_json(config_dir / "multiplexer.json", multiplexer_config())
        qps = config_dir / "qps.csv"
        qps.write_text(qp_csv(self.config.qp_texture, self.config.qp_geometry))

        tmiv = self.config.tmiv or find_tmiv()
        cmd = [
            sys.executable, tmiv.encode_script,
            "-i", self.input_dir, "-o", self.output_dir,
            "-s", CONTENT_ID, "-n", str(frame_count),
            "-c", encoder_cfg, "-r", "RP0", RATE_ID,
            "-v", "VVenC", "-C", tmiv.vvenc_config,
            "-m", mux_cfg, "-q", qps,
            "-t", tmiv.install_dir, "--config-dir", self.input_dir,
            "-j", str(self.config.threads),
        ]
        tmiv_start = time.perf_counter()
        self._runner(cmd, self.config.work_dir / "logs" / "encode.log")
        tmiv_time = time.perf_counter() - tmiv_start

        bitstream = self.output_dir / bitstream_output_path()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_suffix(output_path.suffix + ".tmp")
        shutil.copyfile(bitstream, tmp)
        os.replace(tmp, output_path)

        size = output_path.stat().st_size
        duration = frame_count / self.fps
        cfr = self._is_cfr()
        encode_time = self._convert_time + (time.perf_counter() - start)
        manifest_path = Path(str(output_path) + ".json")
        result = MIVEncodeResult(
            path=output_path, manifest_path=manifest_path, frame_count=frame_count,
            fps=self.fps, duration_s=duration, size_bytes=size,
            bitrate_bps=size * 8.0 / duration, encode_time_s=encode_time,
            tmiv_time_s=tmiv_time, cfr=cfr, clamped_depth_pixels=self._clamped,
        )
        self._write_json(manifest_path, self._manifest(result))
        return result

    def _is_cfr(self) -> bool:
        """Whether every timestamp sits within half a frame of `t0 + i / fps`."""
        timestamps = np.array([f["timestamp"] for f in self._frames])
        expected = timestamps[0] + np.arange(len(timestamps)) / self.fps
        return bool(np.max(np.abs(timestamps - expected)) <= 0.5 / self.fps)

    def _manifest(self, result: MIVEncodeResult) -> dict:
        return {
            "format": MIV_FORMAT,
            "content_id": CONTENT_ID,
            "frame_count": result.frame_count,
            "fps": result.fps,
            "duration_s": result.duration_s,
            "cfr": result.cfr,
            "timing_note": (
                "MIV carries a frame rate, not per-frame timestamps; frame i is presented "
                "at frames[0].timestamp + i / fps. Source timestamps are listed per frame."
            ),
            "views": list(VIEW_NAMES),
            "view_layout": "3x3, V4 = centre/original viewpoint",
            "resolution": [self.width, self.height],
            "depth_range": [self.config.depth_near, self.config.depth_far],
            "depth_unit": "meter",
            "clamped_depth_pixels": result.clamped_depth_pixels,
            "coordinate_systems": {
                "stage1": "OpenCV: x right, y down, z forward; x_cam = R @ (x_world - C)",
                "miv": "TMIV/OMAF: x forward, y left, z up; x_tmiv = P @ x_opencv",
            },
            "size_bytes": result.size_bytes,
            "bitrate_bps": result.bitrate_bps,
            "encode_time_s": result.encode_time_s,
            "qp": {"texture": self.config.qp_texture, "geometry": self.config.qp_geometry},
            "intra_period": self.config.intra_period,
            "frames": self._frames,
        }

    @staticmethod
    def _write_json(path: Path, data: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        return path


def encode_sequence(
    reader, config: MIVEncoderConfig, output_path: str | Path, runner: Runner = run_logged
) -> MIVEncodeResult:
    """Mode A second step: encode a temporal HDF5 sequence (a `SequenceReader`) to MIV."""
    encoder = MIVEncoder(config, runner=runner)
    encoder.begin(fps=reader.info.fps, width=reader.info.width, height=reader.info.height)
    for frame in reader:
        encoder.encode_frame(frame.timestamp, frame.views, pts=frame.pts,
                             source_frame_index=frame.source_frame_index)
    return encoder.finalize(output_path)
