"""Video -> Stage 1 per frame -> spatial sequence -> MIV (spec §7, §30).

Mode A ("two-step", default): all selected frames are spatialized into a
temporal HDF5 sequence first, then the MIV encoder reads that file back frame
by frame. Recommended for debugging (spec §30); the HDF5 stays on disk.

Mode B ("streaming"): each spatial frame goes straight to the MIV encoder
(and to an HDF5 dump only if `dump_hdf5` is set).

Both modes feed `MIVEncoder.encode_frame` the same frames in the same order,
so they produce the same spatial content (tested byte-for-byte).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .contract import SequenceInfo
from .miv.encoder import MIVEncodeResult, MIVEncoder, MIVEncoderConfig, encode_sequence
from .miv.tmiv import TmivInstall, find_tmiv, run_logged
from .report import build_report, peak_rss_bytes, write_report
from .runner import process_frames
from .scheduler import FrameSelection
from .sequence_io import SequenceReader, SequenceWriter
from .spatializer import CameraConfig, Stage1Spatializer
from .video_io import iter_frames, probe_video

MODES = ("two-step", "streaming")


def default_output_size(width: int, height: int) -> tuple[int, int]:
    """The source size rounded down to multiples of 8 (even 4:2:0, TMIV-friendly)."""
    return width // 8 * 8, height // 8 * 8


@dataclass
class PipelineOptions:
    input: Path
    output: Path
    mode: str = "two-step"
    dump_hdf5: Path | None = None
    work_dir: Path | None = None  # default: "<output>.work"
    selection: FrameSelection = field(default_factory=FrameSelection)
    worker_count: int = 1
    camera: CameraConfig = field(default_factory=CameraConfig)
    output_size: tuple[int, int] | None = None  # default: default_output_size(source)
    resume: bool = False
    checkpoint: Path | None = None
    device: str = "default"
    precision: str = "fp32"
    tmiv_dir: Path | None = None
    depth_near: float = 0.1
    depth_far: float = 1000.0
    qp_texture: int = 22
    qp_geometry: int = 8
    intra_period: int = 32
    threads: int = 4

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        self.input, self.output = Path(self.input), Path(self.output)


@dataclass
class PipelineResult:
    miv: MIVEncodeResult
    sequence_path: Path | None
    report: dict
    report_paths: tuple[Path, Path]


def run_video_to_miv(
    options: PipelineOptions,
    spatialize=None,
    runner=run_logged,
    tmiv: TmivInstall | None = None,
) -> PipelineResult:
    """Run the full Stage 2 pipeline. `spatialize`/`runner`/`tmiv` are injectable for tests."""
    start = time.perf_counter()
    video = probe_video(options.input)
    selection = options.selection.validate()
    output_size = options.output_size or default_output_size(video.width, video.height)
    work_dir = Path(options.work_dir or f"{options.output}.work")
    tmiv = tmiv or find_tmiv(options.tmiv_dir)  # fail before hours of Stage 1 work

    if spatialize is None:
        spatialize = Stage1Spatializer(
            work_dir / "stage1", camera=options.camera, output_size=output_size,
            checkpoint_path=options.checkpoint, device=options.device,
            precision=options.precision,
        )
    info = SequenceInfo(
        width=output_size[0], height=output_size[1], fps=video.fps,
        time_base=video.time_base, source_filename=options.input.name,
        extra={
            "horizontal_angle": options.camera.angle_deg,
            "vertical_angle": options.camera.angle_deg,
            "source_width": video.width, "source_height": video.height,
            "source_codec": video.codec,
        },
    )
    miv_config = MIVEncoderConfig(
        work_dir=work_dir / "miv", tmiv=tmiv, depth_near=options.depth_near,
        depth_far=options.depth_far, qp_texture=options.qp_texture,
        qp_geometry=options.qp_geometry, intra_period=options.intra_period,
        threads=options.threads,
    )
    outcomes = process_frames(
        iter_frames(options.input, selection), spatialize, work_dir / "stage1",
        worker_count=options.worker_count, resume=options.resume,
    )

    timings = []
    if options.mode == "two-step":
        sequence_path = Path(options.dump_hdf5 or work_dir / "sequence.h5")
        with SequenceWriter(sequence_path, info) as writer:
            for outcome in outcomes:
                writer.write_frame(outcome.frame)
                timings.append(outcome.timings)
        stage1_time = time.perf_counter() - start
        miv = encode_sequence(SequenceReader(sequence_path), miv_config, options.output,
                              runner=runner)
    else:
        sequence_path = Path(options.dump_hdf5) if options.dump_hdf5 else None
        encoder = MIVEncoder(miv_config, runner=runner)
        encoder.begin(fps=video.fps, width=output_size[0], height=output_size[1])
        writer = SequenceWriter(sequence_path, info) if sequence_path else None
        encode_time = 0.0
        try:
            for outcome in outcomes:
                frame = outcome.frame
                t0 = time.perf_counter()
                encoder.encode_frame(frame.timestamp, frame.views, pts=frame.pts,
                                     source_frame_index=frame.source_frame_index)
                encode_time += time.perf_counter() - t0
                if writer is not None:
                    writer.write_frame(frame)
                timings.append(outcome.timings)
        except BaseException:
            if writer is not None:
                writer.abort()
            raise
        if writer is not None:
            writer.close()
        stage1_time = time.perf_counter() - start - encode_time
        miv = encoder.finalize(options.output)

    total = time.perf_counter() - start
    report = build_report(
        video, output_size, asdict(selection), timings, stage1_time, miv, total,
        peak_rss_bytes(), options.mode,
    )
    report_paths = write_report(report, options.output)
    return PipelineResult(miv=miv, sequence_path=sequence_path, report=report,
                          report_paths=report_paths)
