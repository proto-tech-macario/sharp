"""Command-line entry points for Stage 2 (spec §36, §37.5).

- `sharp_video_to_miv`    video -> MIV (the application)
- `sharp-video-validate`  validate a temporal HDF5 sequence + temporal metrics
- `sharp-video-inspect`   contact sheets, depth maps, camera and timestamp tables
- `sharp-miv-encode`      standalone MIV encoder: temporal HDF5 -> MIV
- `sharp-miv-validate`    decode an MIV file with TmivDecoder and check it
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import click

from .pipeline import MODES, PipelineOptions, run_video_to_miv
from .runner import FrameFailuresError
from .scheduler import FrameSelection
from .spatializer import CameraConfig, load_camera_config

_EXISTING_FILE = click.Path(exists=True, dir_okay=False, path_type=Path)
_OUTPUT_FILE = click.Path(dir_okay=False, path_type=Path)


def _resolution(ctx, param, value):
    if value is None:
        return None
    match = re.fullmatch(r"(\d+)x(\d+)", value)
    if not match:
        raise click.BadParameter("expected WIDTHxHEIGHT, e.g. 1280x720")
    width, height = int(match[1]), int(match[2])
    if width % 8 or height % 8 or width == 0 or height == 0:
        raise click.BadParameter(
            "width and height must each be a positive multiple of 8 (4:2:0 views, "
            "TMIV block alignment)"
        )
    return width, height


def _miv_options(func):
    """Options shared by every command that runs the MIV encoder."""
    for option in reversed([
        click.option("--tmiv-dir", type=click.Path(file_okay=False, path_type=Path),
                     default=None, help="TMIV build (default: $SHARP_TMIV_DIR, then ./.tmiv)."),
        click.option("--depth-near", type=float, default=0.1, show_default=True,
                     help="Nearest representable depth in metres."),
        click.option("--depth-far", type=float, default=1000.0, show_default=True,
                     help="Farthest representable depth in metres."),
        click.option("--qp-texture", type=int, default=22, show_default=True,
                     help="VVC QP for the texture atlases."),
        click.option("--qp-geometry", type=int, default=8, show_default=True,
                     help="VVC QP for the geometry (depth) atlases."),
        click.option("--intra-period", type=click.Choice(["16", "32"]), default="32",
                     show_default=True, help="Intra period / VVC GOP size."),
        click.option("--threads", type=int, default=4, show_default=True,
                     help="Threads for TMIV and VVenC."),
    ]):
        func = option(func)
    return func


def _fail(message: str, code: int = 1) -> None:
    click.echo(f"ERROR: {message}", err=True)
    sys.exit(code)


@click.command()
@click.option("-i", "--input", "input_path", type=_EXISTING_FILE, required=True,
              help="Input video (MP4 with H.264/H.265).")
@click.option("-o", "--output", "output_path", type=_OUTPUT_FILE, required=True,
              help="Output MIV file.")
@click.option("--dump-hdf5", type=_OUTPUT_FILE, default=None,
              help="Also keep the temporal HDF5 spatial sequence here.")
@click.option("--mode", type=click.Choice(MODES), default="two-step", show_default=True,
              help="two-step: video -> HDF5 -> MIV (debug-friendly); streaming: frames go "
              "straight to the encoder.")
@click.option("--start-frame", type=int, default=0, show_default=True,
              help="First frame (presentation order, inclusive).")
@click.option("--end-frame", type=int, default=None, help="Stop before this frame.")
@click.option("--max-frames", type=int, default=None, help="Process at most this many frames.")
@click.option("--worker-count", type=int, default=1, show_default=True,
              help="Frames spatialized concurrently (bounded by GPU memory).")
@click.option("--camera-config", type=_EXISTING_FILE, default=None,
              help='JSON camera configuration, e.g. {"angle_deg": 10}.')
@click.option("--output-resolution", callback=_resolution, default=None,
              help="View size WIDTHxHEIGHT (default: source rounded down to multiples of 8).")
@click.option("--work-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Intermediate files, frame cache and journal (default: <output>.work).")
@click.option("--resume", is_flag=True, help="Reuse frames already completed in --work-dir.")
@click.option("-c", "--checkpoint", type=_EXISTING_FILE, default=None,
              help="SHARP checkpoint (downloads the default model if omitted).")
@click.option("--device", default="default", show_default=True,
              help="SHARP inference device; rendering always needs CUDA.")
@click.option("--precision", type=click.Choice(["fp32", "fp16", "bf16"]), default="fp16",
              show_default=True, help="Autocast dtype for the SHARP forward pass.")
@click.option("--validate", "run_validation", is_flag=True,
              help="Afterwards, decode the MIV file with TmivDecoder and check it.")
@click.option("--no-disk-check", is_flag=True,
              help="Start even if the run looks like it would fill the disk.")
@_miv_options
def video_to_miv_cli(input_path, output_path, dump_hdf5, mode, start_frame, end_frame,
                     max_frames, worker_count, camera_config, output_resolution, work_dir,
                     resume, checkpoint, device, precision, run_validation, no_disk_check, tmiv_dir,
                     depth_near, depth_far, qp_texture, qp_geometry, intra_period, threads):
    """Convert a video into a 9-view spatial MIV file (Stage 2)."""
    from .diskspace import InsufficientDiskSpaceError
    from .miv.tmiv import TmivError, TmivNotFoundError
    from .video_io import UnsupportedVideoError

    try:
        options = PipelineOptions(
            input=input_path, output=output_path, mode=mode, dump_hdf5=dump_hdf5,
            work_dir=work_dir,
            selection=FrameSelection(start_frame, end_frame, max_frames).validate(),
            worker_count=worker_count,
            camera=load_camera_config(camera_config) if camera_config else CameraConfig(),
            output_size=output_resolution, resume=resume, checkpoint=checkpoint,
            device=device, precision=precision, tmiv_dir=tmiv_dir, depth_near=depth_near,
            depth_far=depth_far, qp_texture=qp_texture, qp_geometry=qp_geometry,
            intra_period=int(intra_period), threads=threads, check_disk=not no_disk_check,
        )
    except ValueError as exc:
        _fail(str(exc), code=2)

    try:
        result = run_video_to_miv(options)
    except InsufficientDiskSpaceError as exc:
        _fail(f"{exc} (--no-disk-check starts anyway.)")
    except UnsupportedVideoError as exc:
        _fail(str(exc), code=2)
    except FrameFailuresError as exc:
        work = options.work_dir or f"{output_path}.work"
        _fail(f"{exc}\nNo MIV file was written. Fix the cause and re-run with --resume "
              f"(journal: {work}/stage1/journal.json).")
    except (TmivNotFoundError, TmivError) as exc:
        _fail(str(exc))

    miv = result.miv
    click.echo(f"Wrote {miv.path}: {miv.frame_count} frames x 9 views, "
               f"{miv.size_bytes} bytes, {miv.bitrate_bps / 1000:.1f} kbps.")
    if not miv.cfr:
        click.echo("WARNING: variable frame rate input; per-frame timestamps are kept in "
                   f"{miv.manifest_path}.")
    if result.sequence_path:
        click.echo(f"Spatial sequence: {result.sequence_path}")
    click.echo(f"Report: {result.report_paths[1]}")

    if run_validation:
        if result.sequence_path is None:
            _fail("--validate needs the spatial sequence: use --mode two-step or --dump-hdf5")
        _print_miv_validation(miv.path, result.sequence_path,
                              Path(f"{output_path}.work") / "validate", tmiv_dir)


def _print_miv_validation(miv_path, reference_path, work_dir, tmiv_dir, report_path=None):
    from .miv.tmiv import find_tmiv
    from .miv.validate import validate_miv
    from .sequence_io import SequenceReader

    report = validate_miv(miv_path, SequenceReader(reference_path), work_dir,
                          tmiv=find_tmiv(tmiv_dir))
    if report_path:
        Path(report_path).write_text(json.dumps(
            {"ok": report.ok, "failures": report.failures, "stats": report.stats},
            indent=2, default=str))
    stats = report.stats
    click.echo(f"MIV decoded by TmivDecoder: {stats['frame_count']} frames, "
               f"{len(stats['views'])} views, luma PSNR min {stats['y_psnr_db_min']}, "
               f"depth median rel. error max {stats['depth_median_rel_error_max']}, "
               f"mask agreement min {stats['mask_agreement_min']}.")
    if report.ok:
        click.echo(f"PASS: {miv_path} matches {reference_path}.")
        return
    click.echo(f"FAIL: {miv_path} does not match {reference_path}:", err=True)
    for failure in report.failures:
        click.echo(f"  - {failure}", err=True)
    sys.exit(1)


@click.command()
@click.argument("sequence_path", type=_EXISTING_FILE)
@click.option("--no-frames", is_flag=True,
              help="Skip the per-frame Stage 1 checks (sequence structure only).")
@click.option("--temporal-json", type=_OUTPUT_FILE, default=None,
              help="Write the temporal metrics (flicker, instability) to this JSON file.")
def validate_sequence_cli(sequence_path, no_frames, temporal_json):
    """Validate a temporal HDF5 spatial sequence and measure temporal behaviour."""
    from .metrics import temporal_metrics
    from .sequence_io import SequenceReader
    from .validation import validate_sequence

    report = validate_sequence(sequence_path, check_frames=not no_frames)
    metrics = temporal_metrics(SequenceReader(sequence_path)) if report.frame_count else None
    click.echo(f"{report.frame_count} frames at {report.fps:.3f} fps "
               f"({'constant' if report.cfr else 'variable'} frame rate).")
    if metrics is not None:
        for key, stats in metrics.summary.items():
            click.echo(f"  {key}: mean {stats['mean']:.4g}, p95 {stats['p95']:.4g}, "
                       f"max {stats['max']:.4g}")
        if temporal_json:
            temporal_json.write_text(json.dumps(
                {"summary": metrics.summary, "per_pair": metrics.per_pair}, indent=2))
    if report.ok:
        click.echo(f"PASS: {sequence_path} is a valid spatial sequence.")
        return
    click.echo(f"FAIL: {sequence_path} failed validation:", err=True)
    for failure in report.failures:
        click.echo(f"  - {failure}", err=True)
    sys.exit(1)


@click.command()
@click.argument("sequence_path", type=_EXISTING_FILE)
@click.option("--frame", "frames", type=int, multiple=True,
              help="Frame to inspect (repeatable; default: 0).")
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path),
              required=True, help="Directory for the images and tables.")
def inspect_cli(sequence_path, frames, out_dir):
    """Write RGB/depth contact sheets, camera tables and the timestamp table."""
    from .inspect import timestamp_table, write_inspection
    from .sequence_io import SequenceReader

    reader = SequenceReader(sequence_path)
    for index in frames or (0,):
        if not 0 <= index < len(reader):
            _fail(f"frame {index} out of range (0..{len(reader) - 1})", code=2)
        for path in write_inspection(reader, index, out_dir):
            click.echo(f"wrote {path}")
    table = timestamp_table(reader)
    (Path(out_dir) / "timestamps.json").write_text(json.dumps(table, indent=2))
    deltas = [row["delta_s"] for row in table if row["delta_s"] is not None]
    if deltas:
        click.echo(f"{len(table)} frames, spacing {min(deltas) * 1000:.3f}.."
                   f"{max(deltas) * 1000:.3f} ms")


@click.command()
@click.argument("sequence_path", type=_EXISTING_FILE)
@click.option("-o", "--output", "output_path", type=_OUTPUT_FILE, required=True,
              help="Output MIV file.")
@click.option("--work-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="TMIV working files (default: <output>.work/miv).")
@_miv_options
def miv_encode_cli(sequence_path, output_path, work_dir, tmiv_dir, depth_near, depth_far,
                   qp_texture, qp_geometry, intra_period, threads):
    """Encode a temporal HDF5 spatial sequence as MIV (the standalone MIV encoder)."""
    from .miv.encoder import MIVEncoderConfig, encode_sequence
    from .miv.tmiv import TmivError, TmivNotFoundError, find_tmiv
    from .sequence_io import SequenceReader

    try:
        config = MIVEncoderConfig(
            work_dir=work_dir or Path(f"{output_path}.work") / "miv", tmiv=find_tmiv(tmiv_dir),
            depth_near=depth_near, depth_far=depth_far, qp_texture=qp_texture,
            qp_geometry=qp_geometry, intra_period=int(intra_period), threads=threads,
        )
        result = encode_sequence(SequenceReader(sequence_path), config, output_path)
    except (TmivNotFoundError, TmivError, ValueError) as exc:
        _fail(str(exc))
    click.echo(f"Wrote {result.path}: {result.frame_count} frames, {result.size_bytes} bytes, "
               f"{result.bitrate_bps / 1000:.1f} kbps, {result.encode_time_s:.1f} s.")


@click.command()
@click.argument("miv_path", type=_EXISTING_FILE)
@click.option("--reference", type=_EXISTING_FILE, required=True,
              help="The temporal HDF5 sequence the MIV file was encoded from.")
@click.option("--work-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Decoder output (default: <miv>.validate).")
@click.option("--tmiv-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="TMIV build (default: $SHARP_TMIV_DIR, then ./.tmiv).")
@click.option("--report", "report_path", type=_OUTPUT_FILE, default=None,
              help="Write the validation report as JSON.")
def miv_validate_cli(miv_path, reference, work_dir, tmiv_dir, report_path):
    """Decode an MIV file with the reference decoder and check it against its source."""
    from .miv.tmiv import TmivError, TmivNotFoundError

    try:
        _print_miv_validation(miv_path, reference, work_dir or Path(f"{miv_path}.validate"),
                              tmiv_dir, report_path)
    except (TmivNotFoundError, TmivError) as exc:
        _fail(str(exc))
