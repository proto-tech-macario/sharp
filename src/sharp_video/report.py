"""Performance and output-size report of a video-to-MIV run (spec §35, §37.6).

Written next to the MIV file as `<output>.report.json` and `<output>.report.md`.
Stage 2 measures performance; it does not optimize for real time.
"""

from __future__ import annotations

import json
import resource
import sys
from pathlib import Path

import numpy as np

from .miv.encoder import MIVEncodeResult
from .spatializer import FrameTimings
from .video_io import VideoInfo

SHARP_3DGS_NOTE = (
    "SHARP regresses the 3D Gaussians directly and Stage 1 unprojects them in the same call, "
    "so SHARP inference and 3DGS generation are timed together."
)


def peak_rss_bytes() -> int:
    """Peak resident memory of this process (ru_maxrss is bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _per_frame(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}


def build_report(
    video: VideoInfo,
    output_size: tuple[int, int],
    selection: dict,
    frame_timings: list[FrameTimings | None],
    stage1_time_s: float,
    miv: MIVEncodeResult,
    total_time_s: float,
    peak_memory_bytes: int,
    mode: str,
) -> dict:
    """Assemble the report dictionary."""
    measured = [t for t in frame_timings if t is not None]
    gpu = [t.peak_gpu_bytes for t in measured if t.peak_gpu_bytes is not None]
    frames = len(frame_timings)
    return {
        "mode": mode,
        "input": {
            "path": video.path,
            "resolution": [video.width, video.height],
            "fps": video.fps,
            "duration_s": video.duration,
            "frame_count": video.frame_count,
            "codec": video.codec,
            "pix_fmt": video.pix_fmt,
            "colour": {"primaries": video.color_primaries, "transfer": video.color_trc,
                       "matrix": video.colorspace, "range": video.color_range},
        },
        "selection": selection,
        "output_resolution": list(output_size),
        "stage1": {
            "frames_processed": frames,
            "frames_from_resume_cache": frames - len(measured),
            "sharp_and_3dgs_s_per_frame": _per_frame([t.sharp_and_3dgs_s for t in measured]),
            "sharp_and_3dgs_note": SHARP_3DGS_NOTE,
            "nine_view_render_s_per_frame": _per_frame([t.render_s for t in measured]),
            "validation_s_per_frame": _per_frame([t.validate_s for t in measured]),
            "stage1_s_per_frame": _per_frame([t.total_s for t in measured]),
            "total_stage1_s": stage1_time_s,
            "peak_gpu_memory_bytes": max(gpu) if gpu else None,
        },
        "miv": {
            "path": str(miv.path),
            "encode_time_s": miv.encode_time_s,
            "tmiv_time_s": miv.tmiv_time_s,
            "file_size_bytes": miv.size_bytes,
            "average_bitrate_bps": miv.bitrate_bps,
            "frame_count": miv.frame_count,
            "fps": miv.fps,
            "duration_s": miv.duration_s,
            "constant_frame_rate": miv.cfr,
            "clamped_depth_pixels": miv.clamped_depth_pixels,
        },
        "overall": {
            "total_processing_time_s": total_time_s,
            "average_processing_fps": frames / total_time_s if total_time_s > 0 else None,
            "miv_file_size_bytes": miv.size_bytes,
            "average_miv_bitrate_bps": miv.bitrate_bps,
            "peak_memory_bytes": peak_memory_bytes,
            "peak_gpu_memory_bytes": max(gpu) if gpu else None,
        },
    }


def _fmt(value, unit: str = "", digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}{unit}"
    return f"{value}{unit}"


def _mib(value) -> str:
    return "n/a" if value is None else f"{value / 2**20:.1f} MiB"


def to_markdown(report: dict) -> str:
    """A human-readable rendering of `build_report`'s output."""
    inp, s1, miv, overall = report["input"], report["stage1"], report["miv"], report["overall"]
    rows = [
        ("Input resolution", f"{inp['resolution'][0]}x{inp['resolution'][1]}"),
        ("Input frame rate", _fmt(inp["fps"], " fps")),
        ("Input duration", _fmt(inp["duration_s"], " s")),
        ("Input frame count", _fmt(inp["frame_count"])),
        ("Frames processed", _fmt(s1["frames_processed"])),
        ("Output view resolution", "x".join(str(v) for v in report["output_resolution"])),
        ("SHARP + 3DGS time / frame (mean)", _fmt(s1["sharp_and_3dgs_s_per_frame"]["mean"], " s")),
        ("9-view generation time / frame (mean)",
         _fmt(s1["nine_view_render_s_per_frame"]["mean"], " s")),
        ("Total Stage 1 processing time", _fmt(s1["total_stage1_s"], " s")),
        ("Peak GPU memory", _mib(s1["peak_gpu_memory_bytes"])),
        ("MIV encoding time", _fmt(miv["encode_time_s"], " s")),
        ("MIV file size", f"{miv['file_size_bytes']} B"),
        ("Average MIV bitrate", _fmt(miv["average_bitrate_bps"] / 1000.0, " kbps", 1)),
        ("Total processing time", _fmt(overall["total_processing_time_s"], " s")),
        ("Average processing FPS", _fmt(overall["average_processing_fps"])),
        ("Peak memory (RSS)", _mib(overall["peak_memory_bytes"])),
    ]
    lines = ["# Stage 2 video-to-MIV performance report", "",
             f"Mode: {report['mode']}. Source: `{inp['path']}`.", "",
             "| Measure | Value |", "| --- | --- |"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    lines += ["", f"_{s1['sharp_and_3dgs_note']}_", ""]
    if not miv["constant_frame_rate"]:
        lines += ["**Warning:** the input is not constant-frame-rate; MIV carries a frame rate, "
                  "so per-frame timestamps are preserved only in the `.miv.json` manifest.", ""]
    return "\n".join(lines)


def write_report(report: dict, output_path: str | Path) -> tuple[Path, Path]:
    """Write `<output>.report.json` and `<output>.report.md`; returns both paths."""
    base = str(output_path)
    json_path, md_path = Path(base + ".report.json"), Path(base + ".report.md")
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(to_markdown(report))
    return json_path, md_path
