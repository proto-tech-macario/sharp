#!/usr/bin/env python3
"""Run the Stage 2 test matrix (spec §34) over a directory of clips.

Name each clip after its category, e.g. ``static_office.mp4``,
``camera_motion_street.mp4``, ``moving_object_dog.mp4``,
``camera_object_skate.mp4``, ``difficult_foliage.mp4``. For every clip this runs

    sharp_video_to_miv --dump-hdf5 --validate    (Stage 1 needs the CUDA machine)
    sharp-video-validate --temporal-json

and writes ``matrix_report.md`` / ``matrix_report.json`` in the output
directory. The MIV files it produces are the representative MIV test files of
spec §37.4. Any extra arguments are passed to ``sharp_video_to_miv``::

    python scripts/stage2/run_test_matrix.py clips/ out/ -- --max-frames 120 \
        --output-resolution 1280x720
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

CATEGORIES = {
    "static": "Test 1 - static scene",
    "camera_motion": "Test 2 - camera movement",
    "moving_object": "Test 3 - moving object",
    "camera_object": "Test 4 - camera + object movement",
    "difficult": "Test 5 - difficult geometry",
}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".m4v"}


def categorize(name: str) -> str:
    """The spec §34 category key a clip file name starts with, or "uncategorized"."""
    stem = Path(name).stem.lower()
    for key in sorted(CATEGORIES, key=len, reverse=True):  # "camera_object" before "camera"
        if stem.startswith(key):
            return key
    return "uncategorized"


def _cli(module_function: str) -> list[str]:
    return [sys.executable, "-c", f"import sharp_video.cli as c; c.{module_function}()"]


def run_clip(clip: Path, out_dir: Path, extra: list[str]) -> dict:
    """Run the pipeline and the sequence validator on one clip; returns its matrix row."""
    stem = clip.stem
    miv, sequence = out_dir / f"{stem}.miv", out_dir / f"{stem}.h5"
    temporal = out_dir / f"{stem}.temporal.json"
    log = out_dir / f"{stem}.log"
    start = time.perf_counter()
    with open(log, "w") as handle:
        encode = subprocess.run(
            _cli("video_to_miv_cli") + ["-i", str(clip), "-o", str(miv),
                                         "--dump-hdf5", str(sequence), "--validate", *extra],
            stdout=handle, stderr=subprocess.STDOUT,
        )
        validate = None
        if sequence.exists():
            validate = subprocess.run(
                _cli("validate_sequence_cli") + [str(sequence), "--temporal-json", str(temporal)],
                stdout=handle, stderr=subprocess.STDOUT,
            )
    row = {
        "clip": clip.name,
        "category": categorize(clip.name),
        "pipeline_ok": encode.returncode == 0,
        "sequence_ok": validate is not None and validate.returncode == 0,
        "wall_time_s": time.perf_counter() - start,
        "log": str(log),
    }
    report_path = Path(f"{miv}.report.json")
    if report_path.exists():
        report = json.loads(report_path.read_text())
        row.update({
            "frames": report["stage1"]["frames_processed"],
            "miv_bytes": report["miv"]["file_size_bytes"],
            "bitrate_kbps": report["miv"]["average_bitrate_bps"] / 1000.0,
            "processing_fps": report["overall"]["average_processing_fps"],
        })
    if temporal.exists():
        summary = json.loads(temporal.read_text())["summary"]
        row.update({
            "rgb_flicker": summary["rgb_flicker"]["mean"],
            "depth_flicker": summary["depth_flicker"]["mean"],
            "camera_translation_m": summary["camera_translation"]["mean"],
        })
    return row


def _cell(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summarize(rows: list[dict]) -> str:
    """The matrix as Markdown, plus which spec §34 categories have no clip."""
    columns = [("clip", "Clip"), ("category", "Category"), ("pipeline_ok", "Pipeline + MIV"),
               ("sequence_ok", "Sequence"), ("frames", "Frames"), ("miv_bytes", "MIV bytes"),
               ("bitrate_kbps", "kbps"), ("processing_fps", "Proc. FPS"),
               ("rgb_flicker", "RGB flicker"), ("depth_flicker", "Depth flicker"),
               ("camera_translation_m", "Cam. motion (m)")]
    lines = ["# Stage 2 test matrix", "",
             "| " + " | ".join(title for _, title in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row.get(key)) for key, _ in columns) + " |")
    missing = [CATEGORIES[key] for key in CATEGORIES if key not in {r["category"] for r in rows}]
    lines.append("")
    if missing:
        lines.append("Categories without a clip: " + "; ".join(missing) + ".")
    else:
        lines.append("All five spec §34 categories are covered.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clips", type=Path, help="directory of category-named clips")
    parser.add_argument("out", type=Path, help="output directory")
    parser.add_argument("extra", nargs=argparse.REMAINDER,
                        help="arguments after -- go to sharp_video_to_miv")
    args = parser.parse_args(argv)
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra

    clips = sorted(p for p in args.clips.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    if not clips:
        print(f"no video clips in {args.clips}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for clip in clips:
        print(f"==> {clip.name} ({categorize(clip.name)})", flush=True)
        rows.append(run_clip(clip, args.out, extra))
    (args.out / "matrix_report.json").write_text(json.dumps(rows, indent=2))
    (args.out / "matrix_report.md").write_text(summarize(rows))
    print((args.out / "matrix_report.md").read_text())
    return 0 if all(r["pipeline_ok"] and r["sequence_ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
