"""Validation of a temporal spatial sequence (spec §33.1, §37.5).

Every frame goes through the unchanged Stage 1 validator (shapes, metadata,
camera rotation validity, depth sanity, V3->V5 reprojection); on top of that
the sequence-level contract is checked: frame count, strictly increasing
timestamps, constant intrinsics. Frames are loaded one at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .sequence_io import SequenceReader

REQUIRED_METADATA = (
    "width", "height", "fps", "frame_count", "view_count", "depth_unit",
    "depth_definition", "coordinate_system", "camera_convention",
)
_K_TOLERANCE = 1e-6


@dataclass
class SequenceReport:
    ok: bool
    failures: list[str] = field(default_factory=list)
    frame_count: int = 0
    fps: float = 0.0
    cfr: bool = True  # informational: constant frame rate within half a frame


def is_cfr(timestamps: np.ndarray, fps: float) -> bool:
    """Whether every timestamp sits within half a frame of `t0 + i / fps`."""
    if len(timestamps) < 2 or fps <= 0:
        return True
    expected = timestamps[0] + np.arange(len(timestamps)) / fps
    return bool(np.max(np.abs(timestamps - expected)) <= 0.5 / fps)


def validate_sequence(path: str | Path, check_frames: bool = True) -> SequenceReport:
    """Validate a temporal HDF5 sequence; collects every failure instead of raising."""
    try:
        reader = SequenceReader(path)
    except Exception as exc:
        return SequenceReport(ok=False, failures=[f"cannot read {path}: {exc}"])

    failures: list[str] = []
    meta = reader.metadata
    failures += [f"missing metadata key: {key}" for key in REQUIRED_METADATA if key not in meta]

    count = len(reader)
    if meta.get("frame_count") != count:
        failures.append(
            f"metadata frame_count={meta.get('frame_count')} but the file holds {count} frames"
        )
    if meta.get("view_count", 9) != 9:
        failures.append(f"metadata view_count={meta.get('view_count')}, expected 9")

    timestamps = reader.timestamps()
    for i in range(1, count):
        if timestamps[i] <= timestamps[i - 1]:
            failures.append(
                f"timestamp of frame {i} ({timestamps[i]}) is not after frame {i - 1} "
                f"({timestamps[i - 1]})"
            )

    if check_frames:
        from sharp_spatialize.validation import ValidationError, validate_in_memory

    first_K = None
    for i, frame in enumerate(reader):
        if check_frames:
            try:
                validate_in_memory(frame.to_stage1())
            except ValidationError as exc:
                failures += [f"frame {i}: {part}" for part in str(exc).split("; ")]
        K = frame.stacked()["K"]
        if first_K is None:
            first_K = K
        elif not np.allclose(K, first_K, rtol=_K_TOLERANCE, atol=_K_TOLERANCE):
            failures.append(f"frame {i}: camera intrinsics differ from frame 0")

    return SequenceReport(
        ok=not failures, failures=failures, frame_count=count, fps=reader.info.fps,
        cfr=is_cfr(timestamps, reader.info.fps),
    )
