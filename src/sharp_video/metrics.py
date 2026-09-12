"""Temporal quality measurements of a spatial sequence (spec §33.2).

Stage 2 measures temporal behaviour; it does not correct it. Frames are
spatialized independently, so these numbers show how much RGB, depth and the
virtual cameras change from one frame to the next. Two frames are held in
memory at a time.

Per consecutive frame pair and view:
- rgb_flicker: mean |RGB(t) - RGB(t-1)| (0-255) over pixels valid in both.
- depth_flicker: median |Z(t) - Z(t-1)| / Z(t-1) over pixels valid in both.
- camera_translation: |C(t) - C(t-1)| in metres (view instability).
- camera_rotation_deg: rotation angle between R(t-1) and R(t).
Per frame pair:
- center_depth_change: |median depth of V4 at t - at t-1| in metres
  (frame-to-frame geometry change of the reconstructed scene).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

VIEW_KEYS = ("rgb_flicker", "depth_flicker", "camera_translation", "camera_rotation_deg")
CENTER_VIEW = 4


@dataclass
class TemporalMetrics:
    per_pair: list[dict] = field(default_factory=list)  # one row per frame pair
    summary: dict[str, dict[str, float]] = field(default_factory=dict)


def _rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    # ||R - I||_F = 2*sqrt(2)*sin(theta/2); unlike arccos((trace-1)/2) this stays
    # accurate for the tiny angles between consecutive frames.
    relative = Rb.astype(np.float64) @ Ra.astype(np.float64).T
    chord = np.linalg.norm(relative - np.eye(3)) / (2.0 * np.sqrt(2.0))
    return float(np.degrees(2.0 * np.arcsin(np.clip(chord, 0.0, 1.0))))


def _view_pair(a, b) -> dict[str, float]:
    both = (a.mask == 1) & (b.mask == 1)
    if both.any():
        rgb = float(np.mean(np.abs(b.rgb[both].astype(np.float32) - a.rgb[both])))
        depth = float(np.median(np.abs(b.depth[both] - a.depth[both]) / a.depth[both]))
    else:
        rgb = depth = 0.0
    return {
        "rgb_flicker": rgb,
        "depth_flicker": depth,
        "camera_translation": float(np.linalg.norm(b.C.astype(np.float64) - a.C)),
        "camera_rotation_deg": _rotation_angle_deg(a.R, b.R),
    }


def _median_valid_depth(view) -> float:
    valid = view.mask == 1
    return float(np.median(view.depth[valid])) if valid.any() else 0.0


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "max": 0.0, "p95": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "max": float(arr.max()),
            "p95": float(np.percentile(arr, 95))}


def temporal_metrics(frames) -> TemporalMetrics:
    """Measure frame-to-frame change over an iterable of SpatialFrames (e.g. a SequenceReader)."""
    metrics = TemporalMetrics()
    previous = None
    for t, frame in enumerate(frames):
        if previous is not None:
            views = [_view_pair(a, b) for a, b in zip(previous.views, frame.views)]
            metrics.per_pair.append({
                "from_frame": t - 1,
                "to_frame": t,
                "views": views,
                "center_depth_change": abs(
                    _median_valid_depth(frame.views[CENTER_VIEW])
                    - _median_valid_depth(previous.views[CENTER_VIEW])
                ),
            })
        previous = frame

    for key in VIEW_KEYS:
        metrics.summary[key] = _stats([v[key] for row in metrics.per_pair for v in row["views"]])
    metrics.summary["center_depth_change"] = _stats(
        [row["center_depth_change"] for row in metrics.per_pair]
    )
    return metrics
