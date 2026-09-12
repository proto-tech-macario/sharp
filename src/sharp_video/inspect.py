"""Inspection helpers for a temporal spatial sequence (spec §37.5).

- RGB contact sheet: the 9 views of a frame in their 3x3 layout.
- Depth contact sheet: inverse depth, colour-mapped per frame; invalid pixels black.
- Camera table: K, R, C of every view of a frame (JSON).
- Timestamp table: timestamp, PTS, source index and spacing of every frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

from .contract import SpatialFrame


def _grid(images: list[np.ndarray]) -> np.ndarray:
    rows = [np.concatenate(images[r * 3:(r + 1) * 3], axis=1) for r in range(3)]
    return np.concatenate(rows, axis=0)


def depth_sheet(frame: SpatialFrame) -> np.ndarray:
    """[3H, 3W, 3] uint8 colour-mapped inverse depth for the 9 views of `frame`."""
    valid_all = np.stack([view.mask == 1 for view in frame.views])
    inverse = np.stack([np.where(view.mask == 1, 1.0 / np.maximum(view.depth, 1e-6), 0.0)
                        for view in frame.views])
    if valid_all.any():
        lo, hi = inverse[valid_all].min(), inverse[valid_all].max()
    else:
        lo, hi = 0.0, 1.0
    norm = np.clip((inverse - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    colours = (matplotlib.colormaps["turbo"](norm)[..., :3] * 255).astype(np.uint8)
    colours[~valid_all] = 0
    return _grid(list(colours))


def camera_table(frame: SpatialFrame) -> dict:
    """K, R and C of every view of `frame`, as a JSON-ready dict."""
    return {
        "timestamp": frame.timestamp,
        "pts": frame.pts,
        "source_frame_index": frame.source_frame_index,
        "views": [
            {"view": v, "K": view.K.tolist(), "R": view.R.tolist(), "C": view.C.tolist()}
            for v, view in enumerate(frame.views)
        ],
    }


def write_inspection(reader, frame_index: int, out_dir: str | Path) -> list[Path]:
    """Write the RGB sheet, depth sheet and camera table of one frame; returns the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = reader.frame(frame_index)
    stem = f"frame_{frame_index:06d}"
    rgb_path = out_dir / f"{stem}_rgb.png"
    depth_path = out_dir / f"{stem}_depth.png"
    camera_path = out_dir / f"{stem}_cameras.json"
    Image.fromarray(_grid([view.rgb for view in frame.views])).save(rgb_path)
    Image.fromarray(depth_sheet(frame)).save(depth_path)
    camera_path.write_text(json.dumps(camera_table(frame), indent=2))
    return [rgb_path, depth_path, camera_path]


def timestamp_table(reader) -> list[dict]:
    """One row per frame: output index, timestamp, PTS, source index, spacing."""
    rows = []
    previous = None
    for i, attrs in enumerate(reader.frame_attrs()):
        rows.append({
            "frame": i,
            "timestamp": attrs["timestamp"],
            "pts": attrs["pts"],
            "source_frame_index": attrs["source_frame_index"],
            "delta_s": None if previous is None else attrs["timestamp"] - previous,
        })
        previous = attrs["timestamp"]
    return rows
