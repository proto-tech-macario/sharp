"""Temporal HDF5 serialization of a spatial sequence (spec §15).

Layout::

    /metadata                 group attrs: width, height, fps, frame_count, ...
    /frames/NNNNNN            group attrs: timestamp, pts, source_frame_index,
                              plus the Stage 1 per-frame metadata
        /view_00 .. /view_08  rgb [H,W,3] u8, depth [H,W] f32, mask [H,W] u8,
                              K [3,3] f32, R [3,3] f32, C [3] f32

Frames are numbered in output (presentation) order. The writer streams one
frame at a time to `<path>.tmp` and renames it into place on a successful
close, so an interrupted run never leaves a truncated sequence behind. The
reader loads frames lazily. numpy + h5py only -- no SHARP/torch.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

import h5py
import numpy as np

from .contract import NUM_VIEWS, SequenceInfo, SpatialFrame, SpatialView

_FRAME_ATTRS = ("timestamp", "pts", "source_frame_index")
_NO_VALUE = -1  # HDF5 attrs cannot be None; pts/source index use -1 for "unknown"


def _frame_key(index: int) -> str:
    return f"{index:06d}"


class SequenceWriter:
    """Streams `SpatialFrame`s into a temporal HDF5 file."""

    def __init__(self, path: str | Path, info: SequenceInfo, compression: str | None = "gzip"):
        self.path = Path(path)
        self.info = info
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        self._compression = compression
        self._file = h5py.File(self._tmp_path, "w")
        self._file.create_group("metadata").attrs.update(info.to_attrs(frame_count=0))
        self._frames = self._file.create_group("frames")
        self._count = 0
        self._last_timestamp: float | None = None

    @property
    def frame_count(self) -> int:
        return self._count

    def write_frame(self, frame: SpatialFrame) -> None:
        """Append `frame`; frames must arrive in strictly increasing timestamp order."""
        try:
            self._check(frame)
            group = self._frames.create_group(_frame_key(self._count))
            group.attrs["timestamp"] = float(frame.timestamp)
            group.attrs["pts"] = _NO_VALUE if frame.pts is None else int(frame.pts)
            group.attrs["source_frame_index"] = (
                _NO_VALUE if frame.source_frame_index is None else int(frame.source_frame_index)
            )
            for key, value in frame.metadata.items():
                group.attrs[key] = value
            for v, view in enumerate(frame.views):
                view_group = group.create_group(f"view_{v:02d}")
                for name, dtype, compress in (
                    ("rgb", np.uint8, True), ("depth", np.float32, True),
                    ("mask", np.uint8, True), ("K", np.float32, False),
                    ("R", np.float32, False), ("C", np.float32, False),
                ):
                    data = np.asarray(getattr(view, name)).astype(dtype, copy=False)
                    view_group.create_dataset(
                        name, data=data, compression=self._compression if compress else None
                    )
        except BaseException:
            self.abort()
            raise
        self._count += 1
        self._last_timestamp = float(frame.timestamp)

    def _check(self, frame: SpatialFrame) -> None:
        if len(frame.views) != NUM_VIEWS:
            raise ValueError(f"expected {NUM_VIEWS} views, got {len(frame.views)}")
        if self._last_timestamp is not None and frame.timestamp <= self._last_timestamp:
            raise ValueError(
                f"frame {self._count}: timestamp {frame.timestamp} is not after the previous "
                f"frame's {self._last_timestamp}; frames must be in presentation order"
            )
        for v, view in enumerate(frame.views):
            if view.rgb.shape != (self.info.height, self.info.width, 3):
                raise ValueError(
                    f"frame {self._count} view {v}: resolution {view.rgb.shape[1::-1]} differs "
                    f"from the sequence resolution {(self.info.width, self.info.height)}"
                )
            if view.depth.shape != view.rgb.shape[:2] or view.mask.shape != view.rgb.shape[:2]:
                raise ValueError(f"frame {self._count} view {v}: depth/mask not aligned with rgb")

    def close(self) -> None:
        """Finalize frame_count and atomically move the file into place."""
        if self._file is None:
            return
        self._file["metadata"].attrs["frame_count"] = self._count
        self._file.close()
        self._file = None
        os.replace(self._tmp_path, self.path)

    def abort(self) -> None:
        """Discard everything written so far."""
        if self._file is not None:
            self._file.close()
            self._file = None
        self._tmp_path.unlink(missing_ok=True)

    def __enter__(self) -> SequenceWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


class SequenceReader:
    """Lazy reader for a temporal HDF5 spatial sequence."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with h5py.File(self.path, "r") as f:
            attrs = dict(f["metadata"].attrs)
            self._keys = sorted(f["frames"].keys())
        self.metadata = attrs
        self.info = SequenceInfo.from_attrs(attrs)

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Iterator[SpatialFrame]:
        for index in range(len(self)):
            yield self.frame(index)

    def frame(self, index: int) -> SpatialFrame:
        """Load frame `index` (output order)."""
        with h5py.File(self.path, "r") as f:
            group = f["frames"][self._keys[index]]
            attrs = dict(group.attrs)
            views = [
                SpatialView(**{name: group[f"view_{v:02d}/{name}"][()]
                               for name in ("rgb", "depth", "mask", "K", "R", "C")})
                for v in range(NUM_VIEWS)
            ]
        pts = int(attrs.pop("pts"))
        source_index = int(attrs.pop("source_frame_index"))
        return SpatialFrame(
            timestamp=float(attrs.pop("timestamp")),
            views=views,
            pts=None if pts == _NO_VALUE else pts,
            source_frame_index=None if source_index == _NO_VALUE else source_index,
            metadata=attrs,
        )

    def timestamps(self) -> np.ndarray:
        """All frame timestamps (seconds), without loading any pixel data."""
        with h5py.File(self.path, "r") as f:
            return np.array([f["frames"][key].attrs["timestamp"] for key in self._keys])

    def frame_attrs(self) -> list[dict]:
        """Timestamp, PTS and source frame index of every frame, without pixel data."""
        rows = []
        with h5py.File(self.path, "r") as f:
            for key in self._keys:
                attrs = f["frames"][key].attrs
                pts, source = int(attrs["pts"]), int(attrs["source_frame_index"])
                rows.append({
                    "timestamp": float(attrs["timestamp"]),
                    "pts": None if pts == _NO_VALUE else pts,
                    "source_frame_index": None if source == _NO_VALUE else source,
                })
        return rows
