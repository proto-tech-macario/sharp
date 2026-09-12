"""Runs Stage 1 over the selected frames: workers, cache, resume, ordering (spec §29-§32).

- Frames are independent, so up to `worker_count` run concurrently (threads
  sharing one cached predictor; worker count is bounded by GPU memory).
- Results are yielded strictly in presentation order through a bounded
  reorder window, so memory stays O(worker_count) frames regardless of clip
  length (spec §26, §29).
- Each finished frame is cached as a Stage 1 `spatial_photo.h5` under
  `<work_dir>/frames/NNNNNN.h5`; with `resume=True` valid cached frames are
  reused instead of recomputed.
- A failing frame is journalled with its error and processing continues so
  every failure is found in one pass, but nothing after the gap is yielded and
  `FrameFailuresError` is raised at the end: a failed frame is never silently
  omitted from the spatial video (spec §32).
"""

from __future__ import annotations

import shutil
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .contract import SpatialFrame
from .journal import Journal
from .spatializer import FrameTimings
from .video_io import DecodedFrame

Spatialize = Callable[[DecodedFrame], tuple[SpatialFrame, FrameTimings]]

_STAGE2_KEYS = ("stage2_timestamp", "stage2_pts", "stage2_source_frame_index")


class FrameFailuresError(RuntimeError):
    """One or more selected frames failed Stage 1; see `.failures`."""

    def __init__(self, failures: list[dict]):
        self.failures = failures
        summary = ", ".join(f"frame {f['index']} ({f['error']})" for f in failures)
        super().__init__(f"{len(failures)} frame(s) failed Stage 1: {summary}")


@dataclass
class FrameOutcome:
    frame: SpatialFrame
    timings: FrameTimings | None  # None when the frame came from the resume cache
    cached: bool


def _cache_path(cache_dir: Path, index: int) -> Path:
    return cache_dir / f"{index:06d}.h5"


def _save_cached(cache_dir: Path, frame: SpatialFrame) -> None:
    from sharp_spatialize import hdf5_io

    result = frame.to_stage1()
    result.metadata.update({
        "stage2_timestamp": float(frame.timestamp),
        "stage2_pts": int(frame.pts),
        "stage2_source_frame_index": int(frame.source_frame_index),
    })
    hdf5_io.save(_cache_path(cache_dir, frame.source_frame_index), result)


def _load_cached(cache_dir: Path, decoded: DecodedFrame) -> SpatialFrame | None:
    """The cached frame if it exists, loads, belongs to `decoded`, and validates."""
    from sharp_spatialize import hdf5_io
    from sharp_spatialize.validation import validate_in_memory

    path = _cache_path(cache_dir, decoded.index)
    if not path.exists():
        return None
    try:
        result = hdf5_io.load(path)
        if int(result.metadata["stage2_source_frame_index"]) != decoded.index:
            return None
        timestamp = float(result.metadata["stage2_timestamp"])
        pts = int(result.metadata["stage2_pts"])
        for key in _STAGE2_KEYS:
            result.metadata.pop(key)
        validate_in_memory(result)
    except Exception:
        return None
    return SpatialFrame.from_stage1(result, timestamp, pts=pts, source_frame_index=decoded.index)


def process_frames(
    frames: Iterable[DecodedFrame],
    spatialize: Spatialize,
    work_dir: str | Path,
    worker_count: int = 1,
    resume: bool = False,
) -> Iterator[FrameOutcome]:
    """Spatialize `frames`, yielding outcomes in presentation order."""
    if worker_count < 1:
        raise ValueError(f"worker_count must be >= 1, got {worker_count}")
    work_dir = Path(work_dir)
    cache_dir = work_dir / "frames"
    journal_path = work_dir / "journal.json"
    if not resume:
        shutil.rmtree(cache_dir, ignore_errors=True)
        journal_path.unlink(missing_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    journal = Journal(journal_path)
    return _run(frames, spatialize, cache_dir, journal, worker_count, resume)


def _run(frames, spatialize, cache_dir, journal, worker_count, resume):
    def work(decoded: DecodedFrame) -> FrameOutcome:
        if resume and journal.is_done(decoded.index):
            cached = _load_cached(cache_dir, decoded)
            if cached is not None:
                return FrameOutcome(cached, None, True)
        frame, timings = spatialize(decoded)
        _save_cached(cache_dir, frame)
        return FrameOutcome(frame, timings, False)

    def run_now(decoded: DecodedFrame) -> Future:
        future: Future = Future()
        try:
            future.set_result(work(decoded))
        except Exception as exc:
            future.set_exception(exc)
        return future

    class Deferred:
        """Single-worker mode: compute when the consumer asks, not ahead of it."""

        def __init__(self, decoded):
            self._decoded = decoded

        def result(self):
            return work(self._decoded)

    window = 1 if worker_count == 1 else 2 * worker_count
    executor = ThreadPoolExecutor(max_workers=worker_count) if worker_count > 1 else None
    source = iter(frames)
    pending: deque = deque()
    failures: list[dict] = []
    exhausted = False
    first = True
    try:
        while True:
            while not exhausted and len(pending) < window:
                decoded = next(source, None)
                if decoded is None:
                    exhausted = True
                    break
                if executor is None:
                    pending.append((decoded, Deferred(decoded)))
                elif first:
                    # Run one frame alone first so shared lazy state (the cached
                    # predictor, gsplat's JIT build) initializes on one thread.
                    pending.append((decoded, run_now(decoded)))
                else:
                    pending.append((decoded, executor.submit(work, decoded)))
                first = False
            if not pending:
                break

            decoded, future = pending.popleft()
            try:
                outcome = future.result()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                journal.mark_failed(decoded.index, decoded.pts, decoded.timestamp, error)
                failures.append({"index": decoded.index, "pts": decoded.pts,
                                 "timestamp": decoded.timestamp, "error": error})
                continue
            journal.mark_done(decoded.index, decoded.pts, decoded.timestamp,
                              in_order=not failures)
            if not failures:
                yield outcome
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    if failures:
        raise FrameFailuresError(failures)
