"""Tests for sharp_video.journal and sharp_video.runner."""

from __future__ import annotations

import json
import random
import threading
import time

import numpy as np
import pytest
from sharp_video.contract import SpatialFrame
from sharp_video.journal import Journal
from sharp_video.runner import FrameFailuresError, process_frames
from sharp_video.spatializer import FrameTimings
from sharp_video.video_io import DecodedFrame
from synthetic import make_stage1_result


def _decoded_frames(n):
    return [
        DecodedFrame(index=i, pts=i * 512, timestamp=i / 30.0,
                     rgb=np.zeros((32, 32, 3), np.uint8))
        for i in range(n)
    ]


class FakeSpatializer:
    """Stage 1 stand-in with optional failures and random per-frame delays."""

    def __init__(self, fail_on=(), delay=0.0, seed=0):
        """Configure the test double."""
        self.fail_on = set(fail_on)
        self.delay = delay
        self.calls = []
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def __call__(self, decoded):
        """Stand in for the replaced component for one call."""
        with self._lock:
            self.calls.append(decoded.index)
            delay = self._rng.uniform(0, self.delay)
        time.sleep(delay)
        if decoded.index in self.fail_on:
            raise RuntimeError(f"stage 1 exploded on frame {decoded.index}")
        frame = SpatialFrame.from_stage1(
            make_stage1_result(phase=0.1 * decoded.index), decoded.timestamp,
            pts=decoded.pts, source_frame_index=decoded.index,
        )
        return frame, FrameTimings(0.1, 0.2, 0.01, 0.4, None)


def test_single_worker_yields_all_frames_in_order(tmp_path):
    """Single worker yields all frames in order."""
    spatialize = FakeSpatializer()
    outcomes = list(process_frames(_decoded_frames(4), spatialize, tmp_path))
    assert [o.frame.source_frame_index for o in outcomes] == [0, 1, 2, 3]
    assert [o.frame.timestamp for o in outcomes] == [i / 30.0 for i in range(4)]
    assert all(not o.cached and o.timings is not None for o in outcomes)
    journal = json.loads((tmp_path / "journal.json").read_text())
    assert journal["last_completed"] == 3


def test_parallel_workers_restore_presentation_order(tmp_path):
    """Parallel workers restore presentation order."""
    spatialize = FakeSpatializer(delay=0.02, seed=3)
    outcomes = list(process_frames(_decoded_frames(10), spatialize, tmp_path, worker_count=4))
    assert [o.frame.source_frame_index for o in outcomes] == list(range(10))
    assert sorted(spatialize.calls) == list(range(10))


def test_failed_frame_is_recorded_and_never_silently_dropped(tmp_path):
    """Failed frame is recorded and never silently dropped."""
    spatialize = FakeSpatializer(fail_on={2})
    yielded = []
    with pytest.raises(FrameFailuresError) as info:
        for outcome in process_frames(_decoded_frames(5), spatialize, tmp_path):
            yielded.append(outcome.frame.source_frame_index)

    assert yielded == [0, 1]  # nothing after the gap reaches the consumer
    assert sorted(spatialize.calls) == [0, 1, 2, 3, 4]  # the rest still processed
    (failure,) = info.value.failures
    assert failure["index"] == 2 and "exploded" in failure["error"]
    journal = Journal(tmp_path / "journal.json")
    assert [f["index"] for f in journal.failures] == [2]
    assert journal.is_done(3) and journal.is_done(4)
    assert journal.last_completed == 1  # last frame completed in presentation order


def test_resume_recomputes_only_missing_frames(tmp_path):
    """Resume recomputes only missing frames."""
    with pytest.raises(FrameFailuresError):
        list(process_frames(_decoded_frames(5), FakeSpatializer(fail_on={2}), tmp_path))

    retry = FakeSpatializer()
    outcomes = list(process_frames(_decoded_frames(5), retry, tmp_path, resume=True))
    assert retry.calls == [2]
    assert [o.frame.source_frame_index for o in outcomes] == [0, 1, 2, 3, 4]
    assert [o.cached for o in outcomes] == [True, True, False, True, True]
    assert Journal(tmp_path / "journal.json").failures == []
    # Cached frames come back bit-exact.
    fresh = FakeSpatializer()(_decoded_frames(5)[3])[0]
    np.testing.assert_array_equal(outcomes[3].frame.stacked()["rgb"], fresh.stacked()["rgb"])
    assert outcomes[3].frame.pts == fresh.pts


def test_corrupt_cache_file_is_recomputed(tmp_path):
    """Corrupt cache file is recomputed."""
    list(process_frames(_decoded_frames(3), FakeSpatializer(), tmp_path))
    (tmp_path / "frames" / "000001.h5").write_bytes(b"not hdf5")
    retry = FakeSpatializer()
    list(process_frames(_decoded_frames(3), retry, tmp_path, resume=True))
    assert retry.calls == [1]


def test_without_resume_everything_is_recomputed(tmp_path):
    """Without resume everything is recomputed."""
    list(process_frames(_decoded_frames(3), FakeSpatializer(), tmp_path))
    again = FakeSpatializer()
    list(process_frames(_decoded_frames(3), again, tmp_path))
    assert again.calls == [0, 1, 2]


def test_frames_are_pulled_lazily_within_a_bounded_window(tmp_path):
    """Frames are pulled lazily within a bounded window."""
    pulled = []

    def source():
        for frame in _decoded_frames(20):
            pulled.append(frame.index)
            yield frame

    outcomes = process_frames(source(), FakeSpatializer(delay=0.01), tmp_path, worker_count=2)
    first = next(outcomes)
    assert first.frame.source_frame_index == 0
    assert len(pulled) <= 1 + 2 * 2 + 1  # never races far ahead of the consumer
    rest = list(outcomes)
    assert len(rest) == 19
