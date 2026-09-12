"""Tests for sharp_video.validation, sharp_video.metrics and sharp_video.inspect."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
from PIL import Image
from synthetic import make_frames, make_info

from sharp_video.inspect import timestamp_table, write_inspection
from sharp_video.metrics import temporal_metrics
from sharp_video.sequence_io import SequenceReader, SequenceWriter
from sharp_video.validation import validate_sequence


def _write(path, frames):
    with SequenceWriter(path, make_info()) as writer:
        for frame in frames:
            writer.write_frame(frame)
    return path


def test_consistent_sequence_passes(tmp_path):
    report = validate_sequence(_write(tmp_path / "s.h5", make_frames(3, drift=0.2)))
    assert report.ok, report.failures
    assert report.frame_count == 3 and report.fps == 30.0 and report.cfr


def test_broken_rotation_reported_with_frame_index(tmp_path):
    frames = make_frames(3)
    frames[1].views[5].R = frames[1].views[5].R * np.float32(1.5)
    report = validate_sequence(_write(tmp_path / "s.h5", frames))
    assert not report.ok
    assert any(f.startswith("frame 1:") and "det" in f for f in report.failures)


def test_duplicate_timestamp_and_bad_frame_count_detected(tmp_path):
    path = _write(tmp_path / "s.h5", make_frames(3))
    with h5py.File(path, "r+") as f:
        f["frames/000002"].attrs["timestamp"] = f["frames/000001"].attrs["timestamp"]
        f["metadata"].attrs["frame_count"] = 5
    report = validate_sequence(path)
    assert any("timestamp" in f for f in report.failures)
    assert any("frame_count" in f for f in report.failures)


def test_intrinsics_change_detected(tmp_path):
    frames = make_frames(2)
    for view in frames[1].views:
        view.K = view.K.copy()
        view.K[0, 0] *= 1.01
    report = validate_sequence(_write(tmp_path / "s.h5", frames), check_frames=False)
    assert any("intrinsics" in f for f in report.failures)


def test_static_sequence_has_no_temporal_change(tmp_path):
    reader = SequenceReader(_write(tmp_path / "s.h5", make_frames(3)))
    metrics = temporal_metrics(reader)
    assert len(metrics.per_pair) == 2
    for key in ("rgb_flicker", "depth_flicker", "camera_translation", "center_depth_change"):
        assert metrics.summary[key]["max"] == pytest.approx(0.0, abs=1e-6)
    # Stage 1 stores R as float32 (orthonormal to ~1e-7), i.e. a ~1e-5 degree noise floor.
    assert metrics.summary["camera_rotation_deg"]["max"] < 1e-4


def test_moving_content_and_cameras_are_measured(tmp_path):
    reader = SequenceReader(_write(tmp_path / "s.h5", make_frames(3, drift=0.4, motion=0.3)))
    summary = temporal_metrics(reader).summary
    assert summary["rgb_flicker"]["mean"] > 0
    assert summary["depth_flicker"]["mean"] > 0
    assert summary["camera_translation"]["max"] > 0
    assert summary["center_depth_change"]["max"] == pytest.approx(0.4, rel=0.05)


def test_inspection_writes_sheets_and_cameras(tmp_path):
    reader = SequenceReader(_write(tmp_path / "s.h5", make_frames(2, drift=0.2)))
    paths = write_inspection(reader, 1, tmp_path / "inspect")
    names = {p.name for p in paths}
    assert {"frame_000001_rgb.png", "frame_000001_depth.png",
            "frame_000001_cameras.json"} <= names
    assert Image.open(tmp_path / "inspect" / "frame_000001_rgb.png").size == (96, 96)
    cameras = json.loads((tmp_path / "inspect" / "frame_000001_cameras.json").read_text())
    assert len(cameras["views"]) == 9 and "K" in cameras["views"][0]
    assert cameras["timestamp"] == pytest.approx(1 / 30.0)


def test_timestamp_table(tmp_path):
    reader = SequenceReader(_write(tmp_path / "s.h5", make_frames(3)))
    table = timestamp_table(reader)
    assert [row["frame"] for row in table] == [0, 1, 2]
    assert table[1]["delta_s"] == pytest.approx(1 / 30.0)
    assert table[0]["delta_s"] is None
