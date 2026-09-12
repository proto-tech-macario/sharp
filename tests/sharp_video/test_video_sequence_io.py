"""Tests for sharp_video.sequence_io (temporal HDF5)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from sharp_video.sequence_io import SequenceReader, SequenceWriter
from synthetic import make_frames, make_info


def _write(path, frames, info=None):
    with SequenceWriter(path, info or make_info()) as writer:
        for frame in frames:
            writer.write_frame(frame)


def test_round_trip_is_bit_exact(tmp_path):
    """Round trip is bit exact."""
    frames = make_frames(3, drift=0.2)
    path = tmp_path / "seq.h5"
    _write(path, frames)

    reader = SequenceReader(path)
    assert len(reader) == 3
    for original, loaded in zip(frames, reader):
        assert loaded.timestamp == original.timestamp
        assert loaded.pts == original.pts
        assert loaded.source_frame_index == original.source_frame_index
        for a, b in zip(original.views, loaded.views):
            for name in ("rgb", "depth", "mask", "K", "R", "C"):
                np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
                assert getattr(a, name).dtype == getattr(b, name).dtype
        assert loaded.metadata == original.metadata


def test_schema_matches_spec_layout(tmp_path):
    """Schema matches spec layout."""
    path = tmp_path / "seq.h5"
    _write(path, make_frames(2))
    with h5py.File(path, "r") as f:
        meta = f["metadata"].attrs
        assert meta["frame_count"] == 2
        assert meta["view_count"] == 9
        assert meta["width"] == 32 and meta["height"] == 32
        assert meta["fps"] == 30.0
        assert "timestamp" in f["frames/000001"].attrs
        for v in range(9):
            group = f[f"frames/000000/view_{v:02d}"]
            assert group["rgb"].shape == (32, 32, 3) and group["rgb"].dtype == np.uint8
            assert group["depth"].shape == (32, 32) and group["depth"].dtype == np.float32
            assert group["mask"].dtype == np.uint8
            assert group["K"].shape == (3, 3) and group["R"].shape == (3, 3)
            assert group["C"].shape == (3,)


def test_reader_info_and_timestamps(tmp_path):
    """Reader info and timestamps."""
    path = tmp_path / "seq.h5"
    _write(path, make_frames(4, fps=25.0), make_info(fps=25.0))
    reader = SequenceReader(path)
    assert reader.info.fps == 25.0
    assert reader.info.width == 32
    np.testing.assert_allclose(reader.timestamps(), [0.0, 0.04, 0.08, 0.12])
    assert reader.frame(2).timestamp == pytest.approx(0.08)


def test_non_increasing_timestamp_rejected(tmp_path):
    """Non increasing timestamp rejected."""
    frames = make_frames(2)
    frames[1].timestamp = frames[0].timestamp
    with pytest.raises(ValueError, match="timestamp"):
        _write(tmp_path / "seq.h5", frames)


def test_resolution_change_rejected(tmp_path):
    """Resolution change rejected."""
    frames = [make_frames(1)[0], make_frames(2, width=16, height=16)[1]]
    with pytest.raises(ValueError, match="resolution"):
        _write(tmp_path / "seq.h5", frames)


def test_failed_write_leaves_no_file(tmp_path):
    """Failed write leaves no file."""
    path = tmp_path / "seq.h5"
    frames = make_frames(2)
    frames[1].timestamp = -1.0
    with pytest.raises(ValueError):
        _write(path, frames)
    assert not path.exists()
    assert not path.with_suffix(".h5.tmp").exists()


def test_file_only_appears_on_close(tmp_path):
    """File only appears on close."""
    path = tmp_path / "seq.h5"
    writer = SequenceWriter(path, make_info())
    writer.write_frame(make_frames(1)[0])
    assert not path.exists()
    writer.close()
    assert path.exists()


def test_reader_is_lazy(tmp_path, monkeypatch):
    """Reader is lazy."""
    path = tmp_path / "seq.h5"
    _write(path, make_frames(3))
    reader = SequenceReader(path)
    loaded = []
    original = SequenceReader.frame

    def spy(self, index):
        loaded.append(index)
        return original(self, index)

    monkeypatch.setattr(SequenceReader, "frame", spy)
    iterator = iter(reader)
    next(iterator)
    assert loaded == [0]
