"""Tests for sharp_video.video_io and sharp_video.scheduler."""

from __future__ import annotations

from fractions import Fraction

import av
import numpy as np
import pytest
from sharp_video.scheduler import FrameSelection
from sharp_video.video_io import UnsupportedVideoError, iter_frames, probe_video

NTSC = Fraction(30000, 1001)


def _frame_rgb(i: int, width: int, height: int) -> np.ndarray:
    """A grey background with a coloured square that moves 2 px per frame."""
    rgb = np.full((height, width, 3), 96, dtype=np.uint8)
    x = (2 * i) % (width - 16)
    rgb[16:32, x:x + 16] = (200, 40, 40)
    return rgb


def _write_video(path, codec="libx264", n=12, width=64, height=48, rate=NTSC, bframes=True):
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=rate)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        if codec == "libx264":
            stream.options = {"bf": "2" if bframes else "0", "crf": "12"}
        elif codec == "libx265":
            stream.options = {"x265-params": "log-level=error:crf=12"}
        for i in range(n):
            frame = av.VideoFrame.from_ndarray(_frame_rgb(i, width, height), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


@pytest.fixture(scope="module")
def h264_clip(tmp_path_factory):
    """A 12-frame 30000/1001 fps H.264 clip with B-frames."""
    return _write_video(tmp_path_factory.mktemp("video") / "clip.mp4")


def test_probe_reports_input_properties(h264_clip):
    """Probe reports input properties."""
    info = probe_video(h264_clip)
    assert (info.width, info.height) == (64, 48)
    assert info.codec == "h264"
    assert info.fps == pytest.approx(30000 / 1001)
    assert info.frame_count == 12
    assert info.duration == pytest.approx(12 * 1001 / 30000, abs=1e-3)
    assert info.pix_fmt == "yuv420p"
    assert info.time_base  # e.g. "1/30000"
    for key in ("color_primaries", "color_trc", "colorspace", "color_range"):
        assert hasattr(info, key)


def test_frames_decode_in_presentation_order_with_timestamps(h264_clip):
    """Frames decode in presentation order with timestamps."""
    frames = list(iter_frames(h264_clip))
    assert [f.index for f in frames] == list(range(12))
    timestamps = np.array([f.timestamp for f in frames])
    np.testing.assert_allclose(timestamps, np.arange(12) * 1001 / 30000, atol=1e-3)
    assert all(b.pts > a.pts for a, b in zip(frames, frames[1:]))


def test_decoded_rgb_matches_source(h264_clip):
    """Decoded rgb matches source."""
    for frame in iter_frames(h264_clip):
        assert frame.rgb.shape == (48, 64, 3) and frame.rgb.dtype == np.uint8
        expected = _frame_rgb(frame.index, 64, 48).astype(np.float32)
        assert np.abs(frame.rgb.astype(np.float32) - expected).mean() < 6.0


def test_selection_start_end(h264_clip):
    """Selection start end."""
    frames = list(iter_frames(h264_clip, FrameSelection(start_frame=3, end_frame=7)))
    assert [f.index for f in frames] == [3, 4, 5, 6]
    # Original timing is retained, not re-based to zero.
    assert frames[0].timestamp == pytest.approx(3 * 1001 / 30000, abs=1e-3)


def test_selection_max_frames(h264_clip):
    """Selection max frames."""
    frames = list(iter_frames(h264_clip, FrameSelection(start_frame=2, max_frames=3)))
    assert [f.index for f in frames] == [2, 3, 4]


def test_selection_validation():
    """Selection validation."""
    with pytest.raises(ValueError):
        FrameSelection(start_frame=-1).validate()
    with pytest.raises(ValueError):
        FrameSelection(start_frame=5, end_frame=5).validate()
    with pytest.raises(ValueError):
        FrameSelection(max_frames=0).validate()


def test_hevc_is_supported(tmp_path):
    """Hevc is supported."""
    path = _write_video(tmp_path / "clip_hevc.mp4", codec="libx265", n=4)
    info = probe_video(path)
    assert info.codec == "hevc"
    assert len(list(iter_frames(path))) == 4


def test_unsupported_codec_rejected(tmp_path):
    """Unsupported codec rejected."""
    path = _write_video(tmp_path / "clip_mpeg4.mp4", codec="mpeg4", n=3)
    with pytest.raises(UnsupportedVideoError, match="mpeg4"):
        probe_video(path)
