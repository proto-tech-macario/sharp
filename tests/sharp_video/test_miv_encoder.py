"""Tests for sharp_video.miv.encoder (streaming SpatialSequence -> MIV)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from synthetic import make_frames, make_info

from sharp_video.miv.camera import to_tmiv_pose
from sharp_video.miv.depth import geometry_to_depth
from sharp_video.miv.encoder import MIVEncoder, MIVEncoderConfig, encode_sequence
from sharp_video.miv.tmiv import TmivInstall, TmivNotFoundError, find_tmiv
from sharp_video.miv.tmiv_config import (
    CONTENT_ID,
    VIEW_NAMES,
    bitstream_output_path,
    geometry_input_path,
    texture_input_path,
)
from sharp_video.miv.yuv import frame_nbytes, read_yuv420_frame, yuv420_to_rgb
from sharp_video.sequence_io import SequenceReader, SequenceWriter

W = H = 32


class FakeRunner:
    """Records the TMIV command and produces the bitstream encode.py would."""

    def __init__(self, payload=b"\x00MIV" * 250):
        self.commands = []
        self.payload = payload

    def __call__(self, cmd, log_path, cwd=None, env=None):
        self.commands.append([str(c) for c in cmd])
        output_dir = cmd[cmd.index("-o") + 1]
        target = output_dir / bitstream_output_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.payload)


def _config(tmp_path, **kwargs):
    return MIVEncoderConfig(work_dir=tmp_path / "work", tmiv=TmivInstall(tmp_path / "tmiv"),
                            **kwargs)


def _encode(tmp_path, frames, runner=None, fps=30.0, **kwargs):
    runner = runner or FakeRunner()
    encoder = MIVEncoder(_config(tmp_path, **kwargs), runner=runner)
    encoder.begin(fps=fps, width=W, height=H)
    for frame in frames:
        encoder.encode_frame(frame.timestamp, frame.views, pts=frame.pts,
                             source_frame_index=frame.source_frame_index)
    return encoder.finalize(tmp_path / "out.miv"), runner


def test_finalize_runs_tmiv_and_writes_output(tmp_path):
    frames = make_frames(4, drift=0.3)
    result, runner = _encode(tmp_path, frames)

    (cmd,) = runner.commands
    assert cmd[1].endswith("scripts/encode.py")
    assert cmd[cmd.index("-n") + 1] == "4"
    assert cmd[cmd.index("-s") + 1] == CONTENT_ID
    rates = cmd.index("-r")
    assert cmd[rates + 1:rates + 3] == ["RP0", "RP1"]
    assert cmd[cmd.index("-v") + 1] == "VVenC"
    assert cmd[cmd.index("-t") + 1] == str(tmp_path / "tmiv" / "install")

    assert (tmp_path / "out.miv").read_bytes() == runner.payload
    assert result.frame_count == 4 and result.fps == 30.0
    assert result.size_bytes == len(runner.payload)
    assert result.duration_s == pytest.approx(4 / 30.0)
    assert result.bitrate_bps == pytest.approx(len(runner.payload) * 8 / (4 / 30.0))
    assert result.cfr is True


def test_view_files_hold_every_frame(tmp_path):
    frames = make_frames(3)
    _encode(tmp_path, frames)
    input_dir = tmp_path / "work" / "input"
    for name in VIEW_NAMES:
        tex = input_dir / texture_input_path(name, W, H)
        geo = input_dir / geometry_input_path(name, W, H)
        assert tex.stat().st_size == 3 * frame_nbytes(W, H, 10)
        assert geo.stat().st_size == 3 * frame_nbytes(W, H, 16)


def test_texture_and_depth_written_faithfully(tmp_path):
    frames = make_frames(2)
    _encode(tmp_path, frames, depth_near=0.1, depth_far=1000.0)
    input_dir = tmp_path / "work" / "input"
    view = frames[1].views[4]
    with open(input_dir / texture_input_path("v4", W, H), "rb") as f:
        read_yuv420_frame(f, W, H, 10)
        rgb = yuv420_to_rgb(*read_yuv420_frame(f, W, H, 10), bit_depth=10)
    assert np.abs(rgb.astype(np.float32) - view.rgb).mean() < 3.0
    with open(input_dir / geometry_input_path("v4", W, H), "rb") as f:
        read_yuv420_frame(f, W, H, 16)
        samples, _, _ = read_yuv420_frame(f, W, H, 16)
    depth, mask = geometry_to_depth(samples, 0.1, 1000.0)
    np.testing.assert_array_equal(mask, view.mask)
    np.testing.assert_allclose(depth[mask == 1], view.depth[mask == 1], rtol=2e-3)


def test_per_frame_camera_files_follow_moving_cameras(tmp_path):
    frames = make_frames(3, drift=0.5)
    _encode(tmp_path, frames)
    for t, frame in enumerate(frames):
        seq = json.loads((tmp_path / "work" / "input" / CONTENT_ID / "seq"
                          / f"{t:06d}.json").read_text())
        assert seq["Frames_number"] == 3 and seq["Fps"] == 30.0
        cam = {c["Name"]: c for c in seq["cameras"]}["v0"]
        position, rotation = to_tmiv_pose(frame.views[0].R, frame.views[0].C)
        np.testing.assert_allclose(cam["Position"], position, atol=1e-6)
        np.testing.assert_allclose(cam["Rotation"], rotation, atol=1e-6)
    first = json.loads((tmp_path / "work/input/S2/seq/000000.json").read_text())
    last = json.loads((tmp_path / "work/input/S2/seq/000002.json").read_text())
    assert first["cameras"][0]["Position"] != last["cameras"][0]["Position"]


def test_manifest_records_timing_and_conventions(tmp_path):
    frames = make_frames(3)
    result, _ = _encode(tmp_path, frames)
    manifest = json.loads(result.manifest_path.read_text())
    assert result.manifest_path == tmp_path / "out.miv.json"
    assert manifest["frame_count"] == 3 and manifest["fps"] == 30.0
    assert [f["timestamp"] for f in manifest["frames"]] == [f.timestamp for f in frames]
    assert [f["pts"] for f in manifest["frames"]] == [f.pts for f in frames]
    assert manifest["views"] == list(VIEW_NAMES)
    assert manifest["resolution"] == [W, H]
    assert manifest["depth_range"] == [0.1, 1000.0]
    assert manifest["cfr"] is True


def test_variable_frame_rate_is_flagged(tmp_path):
    frames = make_frames(4)
    for frame, t in zip(frames, (0.0, 0.0333, 0.1, 0.1333)):
        frame.timestamp = t
    result, _ = _encode(tmp_path, frames)
    assert result.cfr is False


def test_out_of_range_depth_is_counted(tmp_path):
    frames = make_frames(1)
    result, _ = _encode(tmp_path, frames, depth_near=0.1, depth_far=4.0)  # plane is at ~5 m
    assert result.clamped_depth_pixels > 0


@pytest.mark.parametrize("problem", ["eight_views", "k_change", "timestamp", "size"])
def test_bad_input_rejected(tmp_path, problem):
    encoder = MIVEncoder(_config(tmp_path), runner=FakeRunner())
    encoder.begin(fps=30.0, width=W, height=H)
    frames = make_frames(2)
    encoder.encode_frame(frames[0].timestamp, frames[0].views)
    views = frames[1].views
    timestamp = frames[1].timestamp
    if problem == "eight_views":
        views = views[:8]
    elif problem == "k_change":
        views[3].K = views[3].K * 1.1
    elif problem == "timestamp":
        timestamp = frames[0].timestamp
    elif problem == "size":
        views = make_frames(1, width=16, height=16)[0].views
    with pytest.raises(ValueError):
        encoder.encode_frame(timestamp, views)


def test_encode_before_begin_and_odd_size(tmp_path):
    encoder = MIVEncoder(_config(tmp_path), runner=FakeRunner())
    with pytest.raises(RuntimeError):
        encoder.encode_frame(0.0, make_frames(1)[0].views)
    with pytest.raises(ValueError, match="even"):
        encoder.begin(fps=30.0, width=33, height=32)


def test_encode_sequence_from_hdf5_matches_streaming(tmp_path):
    frames = make_frames(3, drift=0.2)
    path = tmp_path / "seq.h5"
    with SequenceWriter(path, make_info()) as writer:
        for frame in frames:
            writer.write_frame(frame)

    streamed = tmp_path / "streamed"
    streamed.mkdir()
    _encode(streamed, frames)

    from_file = tmp_path / "from_file"
    from_file.mkdir()
    encode_sequence(SequenceReader(path), _config(from_file), from_file / "out.miv",
                    runner=FakeRunner())

    for name in VIEW_NAMES:
        for rel in (texture_input_path(name, W, H), geometry_input_path(name, W, H)):
            a = (streamed / "work" / "input" / rel).read_bytes()
            b = (from_file / "work" / "input" / rel).read_bytes()
            assert a == b


def _tmiv_or_skip():
    try:
        return find_tmiv()
    except TmivNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize("intra_period", [0, 1, 4, 8, 12, 64])
def test_intra_period_limited_to_vvenc_gop_sizes(tmp_path, intra_period):
    with pytest.raises(ValueError, match="intra_period"):
        _config(tmp_path, intra_period=intra_period)


@pytest.mark.slow
def test_real_tmiv_encodes_moving_camera_sequence(tmp_path):
    tmiv = _tmiv_or_skip()
    frames = make_frames(4, width=64, height=64, drift=0.3, motion=0.2)
    encoder = MIVEncoder(MIVEncoderConfig(work_dir=tmp_path / "work", tmiv=tmiv, intra_period=16,
                                          threads=2))
    encoder.begin(fps=30.0, width=64, height=64)
    for frame in frames:
        encoder.encode_frame(frame.timestamp, frame.views, pts=frame.pts)
    result = encoder.finalize(tmp_path / "out.miv")
    assert result.path.exists() and result.size_bytes > 0
