"""Tests for sharp_video.miv.validate (MIV correctness against the reference sequence)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from synthetic import make_frames, make_info

from sharp_video.miv.decode import DecodedView
from sharp_video.miv.tmiv_config import VIEW_NAMES
from sharp_video.miv.validate import Thresholds, compare_to_reference
from sharp_video.sequence_io import SequenceReader, SequenceWriter


class FakeDecoded:
    """Duck-types DecodedMiv from a list of SpatialFrames."""

    def __init__(self, frames, fps=30.0, width=32, height=32):
        self.frame_count = len(frames)
        self.fps = fps
        self.resolution = (width, height)
        self.view_names = list(VIEW_NAMES)
        self._frames = frames

    def frames(self):
        for frame in self._frames:
            yield {
                name: DecodedView(rgb=v.rgb, depth=v.depth, mask=v.mask, K=v.K, R=v.R, C=v.C)
                for name, v in zip(VIEW_NAMES, frame.views)
            }


@pytest.fixture
def reference(tmp_path):
    frames = make_frames(3, drift=0.3)
    path = tmp_path / "seq.h5"
    with SequenceWriter(path, make_info()) as writer:
        for frame in frames:
            writer.write_frame(frame)
    return SequenceReader(path), frames


def _manifest(frames, fps=30.0):
    return {"frame_count": len(frames), "fps": fps,
            "frames": [{"timestamp": f.timestamp} for f in frames]}


def test_perfect_decode_passes(reference):
    reader, frames = reference
    report = compare_to_reference(FakeDecoded(frames), reader, _manifest(frames))
    assert report.ok, report.failures
    assert report.stats["frame_count"] == 3
    assert report.stats["y_psnr_db_min"] == float("inf")
    assert report.stats["depth_median_rel_error_max"] == 0.0
    assert report.stats["mask_agreement_min"] == 1.0


def test_shifted_camera_named_in_failure(reference):
    reader, frames = reference
    decoded = copy.deepcopy(frames)
    decoded[1].views[2].C = decoded[1].views[2].C + np.float32(0.05)
    report = compare_to_reference(FakeDecoded(decoded), reader, _manifest(frames))
    assert not report.ok
    assert any("frame 1" in f and "v2" in f and "C" in f for f in report.failures)


def test_missing_frame_fails(reference):
    reader, frames = reference
    report = compare_to_reference(FakeDecoded(frames[:2]), reader, _manifest(frames))
    assert not report.ok
    assert any("frame count" in f for f in report.failures)


def test_wrong_frame_rate_fails(reference):
    reader, frames = reference
    report = compare_to_reference(FakeDecoded(frames, fps=25.0), reader, _manifest(frames))
    assert any("fps" in f for f in report.failures)


def test_timing_mismatch_fails(reference):
    reader, frames = reference
    manifest = _manifest(frames)
    manifest["frames"][2]["timestamp"] += 0.5
    report = compare_to_reference(FakeDecoded(frames), reader, manifest)
    assert any("timestamp" in f for f in report.failures)


def test_noisy_texture_fails_psnr_threshold(reference):
    reader, frames = reference
    decoded = copy.deepcopy(frames)
    rng = np.random.default_rng(0)
    for view in decoded[0].views:
        noise = rng.integers(-80, 80, size=view.rgb.shape)
        view.rgb = np.clip(view.rgb.astype(int) + noise, 0, 255).astype(np.uint8)
    report = compare_to_reference(FakeDecoded(decoded), reader, _manifest(frames),
                                  Thresholds(y_psnr_db=30.0))
    assert any("PSNR" in f for f in report.failures)


def test_texture_outside_the_valid_region_is_ignored(reference):
    """MIV codes only occupied samples; the decoder fills the rest with neutral grey."""
    reader, frames = reference
    decoded = copy.deepcopy(frames)
    for view in decoded[0].views:
        view.rgb = np.where(view.mask[..., None] == 1, view.rgb, np.uint8(130)).astype(np.uint8)
    report = compare_to_reference(FakeDecoded(decoded), reader, _manifest(frames))
    assert report.ok, report.failures


def test_lost_view_occupancy_fails(reference):
    reader, frames = reference
    decoded = copy.deepcopy(frames)
    decoded[1].views[4].mask = np.zeros_like(decoded[1].views[4].mask)
    report = compare_to_reference(FakeDecoded(decoded), reader, _manifest(frames))
    assert any("frame 1 v4" in f and "mask" in f for f in report.failures)


def test_depth_error_fails(reference):
    reader, frames = reference
    decoded = copy.deepcopy(frames)
    decoded[2].views[4].depth = decoded[2].views[4].depth * np.float32(1.2)
    report = compare_to_reference(FakeDecoded(decoded), reader, _manifest(frames))
    assert any("depth" in f and "frame 2" in f for f in report.failures)


@pytest.mark.slow
def test_real_tmiv_round_trip_preserves_views_cameras_and_timing(tmp_path):
    """Encode with the patched TMIV, decode with TmivDecoder, compare with the source."""
    from sharp_video.miv.decode import DecodedMiv
    from sharp_video.miv.encoder import MIVEncoderConfig, encode_sequence
    from sharp_video.miv.tmiv import TmivNotFoundError, find_tmiv
    from sharp_video.miv.validate import validate_miv

    try:
        tmiv = find_tmiv()
    except TmivNotFoundError as exc:
        pytest.skip(str(exc))

    # A tilted plane: like real SHARP depth, no view sees one exactly constant depth.
    frames = make_frames(4, width=64, height=64, drift=0.3, motion=0.2, tilt=0.4)
    path = tmp_path / "seq.h5"
    with SequenceWriter(path, make_info(64, 64)) as writer:
        for frame in frames:
            writer.write_frame(frame)
    reader = SequenceReader(path)
    config = MIVEncoderConfig(work_dir=tmp_path / "enc", tmiv=tmiv, intra_period=16, threads=2)
    encode_sequence(reader, config, tmp_path / "out.miv")

    report = validate_miv(tmp_path / "out.miv", reader, tmp_path / "dec", tmiv=tmiv)
    assert report.ok, report.failures
    assert report.stats["frame_count"] == 4 and len(report.stats["views"]) == 9
    # The moving cameras reached the decoder as per-frame view parameter updates.
    assert report.stats["camera_update_frames"] == [0, 1, 2, 3]
    assert DecodedMiv(tmp_path / "dec" / "decoded", 4).camera_update_frames == [0, 1, 2, 3]
