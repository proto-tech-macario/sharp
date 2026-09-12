"""Tests for sharp_video.pipeline and sharp_video.report (video -> MIV orchestration)."""

from __future__ import annotations

import json

import pytest
from fakes import FakeRunner, FakeSpatializer
from sharp_video.miv.tmiv import TmivInstall
from sharp_video.miv.tmiv_config import VIEW_NAMES, geometry_input_path, texture_input_path
from sharp_video.pipeline import PipelineOptions, default_output_size, run_video_to_miv
from sharp_video.runner import FrameFailuresError
from sharp_video.scheduler import FrameSelection
from sharp_video.sequence_io import SequenceReader
from test_video_io import _write_video


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """A 6-frame 30000/1001 fps H.264 clip."""
    return _write_video(tmp_path_factory.mktemp("clip") / "clip.mp4", n=6)


def _run(tmp_path, clip, **kwargs):
    options = PipelineOptions(
        input=clip, output=tmp_path / "out.miv", output_size=(32, 32),
        work_dir=tmp_path / "work", **kwargs,
    )
    runner = FakeRunner()
    result = run_video_to_miv(options, spatialize=FakeSpatializer(),
                              runner=runner, tmiv=TmivInstall(tmp_path / "tmiv"))
    return result, runner


def test_default_output_size_rounds_down_to_multiple_of_8():
    """Default output size rounds down to multiple of 8."""
    assert default_output_size(1920, 1080) == (1920, 1080)
    assert default_output_size(1918, 1078) == (1912, 1072)
    assert default_output_size(64, 48) == (64, 48)


def test_two_step_mode_writes_sequence_then_miv(tmp_path, clip):
    """Two step mode writes sequence then miv."""
    result, runner = _run(tmp_path, clip)
    assert result.sequence_path == tmp_path / "work" / "sequence.h5"
    reader = SequenceReader(result.sequence_path)
    assert len(reader) == 6
    assert reader.info.fps == pytest.approx(30000 / 1001)
    assert (tmp_path / "out.miv").exists()
    assert result.miv.frame_count == 6
    assert len(runner.commands) == 1


def test_streaming_mode_with_dump_matches_two_step(tmp_path, clip):
    """Streaming mode with dump matches two step."""
    two = tmp_path / "two"
    two.mkdir()
    streaming = tmp_path / "streaming"
    streaming.mkdir()
    _run(two, clip, mode="two-step")
    result, _ = _run(streaming, clip, mode="streaming", dump_hdf5=streaming / "dump.h5")

    assert result.sequence_path == streaming / "dump.h5"
    assert len(SequenceReader(streaming / "dump.h5")) == 6
    for name in VIEW_NAMES:
        for rel in (texture_input_path(name, 32, 32), geometry_input_path(name, 32, 32)):
            a = (two / "work" / "miv" / "input" / rel).read_bytes()
            b = (streaming / "work" / "miv" / "input" / rel).read_bytes()
            assert a == b  # both modes feed the encoder identical spatial content
    for t in range(6):
        rel = f"S2/seq/{t:06d}.json"
        assert (two / "work/miv/input" / rel).read_text() == \
            (streaming / "work/miv/input" / rel).read_text()


def test_streaming_without_dump_writes_no_sequence(tmp_path, clip):
    """Streaming without dump writes no sequence."""
    result, _ = _run(tmp_path, clip, mode="streaming")
    assert result.sequence_path is None
    assert not (tmp_path / "work" / "sequence.h5").exists()


def test_selection_is_applied_and_timing_retained(tmp_path, clip):
    """Selection is applied and timing retained."""
    result, _ = _run(tmp_path, clip, selection=FrameSelection(start_frame=2, max_frames=3))
    manifest = json.loads(result.miv.manifest_path.read_text())
    assert [f["source_frame_index"] for f in manifest["frames"]] == [2, 3, 4]
    assert manifest["frames"][0]["timestamp"] == pytest.approx(2 * 1001 / 30000, abs=1e-3)


def test_failed_frame_stops_before_miv(tmp_path, clip):
    """Failed frame stops before miv."""
    options = PipelineOptions(input=clip, output=tmp_path / "out.miv", output_size=(32, 32),
                              work_dir=tmp_path / "work")
    runner = FakeRunner()
    with pytest.raises(FrameFailuresError):
        run_video_to_miv(options, spatialize=FakeSpatializer(fail_on={3}), runner=runner,
                         tmiv=TmivInstall(tmp_path / "tmiv"))
    assert runner.commands == []
    assert not (tmp_path / "out.miv").exists()


def test_report_has_every_performance_field(tmp_path, clip):
    """Report has every performance field."""
    result, _ = _run(tmp_path, clip)
    report = json.loads((tmp_path / "out.miv.report.json").read_text())
    assert result.report_paths == (tmp_path / "out.miv.report.json",
                                   tmp_path / "out.miv.report.md")
    assert report["input"]["resolution"] == [64, 48]
    assert report["input"]["fps"] == pytest.approx(30000 / 1001)
    assert report["input"]["frame_count"] == 6
    assert report["input"]["duration_s"] > 0
    stage1 = report["stage1"]
    for key in ("sharp_and_3dgs_s_per_frame", "nine_view_render_s_per_frame",
                "total_stage1_s", "peak_gpu_memory_bytes"):
        assert key in stage1
    assert stage1["sharp_and_3dgs_s_per_frame"]["mean"] == pytest.approx(0.5)
    assert stage1["peak_gpu_memory_bytes"] == 3 * 2**30
    miv = report["miv"]
    for key in ("encode_time_s", "file_size_bytes", "average_bitrate_bps"):
        assert key in miv
    overall = report["overall"]
    for key in ("total_processing_time_s", "average_processing_fps", "miv_file_size_bytes",
                "average_miv_bitrate_bps", "peak_memory_bytes"):
        assert key in overall
    markdown = (tmp_path / "out.miv.report.md").read_text()
    assert "Average processing FPS" in markdown and "MIV file size" in markdown


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["two-step", "streaming"])
def test_real_tmiv_pipeline_on_h264_clip(tmp_path, clip, mode, monkeypatch):
    """H.264 clip -> (fake) Stage 1 -> real TMIV -> MIV that TmivDecoder validates."""
    from pathlib import Path

    from sharp_video.miv.tmiv import TmivNotFoundError, find_tmiv
    from sharp_video.miv.validate import validate_miv

    try:
        tmiv = find_tmiv()
    except TmivNotFoundError as exc:
        pytest.skip(str(exc))
    monkeypatch.chdir(tmp_path)  # anything written to the cwd would land here

    options = PipelineOptions(
        input=clip, output=tmp_path / "out.miv", mode=mode, output_size=(64, 64),
        work_dir=tmp_path / "work", dump_hdf5=tmp_path / "seq.h5", intra_period=16, threads=2,
    )
    result = run_video_to_miv(options, spatialize=FakeSpatializer(64, 64, drift=0.1, tilt=0.4),
                              tmiv=tmiv)
    assert result.miv.frame_count == 6

    report = validate_miv(result.miv.path, SequenceReader(result.sequence_path),
                          tmp_path / "dec", tmiv=tmiv)
    assert report.ok, report.failures
    assert report.stats["camera_update_frames"] == list(range(6))
    assert not (Path(tmp_path) / "rec.yuv").exists()
