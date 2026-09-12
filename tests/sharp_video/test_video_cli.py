"""Tests for sharp_video.cli."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner
from synthetic import make_frames, make_info

from sharp_video import cli
from sharp_video.sequence_io import SequenceWriter


@pytest.fixture
def sequence(tmp_path):
    path = tmp_path / "seq.h5"
    with SequenceWriter(path, make_info()) as writer:
        for frame in make_frames(3, drift=0.2):
            writer.write_frame(frame)
    return path


@pytest.mark.parametrize("command", [cli.video_to_miv_cli, cli.validate_sequence_cli,
                                     cli.inspect_cli, cli.miv_encode_cli, cli.miv_validate_cli])
def test_help(command):
    result = CliRunner().invoke(command, ["--help"])
    assert result.exit_code == 0, result.output


def test_video_to_miv_parses_options(tmp_path, monkeypatch):
    captured = {}

    def fake_run(options, **kwargs):
        captured["options"] = options
        raise SystemExit(0)

    monkeypatch.setattr(cli, "run_video_to_miv", fake_run)
    camera = tmp_path / "camera.json"
    camera.write_text(json.dumps({"angle_deg": 12}))
    video = tmp_path / "in.mp4"
    video.write_bytes(b"")
    CliRunner().invoke(cli.video_to_miv_cli, [
        "--input", str(video), "--output", str(tmp_path / "o.miv"),
        "--dump-hdf5", str(tmp_path / "d.h5"), "--start-frame", "5", "--end-frame", "20",
        "--max-frames", "10", "--worker-count", "2", "--camera-config", str(camera),
        "--output-resolution", "1280x720", "--mode", "streaming", "--intra-period", "16",
    ])
    options = captured["options"]
    assert options.selection.start_frame == 5 and options.selection.end_frame == 20
    assert options.selection.max_frames == 10
    assert options.worker_count == 2
    assert options.camera.angle_deg == 12.0
    assert options.output_size == (1280, 720)
    assert options.mode == "streaming"
    assert options.dump_hdf5 == tmp_path / "d.h5"
    assert options.intra_period == 16


def test_bad_output_resolution_rejected(tmp_path):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"")
    result = CliRunner().invoke(cli.video_to_miv_cli, [
        "--input", str(video), "--output", str(tmp_path / "o.miv"),
        "--output-resolution", "1279x720",
    ])
    assert result.exit_code != 0
    assert "multiple of 8" in result.output


def test_validate_sequence_cli(sequence, tmp_path):
    out = tmp_path / "temporal.json"
    result = CliRunner().invoke(cli.validate_sequence_cli,
                                [str(sequence), "--temporal-json", str(out)])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output
    assert "camera_translation" in json.loads(out.read_text())["summary"]


def test_inspect_cli(sequence, tmp_path):
    result = CliRunner().invoke(cli.inspect_cli, [str(sequence), "--frame", "0", "--frame", "2",
                                                  "--out", str(tmp_path / "ins")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "ins" / "frame_000002_depth.png").exists()
    assert (tmp_path / "ins" / "timestamps.json").exists()
