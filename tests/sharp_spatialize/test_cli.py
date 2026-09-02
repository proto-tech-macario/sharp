"""Tests for sharp_spatialize.cli that don't require CUDA or a real model."""

from __future__ import annotations

from click.testing import CliRunner

from sharp_spatialize.cli import generate_cli, validate_cli
from sharp_spatialize.hdf5_io import save


def test_validate_cli_passes_on_a_consistent_file(tmp_path, consistent_spatial_photo_result):
    path = tmp_path / "spatial_photo.h5"
    save(path, consistent_spatial_photo_result)

    result = CliRunner().invoke(validate_cli, [str(path)])

    assert result.exit_code == 0
    assert "PASS" in result.output


def test_validate_cli_fails_on_a_broken_file(tmp_path, consistent_spatial_photo_result):
    broken = consistent_spatial_photo_result
    del broken.metadata["depth_unit"]
    path = tmp_path / "broken.h5"
    save(path, broken)

    result = CliRunner().invoke(validate_cli, [str(path)])

    assert result.exit_code == 1
    assert "FAIL" in result.output
    assert "depth_unit" in result.output


def test_generate_cli_reports_a_clear_error_for_a_missing_input_file(tmp_path):
    result = CliRunner().invoke(
        generate_cli,
        ["--input", str(tmp_path / "does_not_exist.jpg"), "--output", str(tmp_path / "out.h5")],
    )

    assert result.exit_code != 0
    assert "does_not_exist.jpg" in result.output
