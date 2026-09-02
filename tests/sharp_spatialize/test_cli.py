"""Tests for sharp_spatialize.cli that don't require CUDA or a real model."""

from __future__ import annotations

import subprocess
import sys

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


def test_validate_cli_import_does_not_pull_in_gsplat():
    """Verify that importing validate_cli alone doesn't drag in gsplat/sharp.models.

    This test runs in a subprocess with a clean sys.modules to ensure that
    simply importing validate_cli (e.g., to wire up the console script) doesn't
    pull in CUDA-dependent dependencies like gsplat, sharp.models, or inference.
    This is critical because validate_cli is designed to work on machines with
    no GPU and no SHARP installed.
    """
    code = (
        "import sys; "
        "from sharp_spatialize.cli import validate_cli; "
        "assert 'gsplat' not in sys.modules, "
        "'gsplat should not be imported by validate_cli'; "
        "assert 'sharp.models' not in sys.modules, "
        "'sharp.models should not be imported by validate_cli'; "
        "assert 'sharp_spatialize.inference' not in sys.modules, "
        "'sharp_spatialize.inference should not be imported by validate_cli'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd="/Users/macariofang/sharp",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
