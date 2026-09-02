"""Tests for sharp_spatialize.validation."""

from __future__ import annotations

import numpy as np
import pytest

from sharp_spatialize.hdf5_io import save
from sharp_spatialize.validation import ValidationError, validate_file, validate_in_memory


def test_validate_in_memory_accepts_a_consistent_synthetic_scene(consistent_spatial_photo_result):
    validate_in_memory(consistent_spatial_photo_result)  # must not raise


def test_check_shapes_flags_wrong_rgb_shape(consistent_spatial_photo_result):
    result = consistent_spatial_photo_result
    result.rgb = result.rgb[..., :2]

    with pytest.raises(ValidationError, match="rgb must have shape"):
        validate_in_memory(result)


def test_check_metadata_flags_missing_keys(consistent_spatial_photo_result):
    result = consistent_spatial_photo_result
    del result.metadata["depth_unit"]

    with pytest.raises(ValidationError, match="missing required metadata key: depth_unit"):
        validate_in_memory(result)


def test_check_camera_rotations_flags_non_orthonormal_r(consistent_spatial_photo_result):
    result = consistent_spatial_photo_result
    result.R[0] = result.R[0] * 2.0

    with pytest.raises(ValidationError, match=r"R\^T R deviates"):
        validate_in_memory(result)


def test_check_depth_flags_nan(consistent_spatial_photo_result):
    result = consistent_spatial_photo_result
    result.depth[0, 0, 0] = np.nan

    with pytest.raises(ValidationError, match="NaN or Inf"):
        validate_in_memory(result)


def test_check_depth_flags_nonzero_depth_at_invalid_mask(consistent_spatial_photo_result):
    result = consistent_spatial_photo_result
    invalid_pixels = result.mask[4] == 0
    assert invalid_pixels.any(), "fixture must have at least one invalid pixel in view 4"
    result.depth[4][invalid_pixels] = 3.0

    with pytest.raises(ValidationError, match="non-zero at pixels marked invalid"):
        validate_in_memory(result)


def test_reprojection_fails_when_depth_is_corrupted(consistent_spatial_photo_result):
    result = consistent_spatial_photo_result
    result.depth[3] += 2.0  # break consistency between views 3 and 5

    with pytest.raises(ValidationError, match="reprojection"):
        validate_in_memory(result)


def test_validate_file_reports_all_failures_without_raising(tmp_path, consistent_spatial_photo_result):
    result = consistent_spatial_photo_result
    del result.metadata["depth_unit"]
    result.R[0] = result.R[0] * 2.0
    path = tmp_path / "broken.h5"
    save(path, result)

    report = validate_file(path)

    assert not report.ok
    assert any("depth_unit" in failure for failure in report.failures)
    assert any("R^T R" in failure for failure in report.failures)
