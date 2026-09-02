"""Tests for sharp_spatialize.validation."""

from __future__ import annotations

import numpy as np
import pytest
from sharp_spatialize.hdf5_io import save
from sharp_spatialize.validation import ValidationError, validate_file, validate_in_memory


def test_validate_in_memory_accepts_a_consistent_synthetic_scene(consistent_spatial_photo_result):
    """validate_in_memory does not raise for a self-consistent synthetic scene."""
    validate_in_memory(consistent_spatial_photo_result)  # must not raise


def test_check_shapes_flags_wrong_rgb_shape(consistent_spatial_photo_result):
    """validate_in_memory flags an rgb array with the wrong number of channels."""
    result = consistent_spatial_photo_result
    result.rgb = result.rgb[..., :2]

    with pytest.raises(ValidationError, match="rgb must have shape"):
        validate_in_memory(result)


def test_check_metadata_flags_missing_keys(consistent_spatial_photo_result):
    """validate_in_memory flags a missing required metadata key."""
    result = consistent_spatial_photo_result
    del result.metadata["depth_unit"]

    with pytest.raises(ValidationError, match="missing required metadata key: depth_unit"):
        validate_in_memory(result)


def test_check_camera_rotations_flags_non_orthonormal_r(consistent_spatial_photo_result):
    """validate_in_memory flags a rotation matrix that isn't orthonormal."""
    result = consistent_spatial_photo_result
    result.R[0] = result.R[0] * 2.0

    with pytest.raises(ValidationError, match=r"R\^T R deviates"):
        validate_in_memory(result)


def test_check_depth_flags_nan(consistent_spatial_photo_result):
    """validate_in_memory flags NaN values in the depth array."""
    result = consistent_spatial_photo_result
    result.depth[0, 0, 0] = np.nan

    with pytest.raises(ValidationError, match="NaN or Inf"):
        validate_in_memory(result)


def test_check_depth_flags_nonzero_depth_at_invalid_mask(consistent_spatial_photo_result):
    """validate_in_memory flags non-zero depth at pixels the mask marks invalid."""
    result = consistent_spatial_photo_result
    invalid_pixels = result.mask[4] == 0
    assert invalid_pixels.any(), "fixture must have at least one invalid pixel in view 4"
    result.depth[4][invalid_pixels] = 3.0

    with pytest.raises(ValidationError, match="non-zero at pixels marked invalid"):
        validate_in_memory(result)


def test_reprojection_fails_when_depth_is_corrupted(consistent_spatial_photo_result):
    """validate_in_memory's reprojection check flags depth that breaks view consistency."""
    result = consistent_spatial_photo_result
    result.depth[3] += 2.0  # break consistency between views 3 and 5

    with pytest.raises(ValidationError, match="reprojection"):
        validate_in_memory(result)


def test_validate_file_reports_all_failures_without_raising(
    tmp_path, consistent_spatial_photo_result
):
    """validate_file collects every failure into one report instead of raising."""
    result = consistent_spatial_photo_result
    del result.metadata["depth_unit"]
    result.R[0] = result.R[0] * 2.0
    path = tmp_path / "broken.h5"
    save(path, result)

    report = validate_file(path)

    assert not report.ok
    assert any("depth_unit" in failure for failure in report.failures)
    assert any("R^T R" in failure for failure in report.failures)


def test_check_shapes_flags_wrong_rgb_dtype(consistent_spatial_photo_result):
    """validate_in_memory flags an rgb array with the wrong dtype."""
    result = consistent_spatial_photo_result
    result.rgb = result.rgb.astype(np.float32)

    with pytest.raises(ValidationError, match="rgb must be dtype uint8"):
        validate_in_memory(result)


def test_check_shapes_flags_wrong_depth_dtype(consistent_spatial_photo_result):
    """validate_in_memory flags a depth array with the wrong dtype."""
    result = consistent_spatial_photo_result
    result.depth = result.depth.astype(np.float64)

    with pytest.raises(ValidationError, match="depth must be dtype float32"):
        validate_in_memory(result)


def test_check_shapes_flags_wrong_mask_dtype(consistent_spatial_photo_result):
    """validate_in_memory flags a mask array with the wrong dtype."""
    result = consistent_spatial_photo_result
    result.mask = result.mask.astype(np.int32)

    with pytest.raises(ValidationError, match="mask must be dtype uint8"):
        validate_in_memory(result)


def test_check_shapes_flags_wrong_k_dtype(consistent_spatial_photo_result):
    """validate_in_memory flags a K (intrinsics) array with the wrong dtype."""
    result = consistent_spatial_photo_result
    result.K = result.K.astype(np.float64)

    with pytest.raises(ValidationError, match="K must be dtype float32"):
        validate_in_memory(result)


def test_check_shapes_flags_mismatched_depth_spatial_dims(consistent_spatial_photo_result):
    """validate_in_memory flags depth whose spatial dims don't match rgb."""
    result = consistent_spatial_photo_result
    result.depth = result.depth[:, :16, :]  # truncate height

    with pytest.raises(ValidationError, match="depth spatial dims"):
        validate_in_memory(result)


def test_check_shapes_flags_mismatched_mask_spatial_dims(consistent_spatial_photo_result):
    """validate_in_memory flags a mask whose spatial dims don't match rgb."""
    result = consistent_spatial_photo_result
    result.mask = result.mask[:, :, :16]  # truncate width

    with pytest.raises(ValidationError, match="mask spatial dims"):
        validate_in_memory(result)


def test_check_shapes_flags_fewer_than_nine_views(consistent_spatial_photo_result):
    """validate_in_memory flags a result with fewer than 9 views."""
    result = consistent_spatial_photo_result
    result.rgb = result.rgb[:4]
    result.depth = result.depth[:4]
    result.mask = result.mask[:4]
    result.K = result.K[:4]
    result.R = result.R[:4]
    result.C = result.C[:4]

    with pytest.raises(ValidationError, match="expected exactly 9 views, got 4"):
        validate_in_memory(result)


def test_check_shapes_flags_metadata_num_views_mismatch(consistent_spatial_photo_result):
    """validate_in_memory flags metadata num_views that disagrees with the actual arrays."""
    result = consistent_spatial_photo_result
    result.metadata["num_views"] = 8

    with pytest.raises(ValidationError, match="metadata num_views=8 does not match"):
        validate_in_memory(result)


def test_check_shapes_flags_metadata_output_width_mismatch(consistent_spatial_photo_result):
    """validate_in_memory flags metadata output_width that disagrees with the actual rgb array."""
    result = consistent_spatial_photo_result
    result.metadata["output_width"] = 999

    with pytest.raises(ValidationError, match="metadata output_width=999 does not match"):
        validate_in_memory(result)


def test_check_shapes_flags_metadata_output_height_mismatch(consistent_spatial_photo_result):
    """validate_in_memory flags metadata output_height that disagrees with the actual rgb array."""
    result = consistent_spatial_photo_result
    result.metadata["output_height"] = 999

    with pytest.raises(ValidationError, match="metadata output_height=999 does not match"):
        validate_in_memory(result)


def test_validate_file_never_raises_on_corrupt_hdf5(tmp_path, consistent_spatial_photo_result):
    """validate_file should return ok=False without raising, even on corrupt HDF5."""
    result = consistent_spatial_photo_result
    path = tmp_path / "corrupt.h5"
    save(path, result)

    # Write some garbage over the HDF5 file to corrupt it
    with open(path, "wb") as f:
        f.write(b"this is not valid HDF5 data")

    # Should not raise, should return a failed report
    report = validate_file(path)
    assert not report.ok
    assert len(report.failures) > 0
    assert any("failed to load or validate" in failure for failure in report.failures)
