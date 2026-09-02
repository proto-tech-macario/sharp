"""Tests for sharp_spatialize.hdf5_io."""

from __future__ import annotations

import numpy as np
from sharp_spatialize.hdf5_io import SpatialPhotoResult, load, save


def _make_result(num_views: int = 9, height: int = 4, width: int = 6) -> SpatialPhotoResult:
    rng = np.random.default_rng(0)
    return SpatialPhotoResult(
        rgb=rng.integers(0, 255, size=(num_views, height, width, 3), dtype=np.uint8),
        depth=(rng.random((num_views, height, width)).astype(np.float32) + 0.1),
        mask=np.ones((num_views, height, width), dtype=np.uint8),
        K=np.tile(np.eye(3, dtype=np.float32), (num_views, 1, 1)),
        R=np.tile(np.eye(3, dtype=np.float32), (num_views, 1, 1)),
        C=np.zeros((num_views, 3), dtype=np.float32),
        metadata={"format_version": "1.0", "model_name": "SHARP", "num_views": num_views},
    )


def test_save_then_load_round_trips_all_datasets(tmp_path):
    """save() then load() reproduces every dataset and metadata field exactly."""
    result = _make_result()
    out_path = tmp_path / "spatial_photo.h5"

    save(out_path, result)
    loaded = load(out_path)

    np.testing.assert_array_equal(loaded.rgb, result.rgb)
    np.testing.assert_array_equal(loaded.depth, result.depth)
    np.testing.assert_array_equal(loaded.mask, result.mask)
    np.testing.assert_array_equal(loaded.K, result.K)
    np.testing.assert_array_equal(loaded.R, result.R)
    np.testing.assert_array_equal(loaded.C, result.C)
    assert loaded.metadata["format_version"] == "1.0"
    assert loaded.metadata["num_views"] == 9


def test_result_save_method_matches_module_level_save(tmp_path):
    """SpatialPhotoResult.save() behaves the same as the module-level save() function."""
    result = _make_result()
    out_path = tmp_path / "spatial_photo.h5"

    result.save(out_path)

    assert out_path.exists()
    loaded = load(out_path)
    np.testing.assert_array_equal(loaded.rgb, result.rgb)


def test_save_does_not_leave_a_tmp_file_behind(tmp_path):
    """save() cleans up its temporary file, leaving only the final output path."""
    result = _make_result()
    out_path = tmp_path / "spatial_photo.h5"

    save(out_path, result)

    assert list(tmp_path.iterdir()) == [out_path]
