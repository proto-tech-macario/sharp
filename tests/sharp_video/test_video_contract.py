"""Tests for sharp_video.contract."""

from __future__ import annotations

import numpy as np
import pytest
from sharp_video.contract import SequenceInfo, SpatialFrame
from synthetic import make_frames, make_stage1_result


def test_from_stage1_splits_into_nine_views():
    """From stage1 splits into nine views."""
    result = make_stage1_result()
    frame = SpatialFrame.from_stage1(result, timestamp=0.5, pts=7, source_frame_index=3)

    assert len(frame.views) == 9
    assert frame.timestamp == 0.5 and frame.pts == 7 and frame.source_frame_index == 3
    assert (frame.width, frame.height) == (32, 32)
    for v, view in enumerate(frame.views):
        np.testing.assert_array_equal(view.rgb, result.rgb[v])
        np.testing.assert_array_equal(view.depth, result.depth[v])
        np.testing.assert_array_equal(view.mask, result.mask[v])
        np.testing.assert_array_equal(view.K, result.K[v])
        np.testing.assert_array_equal(view.R, result.R[v])
        np.testing.assert_array_equal(view.C, result.C[v])
    assert frame.metadata == result.metadata


def test_to_stage1_round_trips():
    """To stage1 round trips."""
    result = make_stage1_result()
    back = SpatialFrame.from_stage1(result, timestamp=0.0).to_stage1()
    for name in ("rgb", "depth", "mask", "K", "R", "C"):
        np.testing.assert_array_equal(getattr(back, name), getattr(result, name))
        assert getattr(back, name).dtype == getattr(result, name).dtype
    assert back.metadata == result.metadata


def test_stacked_shapes():
    """Stacked shapes."""
    stacked = make_frames(1)[0].stacked()
    assert stacked["rgb"].shape == (9, 32, 32, 3)
    assert stacked["depth"].shape == (9, 32, 32)
    assert stacked["K"].shape == (9, 3, 3)
    assert stacked["C"].shape == (9, 3)


def test_drift_moves_outer_cameras_but_not_center():
    """Drift moves outer cameras but not center."""
    frames = make_frames(2, drift=0.5)
    c0, c1 = frames[0].stacked()["C"], frames[1].stacked()["C"]
    np.testing.assert_allclose(c0[4], c1[4])  # V4 is the reference camera
    assert np.abs(c0[0] - c1[0]).max() > 1e-3


def test_sequence_info_attrs_have_spec_fields():
    """Sequence info attrs have spec fields."""
    attrs = SequenceInfo(width=64, height=48, fps=25.0).to_attrs()
    for key in ("width", "height", "fps", "frame_count", "view_count", "depth_unit",
                "coordinate_system", "camera_convention", "depth_definition",
                "invalid_depth_value", "format_version", "view_layout"):
        assert key in attrs
    assert attrs["view_count"] == 9
    assert attrs["depth_unit"] == "meter"
    assert attrs["coordinate_system"] == "OpenCV"


def test_view_count_other_than_nine_rejected():
    """View count other than nine rejected."""
    frame = make_frames(1)[0]
    with pytest.raises(ValueError, match="9 views"):
        SpatialFrame(timestamp=0.0, views=frame.views[:8])
