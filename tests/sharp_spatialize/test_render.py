"""Tests for sharp_spatialize.render.

Only the "no CUDA" guard clause is meaningfully testable on a machine without
a GPU; the real rendering path is skip-marked and runs wherever CUDA is
available (see also tests/sharp_spatialize/test_e2e.py).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sharp.utils.gaussians import Gaussians3D
from sharp_spatialize.cameras import CameraPose
from sharp_spatialize.render import render_views


def _dummy_gaussians() -> Gaussians3D:
    # Placed at z=2.0 (positive, in front of the identity-pose camera at the
    # origin looking down +Z) -- a Gaussian at the camera's own position
    # (z=0) would be culled by gsplat's depth<=1e-2 cutoff, producing an
    # empty (all-zero, alpha=0) frame that would pass shape/dtype checks
    # even for a completely broken renderer.
    return Gaussians3D(
        mean_vectors=torch.tensor([[[0.0, 0.0, 2.0]]]),
        singular_values=torch.ones(1, 1, 3),
        quaternions=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        colors=torch.zeros(1, 1, 3),
        opacities=torch.ones(1, 1),
    )


def _dummy_pose() -> CameraPose:
    # A real intrinsic centered on the 8x8 test image, not np.eye(3) (fx=fy=1,
    # cx=cy=0): with cx=cy=0 the Gaussian's projected peak lands on the image
    # corner where no pixel center samples it, so no pixel would ever clear
    # the alpha threshold and `mask.any()` would fail even for a correct
    # renderer. fx=fy=8 keeps the projected footprint comfortably within a
    # few pixels of center for the z=2.0 Gaussian in _dummy_gaussians().
    return CameraPose(
        K=np.array([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        R=np.eye(3, dtype=np.float32),
        C=np.zeros(3, dtype=np.float32),
    )


@pytest.mark.skipif(torch.cuda.is_available(), reason="only meaningful without a CUDA GPU")
def test_render_views_raises_a_clear_error_without_cuda():
    """render_views raises a clear RuntimeError when no CUDA GPU is available."""
    with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
        render_views(_dummy_gaussians(), [_dummy_pose()], output_width=8, output_height=8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_render_views_produces_expected_shapes_on_cuda():
    """render_views produces correctly shaped and typed rgb/depth/mask arrays on CUDA."""
    rgb, depth, mask = render_views(
        _dummy_gaussians(), [_dummy_pose(), _dummy_pose()], output_width=8, output_height=8
    )
    assert rgb.shape == (2, 8, 8, 3)
    assert rgb.dtype == np.uint8
    assert depth.shape == (2, 8, 8)
    assert depth.dtype == np.float32
    assert mask.shape == (2, 8, 8)
    assert mask.dtype == np.uint8
    assert np.isfinite(depth).all()
    assert mask.any()  # something actually rendered as valid, not an empty frame
    assert set(np.unique(mask)).issubset({0, 1})  # mask is strictly binary
    assert np.all(depth[mask == 0] == 0.0)  # invalid depth is exactly zero
