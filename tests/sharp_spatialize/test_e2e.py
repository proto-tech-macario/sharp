"""End-to-end tests requiring a real CUDA GPU.

Exercises SHARP inference + gsplat rendering + HDF5 packaging + validation,
all for real, using the repo's bundled sample image. Skipped everywhere else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from sharp_spatialize.api import generate_spatial_photo
from sharp_spatialize.cameras import camera_at
from sharp_spatialize.hdf5_io import load
from sharp_spatialize.inference import infer
from sharp_spatialize.render import render_views
from sharp_spatialize.validation import validate_file

SAMPLE_IMAGE = Path(__file__).resolve().parents[2] / "data" / "teaser.jpg"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


def test_generate_spatial_photo_end_to_end(tmp_path):
    """generate_spatial_photo produces a valid, round-trippable spatial photo on real CUDA."""
    result = generate_spatial_photo(SAMPLE_IMAGE, angle_deg=10.0, device="cuda")

    assert result.rgb.shape[0] == 9
    assert result.rgb.dtype == np.uint8
    assert result.depth.shape[0] == 9
    assert len(np.unique(result.C, axis=0)) == 9  # real parallax, not a collapsed rig
    assert not np.array_equal(result.rgb[3], result.rgb[5])  # views actually differ
    assert result.mask.any()  # something rendered as valid

    out_path = tmp_path / "spatial_photo.h5"
    result.save(out_path)

    report = validate_file(out_path)
    assert report.ok, report.failures

    reloaded = load(out_path)
    assert reloaded.rgb.shape == result.rgb.shape


def test_arbitrary_intermediate_camera_angle_renders():
    """Acceptance criterion #10: an arbitrary angle inside the configured range renders."""
    scene = infer(SAMPLE_IMAGE, device="cuda")
    pose = camera_at(
        scene.gaussians.mean_vectors, scene.f_px, scene.width, scene.height,
        h_angle_deg=3.7, v_angle_deg=-4.2, output_width=scene.width, output_height=scene.height,
    )
    rgb, depth, mask = render_views(scene.gaussians, [pose], scene.width, scene.height)

    assert rgb.shape == (1, scene.height, scene.width, 3)
    assert np.isfinite(depth).all()
    assert mask.any()
