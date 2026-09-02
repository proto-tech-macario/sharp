"""Tests for sharp_spatialize.api.

inference.infer and render.render_views are monkeypatched so this file needs
no SHARP checkpoint, network, or GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import sharp_spatialize.api as api_module
import torch
from sharp.utils.gaussians import Gaussians3D
from sharp_spatialize.inference import SceneBundle
from sharp_spatialize.validation import ValidationError


def _fake_scene(width: int = 32, height: int = 32) -> SceneBundle:
    xs = torch.linspace(-2.0, 2.0, 10)
    ys = torch.linspace(-2.0, 2.0, 10)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    xy = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    mean_vectors = torch.cat([xy, torch.full((xy.shape[0], 1), 5.0)], dim=-1).unsqueeze(0)
    n = mean_vectors.shape[1]
    gaussians = Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=torch.ones(1, n, 3) * 0.01,
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(1, n, 1),
        colors=torch.rand(1, n, 3),
        opacities=torch.ones(1, n),
    )
    return SceneBundle(
        gaussians=gaussians, f_px=40.0, width=width, height=height,
        device="cpu", model_version="fake-v1",
    )


def _fake_render_views(gaussians, rig, output_width, output_height, mask_alpha_threshold=0.5):
    n = len(rig)
    rgb = np.zeros((n, output_height, output_width, 3), dtype=np.uint8)
    depth = np.full((n, output_height, output_width), 5.0, dtype=np.float32)
    mask = np.ones((n, output_height, output_width), dtype=np.uint8)
    return rgb, depth, mask


def test_generate_spatial_photo_wires_the_pipeline_and_returns_a_valid_result(
    tmp_path, monkeypatch
):
    """generate_spatial_photo wires inference/render/validation into a valid result."""
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"not a real jpeg -- inference.infer is mocked")

    monkeypatch.setattr(api_module.inference, "infer", lambda *a, **k: _fake_scene())
    monkeypatch.setattr(api_module.render, "render_views", _fake_render_views)

    result = api_module.generate_spatial_photo(image_path, angle_deg=10.0)

    assert result.rgb.shape == (9, 32, 32, 3)
    assert result.depth.shape == (9, 32, 32)
    assert result.mask.shape == (9, 32, 32)
    assert result.K.shape == (9, 3, 3)
    assert result.R.shape == (9, 3, 3)
    assert result.C.shape == (9, 3)
    assert result.metadata["num_views"] == 9
    assert result.metadata["horizontal_angle"] == 10.0
    assert result.metadata["vertical_angle"] == 10.0
    assert result.metadata["source_filename"] == "input.jpg"
    assert result.metadata["depth_unit"] == "meter"

    out_path = tmp_path / "spatial_photo.h5"
    result.save(out_path)
    assert out_path.exists()


def test_generate_spatial_photo_raises_when_validation_fails(tmp_path, monkeypatch):
    """generate_spatial_photo raises ValidationError when the rendered result is invalid."""
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"not a real jpeg -- inference.infer is mocked")

    def _broken_render_views(gaussians, rig, output_width, output_height, mask_alpha_threshold=0.5):
        rgb, depth, mask = _fake_render_views(gaussians, rig, output_width, output_height)
        depth[0, 0, 0] = np.nan
        return rgb, depth, mask

    monkeypatch.setattr(api_module.inference, "infer", lambda *a, **k: _fake_scene())
    monkeypatch.setattr(api_module.render, "render_views", _broken_render_views)

    with pytest.raises(ValidationError):
        api_module.generate_spatial_photo(image_path)


def test_generate_spatial_photo_does_not_silently_override_explicit_zero_width_height(
    tmp_path, monkeypatch
):
    """output_width=0/output_height=0 must not silently fall back to scene.width/height.

    `output_width or scene.width` would treat 0 as falsy and silently use
    scene.width instead; the fix uses explicit `is None` checks. This test
    doesn't need a full valid render -- it just confirms the literal 0 values
    reach render.render_views unmodified, by raising as soon as they're
    captured.
    """
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"not a real jpeg -- inference.infer is mocked")
    monkeypatch.setattr(api_module.inference, "infer", lambda *a, **k: _fake_scene())

    captured = {}

    def _capturing_render_views(
        gaussians, rig, output_width, output_height, mask_alpha_threshold=0.5
    ):
        captured["output_width"] = output_width
        captured["output_height"] = output_height
        raise RuntimeError("stop after capturing width/height")

    monkeypatch.setattr(api_module.render, "render_views", _capturing_render_views)

    with pytest.raises(RuntimeError, match="stop after capturing width/height"):
        api_module.generate_spatial_photo(image_path, output_width=0, output_height=0)

    assert captured["output_width"] == 0
    assert captured["output_height"] == 0


def test_load_spatial_photo_round_trips_a_saved_result(tmp_path, monkeypatch):
    """load_spatial_photo reads back exactly what generate_spatial_photo saved."""
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"not a real jpeg -- inference.infer is mocked")
    monkeypatch.setattr(api_module.inference, "infer", lambda *a, **k: _fake_scene())
    monkeypatch.setattr(api_module.render, "render_views", _fake_render_views)

    result = api_module.generate_spatial_photo(image_path)
    out_path = tmp_path / "spatial_photo.h5"
    result.save(out_path)

    reloaded = api_module.load_spatial_photo(out_path)
    np.testing.assert_array_equal(reloaded.rgb, result.rgb)
