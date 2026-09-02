"""Tests for sharp_spatialize.inference.

`_run_forward_pass` is tested with a fake predictor stub so this file needs
no network access, checkpoint download, or GPU.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

import sharp_spatialize.inference as inference_module
from sharp.utils.gaussians import Gaussians3D


class _FakePredictor(torch.nn.Module):
    """Stands in for the real SHARP predictor: returns a fixed-size Gaussians3D."""

    def __init__(self, num_points: int = 10) -> None:
        super().__init__()
        self.num_points = num_points

    def forward(self, image, disparity_factor):  # noqa: ARG002 - matches predictor signature
        n = self.num_points
        mean_vectors = torch.rand(1, n, 3) * 0.5 + 0.25
        return Gaussians3D(
            mean_vectors=mean_vectors,
            singular_values=torch.ones(1, n, 3) * 0.01,
            quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(1, n, 1),
            colors=torch.rand(1, n, 3),
            opacities=torch.ones(1, n) * 0.9,
        )


def test_pick_device_passes_through_an_explicit_choice():
    assert inference_module.pick_device("cpu") == "cpu"


def test_run_forward_pass_returns_finite_gaussians_of_expected_batch_size():
    image = (np.random.default_rng(0).random((64, 96, 3)) * 255).astype(np.uint8)
    gaussians = inference_module._run_forward_pass(_FakePredictor(num_points=12), image, 80.0, "cpu")

    assert gaussians.mean_vectors.shape == (1, 12, 3)
    assert torch.isfinite(gaussians.mean_vectors).all()


def test_infer_wires_together_load_predictor_and_forward_pass(tmp_path, monkeypatch):
    image_path = tmp_path / "input.png"
    Image.fromarray(
        (np.random.default_rng(1).random((40, 50, 3)) * 255).astype(np.uint8)
    ).save(image_path)

    monkeypatch.setattr(
        inference_module, "_load_predictor",
        lambda checkpoint_path, device: (_FakePredictor(num_points=8), "fake-model-v1"),
    )

    scene = inference_module.infer(image_path, device="cpu")

    assert scene.width == 50
    assert scene.height == 40
    assert scene.device == "cpu"
    assert scene.model_version == "fake-model-v1"
    assert scene.gaussians.mean_vectors.shape[0] == 1
    assert torch.isfinite(scene.gaussians.mean_vectors).all()
