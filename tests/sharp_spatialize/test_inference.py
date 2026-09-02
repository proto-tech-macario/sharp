"""Tests for sharp_spatialize.inference.

`_run_forward_pass` is tested with a fake predictor stub so this file needs
no network access, checkpoint download, or GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sharp_spatialize.inference as inference_module
from sharp.cli.predict import DEFAULT_MODEL_URL
from sharp.models import PredictorParams, create_predictor
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


def test_load_predictor_checkpoint_path_branch(tmp_path):
    """Test the checkpoint-path branch of _load_predictor."""
    # Build a real predictor and save its state dict
    predictor_fresh = create_predictor(PredictorParams())
    state_dict = predictor_fresh.state_dict()

    checkpoint_path = tmp_path / "test_model.pt"
    torch.save(state_dict, checkpoint_path)

    # Load via the checkpoint path
    loaded_predictor, model_version = inference_module._load_predictor(checkpoint_path, "cpu")

    # Verify it's the right type
    from sharp.models import RGBGaussianPredictor
    assert isinstance(loaded_predictor, RGBGaussianPredictor)

    # Verify model_version is extracted correctly from checkpoint filename
    assert model_version == "test_model.pt"

    # Verify it's in eval mode
    assert loaded_predictor.training is False


def test_load_predictor_default_download_branch(monkeypatch):
    """Test the default-download branch of _load_predictor."""
    # Build a fresh predictor's state dict to use as the mock return value
    predictor_fresh = create_predictor(PredictorParams())
    state_dict = predictor_fresh.state_dict()

    # Track if the mock was called
    call_tracker = {"called": False}

    def mock_load_state_dict_from_url(url, progress=False):  # noqa: ARG001
        call_tracker["called"] = True
        return state_dict

    # Monkeypatch torch.hub.load_state_dict_from_url
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", mock_load_state_dict_from_url)

    # Load via the default path (checkpoint_path=None)
    loaded_predictor, model_version = inference_module._load_predictor(None, "cpu")

    # Verify the mock was called
    assert call_tracker["called"]

    # Verify model_version is extracted correctly from DEFAULT_MODEL_URL
    # DEFAULT_MODEL_URL is something like "https://...../sharp_2572gikvuh.pt"
    expected_filename = Path(DEFAULT_MODEL_URL.split("/")[-1])
    assert model_version == expected_filename.name

    # Verify it's the right type and in eval mode
    from sharp.models import RGBGaussianPredictor
    assert isinstance(loaded_predictor, RGBGaussianPredictor)
    assert loaded_predictor.training is False
