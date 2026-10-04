"""Tests for sharp_spatialize.inference.

`_run_forward_pass` is tested with a fake predictor stub so this file needs
no network access, checkpoint download, or GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sharp_spatialize.inference as inference_module
import torch
from PIL import Image
from sharp.cli.predict import DEFAULT_MODEL_URL
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
    """pick_device returns an explicitly requested device unchanged."""
    assert inference_module.pick_device("cpu") == "cpu"


def test_run_forward_pass_returns_finite_gaussians_of_expected_batch_size():
    """_run_forward_pass returns finite Gaussians with the predictor's point count."""
    image = (np.random.default_rng(0).random((64, 96, 3)) * 255).astype(np.uint8)
    gaussians = inference_module._run_forward_pass(
        _FakePredictor(num_points=12), image, 80.0, "cpu"
    )

    assert gaussians.mean_vectors.shape == (1, 12, 3)
    assert torch.isfinite(gaussians.mean_vectors).all()


def test_infer_wires_together_load_predictor_and_forward_pass(tmp_path, monkeypatch):
    """infer() loads the predictor and runs the forward pass to build a SceneBundle."""
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


class _TinyPredictor(torch.nn.Module):
    """A few-parameter stand-in for RGBGaussianPredictor in the loader tests.

    The real model is ~2.8 GB; building it (and saving a copy to a tmpfs /tmp)
    made these tests take minutes and push a WSL2 box into swap. The loader's
    logic doesn't depend on the architecture, and test_e2e loads the real
    checkpoint into the real model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)


def test_load_predictor_checkpoint_path_branch(tmp_path, monkeypatch):
    """_load_predictor loads weights from a checkpoint path and names the model after the file."""
    monkeypatch.setattr(inference_module, "create_predictor", lambda params: _TinyPredictor())
    saved = _TinyPredictor()
    checkpoint_path = tmp_path / "test_model.pt"
    torch.save(saved.state_dict(), checkpoint_path)

    loaded_predictor, model_version = inference_module._load_predictor(checkpoint_path, "cpu")

    assert isinstance(loaded_predictor, _TinyPredictor)
    # A fresh _TinyPredictor has different random weights, so equality proves the file was loaded.
    assert torch.equal(loaded_predictor.linear.weight, saved.linear.weight)
    assert model_version == "test_model.pt"
    assert loaded_predictor.training is False


def test_load_predictor_default_download_branch(monkeypatch):
    """With no checkpoint path, _load_predictor fetches DEFAULT_MODEL_URL and is named after it."""
    monkeypatch.setattr(inference_module, "create_predictor", lambda params: _TinyPredictor())
    saved = _TinyPredictor()
    requested_urls = []

    def fake_load_state_dict_from_url(url, progress=False):  # noqa: ARG001
        requested_urls.append(url)
        return saved.state_dict()

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", fake_load_state_dict_from_url)

    loaded_predictor, model_version = inference_module._load_predictor(None, "cpu")

    assert requested_urls == [DEFAULT_MODEL_URL]
    assert torch.equal(loaded_predictor.linear.weight, saved.linear.weight)
    assert model_version == Path(DEFAULT_MODEL_URL.split("/")[-1]).name
    assert loaded_predictor.training is False
