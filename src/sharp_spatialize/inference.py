"""Isolates all SHARP-specific model loading and inference behind `infer()`.

No other module in sharp_spatialize imports `sharp.models` or touches the
predictor directly -- everything downstream only sees the `SceneBundle` this
module returns. This is the `scene = sharp.infer(image)` boundary the spec
calls for.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
from sharp.cli.predict import DEFAULT_MODEL_URL
from sharp.models import PredictorParams, RGBGaussianPredictor, create_predictor
from sharp.utils import io as sharp_io
from sharp.utils.gaussians import Gaussians3D, unproject_gaussians

INTERNAL_SHAPE = (1536, 1536)  # mirrors sharp.cli.predict.predict_image

PRECISIONS = ("fp32", "fp16", "bf16")
_AUTOCAST_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


@dataclass
class SceneBundle:
    """Everything downstream sharp_spatialize modules need about one SHARP scene."""

    gaussians: Gaussians3D
    f_px: float
    width: int
    height: int
    device: str
    model_version: str


def pick_device(requested: str) -> str:
    """Resolve "default" to the best available device; pass through anything else."""
    if requested != "default":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.mps.is_available():
        return "mps"
    return "cpu"


def _load_predictor(
    checkpoint_path: Path | None, device: str
) -> tuple[RGBGaussianPredictor, str]:
    """Load the SHARP predictor and return it with a model-version string."""
    if checkpoint_path is None:
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
        model_version = Path(urlparse(DEFAULT_MODEL_URL).path).name
    else:
        state_dict = torch.load(checkpoint_path, weights_only=True)
        model_version = Path(checkpoint_path).name

    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(device)
    return predictor, model_version


@lru_cache(maxsize=1)
def load_predictor(
    checkpoint_path: Path | None, device: str
) -> tuple[RGBGaussianPredictor, str]:
    """`_load_predictor` memoized on its arguments, for callers that infer repeatedly.

    Loading the checkpoint costs ~12s, so a long-lived process (the web UI, a
    batch job) should not pay it per image. The predictor is in eval mode and
    used under `no_grad`, so sharing one instance across calls is safe. Only
    the most recent (checkpoint, device) pair is kept, since each cached entry
    pins a full model's worth of device memory.

    Call `load_predictor.cache_clear()` to drop it.
    """
    return _load_predictor(checkpoint_path, device)


def _run_forward_pass(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    f_px: float,
    device: str,
    precision: str = "fp32",
) -> Gaussians3D:
    """Run one SHARP forward pass and unproject the result into metric world space.

    `precision` selects the autocast dtype for the forward pass: "fp32" (no
    autocast), "fp16", or "bf16". Autocast applies only on CUDA; on any other
    device the pass runs in fp32 regardless. The predicted Gaussians are cast
    back to fp32 before unprojection, so everything downstream -- the camera
    rig, the rasterizer, the saved file -- sees fp32 either way.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")

    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)
    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(INTERNAL_SHAPE[1], INTERNAL_SHAPE[0]),
        mode="bilinear",
        align_corners=True,
    )
    autocast_dtype = _AUTOCAST_DTYPES.get(precision)
    use_autocast = autocast_dtype is not None and torch.device(device).type == "cuda"
    with torch.no_grad():
        if use_autocast:
            with torch.autocast("cuda", dtype=autocast_dtype):
                gaussians_ndc = predictor(image_resized_pt, disparity_factor)
            gaussians_ndc = Gaussians3D(*(tensor.float() for tensor in gaussians_ndc))
        else:
            gaussians_ndc = predictor(image_resized_pt, disparity_factor)

    intrinsics = (
        torch.tensor(
            [[f_px, 0, width / 2, 0], [0, f_px, height / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= INTERNAL_SHAPE[0] / width
    intrinsics_resized[1] *= INTERNAL_SHAPE[1] / height

    return unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, INTERNAL_SHAPE
    )


def infer(
    image_path: str | Path,
    checkpoint_path: Path | None = None,
    device: str = "default",
    precision: str = "fp32",
    cache_predictor: bool = False,
) -> SceneBundle:
    """Run SHARP on `image_path` and return the resulting scene.

    `precision` is the forward-pass autocast dtype (see `_run_forward_pass`).
    Set `cache_predictor` to reuse a loaded predictor across calls in the same
    process (see `load_predictor`).
    """
    resolved_device = pick_device(device)
    loader = load_predictor if cache_predictor else _load_predictor
    predictor, model_version = loader(checkpoint_path, resolved_device)

    image, _icc_profile, f_px = sharp_io.load_rgb(Path(image_path))
    height, width = image.shape[:2]
    gaussians = _run_forward_pass(predictor, image, f_px, resolved_device, precision)

    return SceneBundle(
        gaussians=gaussians,
        f_px=f_px,
        width=width,
        height=height,
        device=resolved_device,
        model_version=model_version,
    )
