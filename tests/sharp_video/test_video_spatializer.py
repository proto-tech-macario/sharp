"""Tests for sharp_video.spatializer (per-frame Stage 1 invocation)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image
from synthetic import make_stage1_result

from sharp_video.spatializer import CameraConfig, Stage1Spatializer, load_camera_config
from sharp_video.video_io import DecodedFrame


def _decoded(index=3, width=32, height=32):
    rng = np.random.default_rng(index)
    rgb = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return DecodedFrame(index=index, pts=index * 1001, timestamp=index * 1001 / 30000, rgb=rgb)


class FakeGenerate:
    """Stands in for sharp_spatialize.api.generate_spatial_photo."""

    def __init__(self):
        self.calls = []
        self.png_pixels = None

    def __call__(self, image_path, **kwargs):
        self.calls.append((image_path, kwargs))
        self.png_pixels = np.asarray(Image.open(image_path).convert("RGB"))
        report = kwargs["on_progress"]
        for stage, fraction in (("inference", 0.05), ("rendering", 0.55),
                                ("validating", 0.9), ("done", 1.0)):
            report(stage, fraction)
        return make_stage1_result()


def test_frame_goes_through_stage1_with_forwarded_options(tmp_path):
    fake = FakeGenerate()
    spatializer = Stage1Spatializer(
        work_dir=tmp_path, camera=CameraConfig(angle_deg=12.5), output_size=(32, 32),
        checkpoint_path=None, device="cuda", precision="fp16", generate=fake,
    )
    decoded = _decoded()
    frame, timings = spatializer(decoded)

    (image_path, kwargs), = fake.calls
    assert kwargs["angle_deg"] == 12.5
    assert kwargs["output_width"] == 32 and kwargs["output_height"] == 32
    assert kwargs["device"] == "cuda" and kwargs["precision"] == "fp16"
    assert kwargs["cache_predictor"] is True
    np.testing.assert_array_equal(fake.png_pixels, decoded.rgb)  # lossless hand-off
    assert not image_path.exists()  # temporary PNG removed

    assert frame.timestamp == decoded.timestamp
    assert frame.pts == decoded.pts and frame.source_frame_index == decoded.index
    assert len(frame.views) == 9
    for value in (timings.sharp_and_3dgs_s, timings.render_s, timings.validate_s):
        assert value >= 0.0
    assert timings.total_s >= timings.sharp_and_3dgs_s + timings.render_s


def test_default_output_size_is_source(tmp_path):
    fake = FakeGenerate()
    Stage1Spatializer(work_dir=tmp_path, generate=fake)(_decoded())
    kwargs = fake.calls[0][1]
    assert kwargs["output_width"] is None and kwargs["output_height"] is None
    assert kwargs["angle_deg"] == 10.0


def test_png_removed_even_when_stage1_fails(tmp_path):
    def failing(image_path, **kwargs):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        Stage1Spatializer(work_dir=tmp_path, generate=failing)(_decoded())
    assert not list(tmp_path.rglob("*.png"))


def test_load_camera_config(tmp_path):
    path = tmp_path / "camera.json"
    path.write_text(json.dumps({"angle_deg": 15}))
    assert load_camera_config(path) == CameraConfig(angle_deg=15.0)

    path.write_text(json.dumps({"angle_deg": 15, "fov": 3}))
    with pytest.raises(ValueError, match="fov"):
        load_camera_config(path)

    path.write_text(json.dumps({"angle_deg": 0}))
    with pytest.raises(ValueError, match="positive"):
        load_camera_config(path)
