"""End-to-end Stage 2 with real SHARP: video -> Stage 1 (CUDA) -> HDF5 -> MIV -> TmivDecoder.

Needs a CUDA GPU (gsplat) and a patched TMIV build; skipped otherwise. Runs
unmodified on the CUDA machine after ./setup.sh and scripts/build_tmiv.sh.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU"),
]

TEASER = Path(__file__).resolve().parents[2] / "data" / "teaser.jpg"
WIDTH, HEIGHT, FRAMES = 320, 240, 3


def _pan_clip(path: Path) -> Path:
    """A short H.264 clip panning across the repo's teaser image (real image content)."""
    image = np.asarray(Image.open(TEASER).convert("RGB").resize((WIDTH * 2, HEIGHT)))
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(30000, 1001))
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        for i in range(FRAMES):
            crop = np.ascontiguousarray(image[:, 8 * i: 8 * i + WIDTH])
            for packet in stream.encode(av.VideoFrame.from_ndarray(crop, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


@pytest.fixture
def tmiv():
    """The patched TMIV build; skips the test if there is none."""
    from sharp_video.miv.tmiv import TmivNotFoundError, find_tmiv

    try:
        return find_tmiv()
    except TmivNotFoundError as exc:
        pytest.skip(str(exc))


def test_video_to_miv_with_real_sharp(tmp_path, tmiv):
    """Video to miv with real sharp."""
    from sharp_video.miv.validate import validate_miv
    from sharp_video.pipeline import PipelineOptions, run_video_to_miv
    from sharp_video.sequence_io import SequenceReader
    from sharp_video.validation import validate_sequence

    clip = _pan_clip(tmp_path / "pan.mp4")
    options = PipelineOptions(
        input=clip, output=tmp_path / "out.miv", work_dir=tmp_path / "work",
        dump_hdf5=tmp_path / "seq.h5", precision="fp16", intra_period=16,
    )
    result = run_video_to_miv(options, tmiv=tmiv)

    sequence = validate_sequence(result.sequence_path)
    assert sequence.ok, sequence.failures
    assert sequence.frame_count == FRAMES

    report = validate_miv(result.miv.path, SequenceReader(result.sequence_path),
                          tmp_path / "dec", tmiv=tmiv)
    assert report.ok, report.failures
    stage1 = result.report["stage1"]
    assert stage1["sharp_and_3dgs_s_per_frame"]["mean"] > 0
    assert stage1["peak_gpu_memory_bytes"] > 0
