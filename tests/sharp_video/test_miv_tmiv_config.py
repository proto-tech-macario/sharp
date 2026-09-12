"""Tests for sharp_video.miv.tmiv_config (TMIV sequence/encoder/mux/decoder configs)."""

from __future__ import annotations

import csv
import io
import json

import pytest
from sharp_video.miv.camera import tmiv_view_json
from sharp_video.miv.tmiv_config import (
    CONTENT_ID,
    RATE_ID,
    VIEW_NAMES,
    atlas_budget,
    decoder_config,
    encoder_config,
    multiplexer_config,
    qp_csv,
    sequence_config,
)
from synthetic import make_frames


def _cameras():
    frame = make_frames(1)[0]
    return [
        tmiv_view_json(name, v.K, v.R, v.C, 32, 32, (0.1, 1000.0))
        for name, v in zip(VIEW_NAMES, frame.views)
    ]


def test_view_names():
    """View names."""
    assert VIEW_NAMES == tuple(f"v{i}" for i in range(9))


def test_sequence_config_has_required_fields():
    """Sequence config has required fields."""
    seq = sequence_config(_cameras(), fps=29.97, frame_count=12)
    assert seq["Version"] == "4.0"
    assert seq["Fps"] == 29.97 and seq["Frames_number"] == 12
    assert seq["BoundingBox_center"] == [0.0, 0.0, 0.0]
    assert seq["sourceCameraNames"] == list(VIEW_NAMES)
    assert [c["Name"] for c in seq["cameras"]] == list(VIEW_NAMES)
    json.dumps(seq)


def test_encoder_config_codes_all_views_with_per_frame_cameras():
    """Encoder config codes all views with per frame cameras."""
    cfg = encoder_config(width=32, height=32, fps=30.0, intra_period=8)
    assert cfg["PrunerMethod"] == "NoPruner" and "NoPruner" in cfg
    assert cfg["ViewOptimizerMethod"] == "NoViewOptimizer" and "NoViewOptimizer" in cfg
    assert cfg["interPeriod"] == 1  # one common atlas frame (camera update) per frame
    assert cfg["intraPeriod"] == 8
    assert "{3" in cfg["inputSequenceConfigPathFmt"]  # per-frame camera files
    assert cfg["codecGroupIdc"] == "VVC Main10"
    assert cfg["haveTextureVideo"] and cfg["haveGeometryVideo"]
    assert cfg["bitDepthTextureVideo"] == 10
    for key in ("inputTexturePathFmt", "inputGeometryPathFmt", "outputBitstreamPathFmt",
                "outputTextureVideoDataPathFmt", "outputGeometryVideoDataPathFmt"):
        assert key in cfg
    json.dumps(cfg)


def test_multiplexer_reads_what_encoder_writes():
    """Multiplexer reads what encoder writes."""
    enc, mux = encoder_config(32, 32, 30.0), multiplexer_config()
    assert mux["inputBitstreamPathFmt"] == enc["outputBitstreamPathFmt"]
    assert mux["outputBitstreamPathFmt"].format(4, CONTENT_ID, RATE_ID).endswith(".bit")


def test_decoder_config_outputs_views_and_cameras(tmp_path):
    """Decoder config outputs views and cameras."""
    cfg = decoder_config(tmp_path / "out.miv")
    assert cfg["inputBitstreamPathFmt"] == str(tmp_path / "out.miv")
    for key in ("outputMultiviewTexturePathFmt", "outputMultiviewGeometryPathFmt",
                "outputSequenceConfigPathFmt"):
        assert key in cfg


def test_qp_csv_matches_tmiv_rate_table_format():
    """Qp csv matches tmiv rate table format."""
    rows = list(csv.DictReader(io.StringIO(qp_csv(qp_texture=22, qp_geometry=8))))
    by_component = {row["component_id"]: row for row in rows}
    assert by_component["tex"][RATE_ID] == "22"
    assert by_component["geo"][RATE_ID] == "8"
    assert list(rows[0].keys()) == ["component_id", "rates", RATE_ID]


@pytest.mark.parametrize("width,height", [(32, 32), (1280, 720), (1920, 1080)])
def test_atlas_budget_covers_nine_full_views(width, height):
    """Atlas budget covers nine full views."""
    max_atlases, picture_size, sample_rate = atlas_budget(width, height, fps=30.0)
    assert max_atlases * picture_size >= 9 * width * height
    # texture + full-resolution geometry, every frame
    assert sample_rate >= 2 * 9 * width * height * 30.0
    assert picture_size <= 8912896
