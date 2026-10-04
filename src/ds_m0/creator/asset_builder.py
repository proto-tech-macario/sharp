"""DS Asset Builder: SourceDataset -> logical DS Asset (no file-format knowledge)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .. import GENERATOR_VERSION
from ..config.m0_config import BuilderConfig
from ..errors import InputError
from ..geometry import camera_math
from ..model.asset import ASSET_VERSION, DSAsset
from ..model.layer import Layer
from ..model.source_dataset import SourceDataset, SourceView
from .candidate_generator import CandidateGenerationConfig, Layer1Candidates, generate_candidates
from .candidate_merger import MergeConfig, merge_layer1_candidates

log = logging.getLogger("ds_m0.builder")


@dataclass
class BuildDiagnostics:
    number_of_views: int = 0
    base_view_id: int = 0
    width: int = 0
    height: int = 0
    layer0_valid_pixel_count: int = 0
    layer1_valid_pixel_count: int = 0
    layer1_occupancy: float = 0.0
    candidate_count: int = 0
    candidates_per_view: dict[int, int] = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)


def build_layer0(base_view: SourceView) -> Layer:
    """Layer0 = base view, unmodified (valid additionally requires finite positive depth)."""
    valid = base_view.depth_valid & camera_math.valid_depth(base_view.depth)
    return Layer(
        base_view.width, base_view.height,
        base_view.rgb.copy(), base_view.depth.astype(np.float32, copy=True), valid.copy(),
    )


def build_ds_asset_with_diagnostics(
    source: SourceDataset, config: BuilderConfig, expected_view_count: int | None = 9
) -> tuple[DSAsset, BuildDiagnostics]:
    config.validate()
    source.validate(expected_view_count)
    t0 = time.perf_counter()
    base_view = source.base_view(config.base_view_id)
    log.info("base view %d selected (%dx%d)", base_view.view_id, source.width, source.height)
    layer0 = build_layer0(base_view)
    if not layer0.valid.any():
        raise InputError("base view depth contains no usable samples")
    base_camera = base_view.camera
    t1 = time.perf_counter()

    gen_cfg = CandidateGenerationConfig(
        config.relative_depth_threshold, config.absolute_depth_threshold, config.rounding_mode
    )
    diag = BuildDiagnostics(
        number_of_views=source.view_count(), base_view_id=base_view.view_id,
        width=source.width, height=source.height,
    )
    batches: list[Layer1Candidates] = []
    for view in sorted(source.views, key=lambda v: v.view_id):  # order is irrelevant to the result
        if view.view_id == base_view.view_id:
            continue
        c = generate_candidates(view, layer0, base_camera, gen_cfg)
        diag.candidates_per_view[view.view_id] = len(c)
        batches.append(c)
    all_candidates = Layer1Candidates.concat(batches)
    diag.candidate_count = len(all_candidates)
    log.info("candidate generation: %d candidates", diag.candidate_count)
    t2 = time.perf_counter()

    layer1 = merge_layer1_candidates(
        all_candidates, source.width, source.height, MergeConfig(config.depth_epsilon)
    )
    t3 = time.perf_counter()

    asset = DSAsset(
        version=ASSET_VERSION,
        spatial_width=source.width,
        spatial_height=source.height,
        base_camera=base_camera,
        layers=[layer0, layer1],
        metadata={
            "generator_version": GENERATOR_VERSION,
            "base_view_id": base_view.view_id,
            "depth_format": "float32",
            "depth_scale": 1.0,
            "depth_offset": 0.0,
            "depth_unit": source.depth_unit,
            "hidden_depth_threshold": {
                "relative": config.relative_depth_threshold,
                "absolute": config.absolute_depth_threshold,
            },
            "candidate_merge_threshold": config.depth_epsilon,
            "rounding_mode": config.rounding_mode,
        },
    )
    asset.validate(check_hidden=True)
    diag.layer0_valid_pixel_count = layer0.valid_pixel_count()
    diag.layer1_valid_pixel_count = layer1.valid_pixel_count()
    diag.layer1_occupancy = layer1.occupancy()
    diag.timings_s = {
        "layer0_build": t1 - t0, "layer1_candidate_generation": t2 - t1,
        "layer1_merge": t3 - t2, "asset_build_total": time.perf_counter() - t0,
    }
    log.info("layer1 valid pixels %d (occupancy %.4f)", diag.layer1_valid_pixel_count, diag.layer1_occupancy)
    return asset, diag


def build_ds_asset(source: SourceDataset, config: BuilderConfig, expected_view_count: int | None = 9) -> DSAsset:
    return build_ds_asset_with_diagnostics(source, config, expected_view_count)[0]
