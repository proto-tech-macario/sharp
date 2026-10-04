"""Deterministic Layer-1 candidate merge (spec Part II section 14)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import visibility
from ..model.layer import Layer
from .candidate_generator import Layer1Candidates


@dataclass
class MergeConfig:
    depth_epsilon: float = 1e-5


def merge_layer1_candidates(
    candidates: Layer1Candidates, width: int, height: int, config: MergeConfig
) -> Layer:
    """At most one sample per base pixel: nearest hidden depth, then smaller source-view
    angle, then lower view id (then source index, only to make ties total)."""
    layer = Layer.empty(width, height)
    if len(candidates) == 0:
        return layer
    ok = np.isfinite(candidates.depth_base) & (candidates.depth_base > 0)
    sel = np.flatnonzero(ok)
    group = candidates.target_y[sel] * width + candidates.target_x[sel]
    winners = sel[
        visibility.select_winners(
            group,
            candidates.depth_base[sel],
            [candidates.source_view_angle[sel], candidates.source_view_id[sel], candidates.source_index[sel]],
            config.depth_epsilon,
        )
    ]
    x, y = candidates.target_x[winners], candidates.target_y[winners]
    layer.valid[y, x] = True
    layer.rgb[y, x] = candidates.rgb[winners]
    layer.depth[y, x] = candidates.depth_base[winners].astype(np.float32)
    return layer
