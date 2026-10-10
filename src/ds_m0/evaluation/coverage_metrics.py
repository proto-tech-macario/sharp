from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class CoverageMetrics:
    target_width: int
    target_height: int
    covered_pixel_count: int
    uncovered_pixel_count: int
    coverage_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


def measure_coverage(coverage: np.ndarray) -> CoverageMetrics:
    h, w = coverage.shape
    covered = int(coverage.sum())
    return CoverageMetrics(w, h, covered, w * h - covered, covered / (w * h))
