"""Stage 1 depth (metres, 0 = invalid) <-> TMIV geometry samples.

TMIV geometry is normalized disparity quantized linearly between
`1/far` (sample 1) and `1/near` (max sample), with sample 0 meaning "no
depth" when the camera has `HasInvalidDepth` (TMIV `DepthOccupancyTransform`:
`normDisp = low + (high - low) * sample / maxSample`, depth = 1/normDisp).
"""

from __future__ import annotations

import numpy as np


def _check_range(near: float, far: float) -> None:
    if not 0.0 < near < far:
        raise ValueError(f"depth range must satisfy 0 < near < far, got near={near}, far={far}")


def depth_to_geometry(
    depth: np.ndarray, mask: np.ndarray, near: float, far: float, bit_depth: int = 16
) -> tuple[np.ndarray, int]:
    """Quantize depth into geometry samples; returns (samples uint16, clamped pixel count).

    Valid depths outside [near, far] are clamped to the range and counted, so
    callers can report how much of the scene fell outside the configured range.
    """
    _check_range(near, far)
    max_sample = (1 << bit_depth) - 1
    valid = (mask == 1) & (depth > 0)
    clamped = int(np.count_nonzero(valid & ((depth < near) | (depth > far))))
    z = np.clip(np.where(valid, depth, far), near, far).astype(np.float64)
    level = (1.0 / z - 1.0 / far) / (1.0 / near - 1.0 / far)
    samples = np.clip(np.rint(level * max_sample), 1, max_sample)
    return np.where(valid, samples, 0).astype(np.uint16), clamped


def geometry_to_depth(
    samples: np.ndarray, near: float, far: float, bit_depth: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `depth_to_geometry`: returns (depth float32 metres, mask uint8).

    `near == far` is allowed here: TMIV's dynamic depth range can shrink a
    decoded view's range to a single depth when all of its content lies at one
    distance, and every valid sample then decodes to that depth.
    """
    if not 0.0 < near <= far:
        raise ValueError(f"depth range must satisfy 0 < near <= far, got near={near}, far={far}")
    max_sample = (1 << bit_depth) - 1
    valid = samples > 0
    level = samples.astype(np.float64) / max_sample
    disparity = 1.0 / far + level * (1.0 / near - 1.0 / far)
    depth = np.where(valid, 1.0 / disparity, 0.0).astype(np.float32)
    return depth, valid.astype(np.uint8)
