"""Logical DS layer: dense RGB + depth + validity, all the same W x H."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import ValidationError


@dataclass
class Layer:
    width: int
    height: int
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32, camera-space Z in base-camera coordinates
    valid: np.ndarray  # (H, W) bool; authoritative

    @staticmethod
    def empty(width: int, height: int) -> "Layer":
        return Layer(
            width,
            height,
            np.zeros((height, width, 3), np.uint8),
            np.zeros((height, width), np.float32),
            np.zeros((height, width), bool),
        )

    def validate(self) -> None:
        h, w = self.height, self.width
        if w <= 0 or h <= 0:
            raise ValidationError(f"layer dimensions must be positive, got {w}x{h}")
        if self.rgb.shape != (h, w, 3) or self.rgb.dtype != np.uint8:
            raise ValidationError(f"layer rgb must be uint8 ({h},{w},3), got {self.rgb.dtype} {self.rgb.shape}")
        if self.depth.shape != (h, w) or self.depth.dtype != np.float32:
            raise ValidationError(f"layer depth must be float32 ({h},{w}), got {self.depth.dtype} {self.depth.shape}")
        if self.valid.shape != (h, w) or self.valid.dtype != bool:
            raise ValidationError(f"layer valid must be bool ({h},{w}), got {self.valid.dtype} {self.valid.shape}")
        d = self.depth[self.valid]
        if not (np.isfinite(d).all() and (d > 0).all()):
            raise ValidationError("valid layer samples must have finite, positive depth")

    def valid_pixel_count(self) -> int:
        return int(self.valid.sum())

    def occupancy(self) -> float:
        return self.valid_pixel_count() / float(self.width * self.height)

    def canonical(self) -> "Layer":
        """Copy with unspecified (invalid) rgb/depth zeroed; used for hashing/comparison."""
        rgb = np.where(self.valid[..., None], self.rgb, 0).astype(np.uint8)
        depth = np.where(self.valid, self.depth, 0).astype(np.float32)
        return Layer(self.width, self.height, rgb, depth, self.valid.copy())
