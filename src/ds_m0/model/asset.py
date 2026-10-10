"""The logical DS Asset: the central object every stage communicates through."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import ValidationError
from ..geometry import camera_math
from .camera import Camera
from .layer import Layer

ASSET_VERSION = "1.0"


@dataclass
class DSAsset:
    version: str
    spatial_width: int
    spatial_height: int
    base_camera: Camera
    layers: list[Layer]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    def width(self) -> int:
        return self.spatial_width

    def height(self) -> int:
        return self.spatial_height

    def layer(self, index: int) -> Layer:
        return self.layers[index]

    def validate(self, check_hidden: bool = False) -> None:
        """Check the M0 invariants. `check_hidden` also requires Layer1 to be
        strictly behind Layer0 (only meaningful before depth quantisation)."""
        if self.layer_count != 2:
            raise ValidationError(f"M0 requires layer_count == 2, got {self.layer_count}")
        camera_math.validate(self.base_camera)
        if (self.base_camera.width, self.base_camera.height) != (self.spatial_width, self.spatial_height):
            raise ValidationError("base camera dimensions differ from the spatial resolution")
        for i, layer in enumerate(self.layers):
            if (layer.width, layer.height) != (self.spatial_width, self.spatial_height):
                raise ValidationError(
                    f"layer {i} is {layer.width}x{layer.height}, expected "
                    f"{self.spatial_width}x{self.spatial_height}"
                )
            layer.validate()
        l0, l1 = self.layers
        if l0.valid_pixel_count() == 0:
            raise ValidationError("Layer 0 has no valid samples")
        if check_hidden:
            both = l1.valid & l0.valid
            if (l1.valid & ~l0.valid).any():
                raise ValidationError("Layer 1 sample exists where Layer 0 is invalid")
            if not (l1.depth[both] > l0.depth[both]).all():
                raise ValidationError("Layer 1 depth is not behind Layer 0")

    def content_hash(self) -> str:
        """SHA-256 over the logical content (invalid samples ignored)."""
        h = hashlib.sha256()
        h.update(f"{self.version}|{self.spatial_width}|{self.spatial_height}".encode())
        h.update(repr(self.base_camera.to_dict()).encode())
        for layer in self.layers:
            c = layer.canonical()
            h.update(np.ascontiguousarray(c.valid).tobytes())
            h.update(np.ascontiguousarray(c.rgb).tobytes())
            h.update(np.ascontiguousarray(c.depth).tobytes())
        return h.hexdigest()
