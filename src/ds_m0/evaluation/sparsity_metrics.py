from __future__ import annotations

from dataclasses import asdict, dataclass

from ..io.sparse_layer1 import mask_byte_count
from ..model.layer import Layer


@dataclass
class SparsityMetrics:
    width: int
    height: int
    total_pixels: int
    valid_pixels: int
    occupancy: float
    dense_rgb_bytes: int
    dense_depth_bytes: int
    sparse_mask_bytes: int
    sparse_rgb_bytes: int
    sparse_depth_bytes: int
    dense_payload_bytes: int
    sparse_payload_bytes: int
    sparse_savings_bytes: int
    sparse_savings_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


def measure_layer1(layer1: Layer) -> SparsityMetrics:
    """Uncompressed payload sizes: dense = W*H*(3 rgb + 4 depth); sparse = mask + N*(3 + 4)."""
    w, h = layer1.width, layer1.height
    total, n = w * h, layer1.valid_pixel_count()
    dense_rgb, dense_depth = total * 3, total * 4
    mask, rgb, depth = mask_byte_count(w, h), n * 3, n * 4
    dense, sparse = dense_rgb + dense_depth, mask + rgb + depth
    return SparsityMetrics(w, h, total, n, n / total, dense_rgb, dense_depth, mask, rgb, depth,
                           dense, sparse, dense - sparse, (dense - sparse) / dense)
