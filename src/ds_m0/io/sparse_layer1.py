"""Sparse Layer-1 physical representation: 1-bit mask + packed RGB + packed depth streams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import FormatError
from ..model.layer import Layer


@dataclass
class SparseLayer1Payload:
    mask: bytes  # np.packbits (big-endian bit order), row-major, ceil(W*H/8) bytes
    rgb_stream: np.ndarray  # (N, 3) uint8, raster order of valid pixels
    depth_stream: np.ndarray  # (N,) float32, same order

    @property
    def valid_count(self) -> int:
        return int(self.rgb_stream.shape[0])


@dataclass
class SparseLayer1Config:
    pass


def mask_byte_count(width: int, height: int) -> int:
    return (width * height + 7) // 8


def pack_mask(valid: np.ndarray) -> bytes:
    return np.packbits(valid.reshape(-1).astype(np.uint8)).tobytes()


def unpack_mask(mask: bytes, width: int, height: int) -> np.ndarray:
    expected = mask_byte_count(width, height)
    if len(mask) != expected:
        raise FormatError(f"mask is {len(mask)} bytes, expected {expected} for {width}x{height}")
    bits = np.unpackbits(np.frombuffer(mask, np.uint8))
    if bits[width * height:].any():
        raise FormatError("mask padding bits are not zero")
    return bits[: width * height].astype(bool).reshape(height, width)


def encode_layer1(layer: Layer, config: SparseLayer1Config | None = None) -> SparseLayer1Payload:
    valid = layer.valid
    return SparseLayer1Payload(
        mask=pack_mask(valid),
        rgb_stream=np.ascontiguousarray(layer.rgb[valid]),  # boolean indexing is row-major raster order
        depth_stream=np.ascontiguousarray(layer.depth[valid]),
    )


def decode_layer1(payload: SparseLayer1Payload, width: int, height: int) -> Layer:
    valid = unpack_mask(payload.mask, width, height)
    n = int(valid.sum())
    if payload.rgb_stream.shape != (n, 3):
        raise FormatError(f"Layer 1 RGB stream has {payload.rgb_stream.shape[0]} samples, mask has {n}")
    if payload.depth_stream.shape != (n,):
        raise FormatError(f"Layer 1 depth stream has {payload.depth_stream.shape[0]} samples, mask has {n}")
    layer = Layer.empty(width, height)
    layer.valid[:] = valid
    layer.rgb[valid] = payload.rgb_stream
    layer.depth[valid] = payload.depth_stream
    return layer
