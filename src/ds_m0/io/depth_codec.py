"""Depth codecs. Selected by name; parameters are recorded in the DS metadata."""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import ConfigurationError, FormatError
from ._util import inflate


@dataclass
class DepthCodecConfig:
    name: str = "float32_zlib"
    zlib_level: int = 9


@dataclass
class EncodedDepth:
    data: bytes
    params: dict[str, Any] = field(default_factory=dict)


def encode_depth(depth: np.ndarray, config: DepthCodecConfig) -> EncodedDepth:
    """Encode an array of any shape. Non-positive/non-finite entries are treated as 'invalid'
    (stored as 0 / the minimum code) -- callers mask them with the validity mask on decode."""
    d = np.ascontiguousarray(depth, dtype=np.float32)
    clean = np.where(np.isfinite(d) & (d > 0), d, 0).astype(np.float32)
    base = {"codec": config.name, "shape": list(d.shape)}
    if config.name == "float32_zlib":
        raw = clean.astype("<f4").tobytes()
        return EncodedDepth(zlib.compress(raw, config.zlib_level), {**base, "lossless": True, "dtype": "<f4"})
    if config.name == "uint16_quant_zlib":
        pos = clean[clean > 0].astype(np.float64)
        lo = float(pos.min()) if pos.size else 0.0
        hi = float(pos.max()) if pos.size else 0.0
        scale = (hi - lo) / 65535.0
        q = np.zeros(d.shape, np.uint16) if scale == 0 else np.clip(
            np.rint((clean.astype(np.float64) - lo) / scale), 0, 65535
        ).astype(np.uint16)
        return EncodedDepth(
            zlib.compress(q.astype("<u2").tobytes(), config.zlib_level),
            {**base, "lossless": False, "dtype": "<u2", "depth_min": lo, "depth_max": hi,
             "max_abs_error": scale / 2.0},
        )
    raise ConfigurationError(f"unknown depth codec {config.name!r}")


def decode_depth(data: bytes, params: dict[str, Any]) -> np.ndarray:
    try:
        name = params["codec"]
        shape = tuple(int(s) for s in params["shape"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FormatError(f"malformed depth codec parameters: {exc}") from exc
    count = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if count < 0 or any(s < 0 for s in shape):
        raise FormatError("negative depth shape")
    if name == "float32_zlib":
        raw = inflate(data, count * 4, 'depth payload')
        return np.frombuffer(raw, "<f4").astype(np.float32).reshape(shape)
    if name == "uint16_quant_zlib":
        try:
            lo, hi = float(params["depth_min"]), float(params["depth_max"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FormatError(f"malformed quantisation parameters: {exc}") from exc
        raw = inflate(data, count * 2, 'depth payload')
        q = np.frombuffer(raw, "<u2").reshape(shape).astype(np.float64)
        return (lo + q * ((hi - lo) / 65535.0)).astype(np.float32)
    raise FormatError(f"invalid depth encoding identifier {name!r}")
