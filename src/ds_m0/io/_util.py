from __future__ import annotations

import zlib

from ..errors import FormatError


def inflate(data: bytes, expected: int, what: str) -> bytes:
    """zlib-decompress, refusing anything that does not decode to exactly `expected` bytes."""
    d = zlib.decompressobj()
    try:
        out = d.decompress(data, expected + 1)
    except zlib.error as exc:
        raise FormatError(f"{what} is not valid zlib data: {exc}") from exc
    if not d.eof or d.unused_data:
        raise FormatError(f"{what} is truncated or has trailing data")
    if len(out) != expected:
        raise FormatError(f"{what} decodes to {len(out)} bytes, expected {expected}")
    return out
