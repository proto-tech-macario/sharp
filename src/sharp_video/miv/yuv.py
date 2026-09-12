"""8-bit RGB <-> planar YUV 4:2:0 and raw .yuv file I/O for TMIV.

Texture uses BT.709 coefficients in limited ("video") range, which is what
the VVC texture sub-bitstreams are tagged with. Chroma is subsampled by 2x2
averaging and upsampled by sample repetition. Samples above 8 bits are stored
little-endian in 16-bit words (FFmpeg's `yuv420p10le` / `yuv420p16le`).
"""

from __future__ import annotations

from typing import BinaryIO

import numpy as np

KR, KB = 0.2126, 0.0722
KG = 1.0 - KR - KB


def _check_even(height: int, width: int) -> None:
    if height % 2 or width % 2:
        raise ValueError(f"4:2:0 needs even width and height, got {width}x{height}")


def rgb_to_yuv420(
    rgb: np.ndarray, bit_depth: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[H,W,3] uint8 RGB -> (Y [H,W], U [H/2,W/2], V [H/2,W/2]) uint16."""
    height, width = rgb.shape[:2]
    _check_even(height, width)
    scale = 1 << (bit_depth - 8)
    rgbf = rgb.astype(np.float64) / 255.0
    r, g, b = rgbf[..., 0], rgbf[..., 1], rgbf[..., 2]
    y = KR * r + KG * g + KB * b
    pb = (b - y) / (2.0 * (1.0 - KB))
    pr = (r - y) / (2.0 * (1.0 - KR))

    def down(plane: np.ndarray) -> np.ndarray:
        return plane.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))

    max_value = (1 << bit_depth) - 1
    Y = np.clip(np.rint((16.0 + 219.0 * y) * scale), 0, max_value)
    U = np.clip(np.rint((128.0 + 224.0 * down(pb)) * scale), 0, max_value)
    V = np.clip(np.rint((128.0 + 224.0 * down(pr)) * scale), 0, max_value)
    return Y.astype(np.uint16), U.astype(np.uint16), V.astype(np.uint16)


def yuv420_to_rgb(
    Y: np.ndarray, U: np.ndarray, V: np.ndarray, bit_depth: int = 10
) -> np.ndarray:
    """Inverse of `rgb_to_yuv420` -> [H,W,3] uint8."""
    scale = float(1 << (bit_depth - 8))
    y = (Y.astype(np.float64) / scale - 16.0) / 219.0
    pb = (np.repeat(np.repeat(U, 2, axis=0), 2, axis=1).astype(np.float64) / scale - 128.0) / 224.0
    pr = (np.repeat(np.repeat(V, 2, axis=0), 2, axis=1).astype(np.float64) / scale - 128.0) / 224.0
    r = y + 2.0 * (1.0 - KR) * pr
    b = y + 2.0 * (1.0 - KB) * pb
    g = (y - KR * r - KB * b) / KG
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def geometry_planes(samples: np.ndarray, bit_depth: int = 16):
    """Geometry samples as a 4:2:0 frame: luma = samples, chroma = mid-grey (unused by TMIV)."""
    height, width = samples.shape
    _check_even(height, width)
    mid = np.full((height // 2, width // 2), 1 << (bit_depth - 1), dtype=np.uint16)
    return samples.astype(np.uint16), mid, mid


def frame_nbytes(width: int, height: int, bit_depth: int) -> int:
    """Size in bytes of one 4:2:0 frame."""
    bytes_per_sample = 1 if bit_depth <= 8 else 2
    return (width * height * 3 // 2) * bytes_per_sample


def write_yuv420(stream: BinaryIO, Y: np.ndarray, U: np.ndarray, V: np.ndarray,
                 bit_depth: int) -> None:
    """Append one planar 4:2:0 frame to `stream`."""
    dtype = np.uint8 if bit_depth <= 8 else np.dtype("<u2")
    for plane in (Y, U, V):
        stream.write(np.ascontiguousarray(plane, dtype=dtype).tobytes())


def read_yuv420_frame(
    stream: BinaryIO, width: int, height: int, bit_depth: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the next planar 4:2:0 frame from `stream`; raises EOFError at end of file."""
    dtype = np.uint8 if bit_depth <= 8 else np.dtype("<u2")
    nbytes = frame_nbytes(width, height, bit_depth)
    data = stream.read(nbytes)
    if len(data) < nbytes:
        raise EOFError("end of YUV stream")
    samples = np.frombuffer(data, dtype=dtype).astype(np.uint16)
    luma = width * height
    chroma = luma // 4
    Y = samples[:luma].reshape(height, width)
    U = samples[luma:luma + chroma].reshape(height // 2, width // 2)
    V = samples[luma + chroma:].reshape(height // 2, width // 2)
    return Y, U, V
