"""Primary-JPEG codec (Pillow)."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..errors import FormatError

_SUBSAMPLING = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}


@dataclass
class JPEGConfig:
    quality: int = 90
    subsampling: str = "4:2:0"


def encode_rgb(image: np.ndarray, config: JPEGConfig) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(image, "RGB").save(
        buf, "JPEG", quality=config.quality, subsampling=_SUBSAMPLING[config.subsampling],
        optimize=False, progressive=False,
    )
    return buf.getvalue()


def decode_rgb(jpeg: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(jpeg)) as im:
            if im.format != "JPEG":
                raise FormatError(f"primary image is {im.format}, expected JPEG")
            return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise FormatError(f"primary JPEG cannot be decoded: {exc}") from exc
