"""Writing render outputs to disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .renderer import RenderResult


def save_render(result: RenderResult, directory: str | Path, name: str) -> dict[str, str]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rgb_path = directory / f"{name}_rgb.png"
    cov_path = directory / f"{name}_coverage.png"
    Image.fromarray(result.rgb).save(rgb_path)
    Image.fromarray((result.coverage.astype(np.uint8) * 255)).save(cov_path)
    return {"rgb": str(rgb_path), "coverage": str(cov_path)}
