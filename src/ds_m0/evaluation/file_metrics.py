"""Physical size accounting of a DS-Image (independent of the Writer)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ..io.dsimage_reader import read_container


@dataclass
class FileMetrics:
    primary_jpeg_bytes: int
    metadata_bytes: int
    base_depth_bytes: int
    layer0_mask_bytes: int
    layer1_mask_bytes: int
    layer1_rgb_bytes: int
    layer1_depth_bytes: int
    container_overhead_bytes: int
    ds_total_bytes: int
    jumbf_app11_bytes: int

    def to_dict(self) -> dict:
        return asdict(self)


def measure_ds_image(path: str | Path) -> FileMetrics:
    contents, _ = read_container(path)
    size = lambda label: len(contents.payloads[label][1])  # noqa: E731
    total = Path(path).stat().st_size
    parts = {
        "primary_jpeg_bytes": len(contents.primary_jpeg),
        "metadata_bytes": size("metadata"),
        "base_depth_bytes": size("layer0_depth"),
        "layer0_mask_bytes": size("layer0_mask"),
        "layer1_mask_bytes": size("layer1_mask"),
        "layer1_rgb_bytes": size("layer1_rgb"),
        "layer1_depth_bytes": size("layer1_depth"),
    }
    return FileMetrics(**parts, container_overhead_bytes=total - sum(parts.values()),
                       ds_total_bytes=total, jumbf_app11_bytes=contents.app11_bytes)
