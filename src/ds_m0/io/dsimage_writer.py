"""DS-Image Writer: DS Asset -> JPEG + JUMBF file. Never mutates the asset."""

from __future__ import annotations

import logging
import os
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config.m0_config import DSImageWriteConfig
from ..model.asset import DSAsset
from ..model.metadata import FORMAT_VERSION, DSImageMetadata, serialize_metadata
from .depth_codec import DepthCodecConfig, encode_depth
from .jpeg_codec import JPEGConfig, encode_rgb
from .jumbf_container import DSContainerWriter
from .sparse_layer1 import encode_layer1, pack_mask

log = logging.getLogger("ds_m0.writer")


@dataclass
class WriteResult:
    path: str
    total_bytes: int
    component_bytes: dict[str, int]  # primary_jpeg, metadata, base_depth, layer0_mask, layer1_mask/rgb/depth, container_overhead
    timings_s: dict[str, float] = field(default_factory=dict)


def _descriptor(data: bytes, codec: str) -> dict:
    return {"length": len(data), "crc32": zlib.crc32(data) & 0xFFFFFFFF, "codec": codec}


def write_ds_image(asset: DSAsset, output_path: str | Path, config: DSImageWriteConfig | None = None) -> WriteResult:
    config = config or DSImageWriteConfig()
    config.validate()
    asset.validate()
    t0 = time.perf_counter()
    l0, l1 = asset.layers
    level = config.zlib_level

    jpeg = encode_rgb(l0.rgb, JPEGConfig(config.jpeg_quality, config.jpeg_subsampling))
    log.info("JPEG encoded (%d bytes)", len(jpeg))

    depth_cfg = DepthCodecConfig(config.depth_codec, level)
    d0 = encode_depth(np.where(l0.valid, l0.depth, 0).astype(np.float32), depth_cfg)
    mask0 = zlib.compress(pack_mask(l0.valid), level)
    sparse = encode_layer1(l1)
    mask1 = zlib.compress(sparse.mask, level)
    rgb1 = zlib.compress(np.ascontiguousarray(sparse.rgb_stream).tobytes(), level)
    d1 = encode_depth(sparse.depth_stream, depth_cfg)
    log.info("depth encoded (%s)", config.depth_codec)

    payloads = {
        "layer0_depth": (d0.data, config.depth_codec),
        "layer0_mask": (mask0, "bitpack+zlib"),
        "layer1_mask": (mask1, "bitpack+zlib"),
        "layer1_rgb": (rgb1, "uint8_rgb+zlib"),
        "layer1_depth": (d1.data, config.depth_codec),
    }
    metadata = DSImageMetadata(
        format_version=FORMAT_VERSION,
        asset_version=asset.version,
        spatial_width=asset.spatial_width,
        spatial_height=asset.spatial_height,
        base_view_id=int(asset.metadata.get("base_view_id", 0)),
        layer_count=asset.layer_count,
        coordinate_convention=asset.base_camera.coordinate_convention,
        base_camera=asset.base_camera.to_dict(),
        jpeg_parameters={"quality": config.jpeg_quality, "subsampling": config.jpeg_subsampling},
        depth_parameters={"layer0": d0.params, "layer1": d1.params},
        layer1_parameters={"codec": config.sparse_layer1_codec, "valid_count": sparse.valid_count,
                           "stream_order": "row_major_raster"},
        payload_descriptors={k: _descriptor(*v) for k, v in payloads.items()},
        asset_metadata=dict(asset.metadata),
    )
    meta_bytes = serialize_metadata(metadata)

    container = DSContainerWriter().create()
    container.add_primary_jpeg(jpeg)
    container.add_ds_payload("metadata", "json", meta_bytes)
    for label, (data, _) in payloads.items():
        container.add_ds_payload(label, "bidb", data)
    blob = container.finalize()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_bytes(blob)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    components = {
        "primary_jpeg": len(jpeg), "metadata": len(meta_bytes),
        "base_depth": len(d0.data), "layer0_mask": len(mask0),
        "layer1_mask": len(mask1), "layer1_rgb": len(rgb1), "layer1_depth": len(d1.data),
    }
    components["container_overhead"] = len(blob) - sum(components.values())
    log.info("DS-Image written: %s (%d bytes)", path, len(blob))
    return WriteResult(str(path), len(blob), components, {"write": time.perf_counter() - t0})
