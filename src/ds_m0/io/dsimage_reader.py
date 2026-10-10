"""DS-Image Reader: JPEG + JUMBF file -> validated logical DS Asset."""

from __future__ import annotations

import logging
import zlib
from pathlib import Path

import numpy as np

from ..config.m0_config import DSImageReadConfig
from ..errors import FormatError, GeometryError, ValidationError
from ..geometry import camera_math
from ..model.asset import DSAsset
from ..model.camera import Camera
from ..model.layer import Layer
from ..model.metadata import DSImageMetadata, deserialize_metadata
from ._util import inflate
from .depth_codec import decode_depth
from .jpeg_codec import decode_rgb
from .jumbf_container import ContainerContents, DSContainerReader
from .sparse_layer1 import SparseLayer1Payload, decode_layer1, mask_byte_count, unpack_mask

log = logging.getLogger("ds_m0.reader")

_REQUIRED_PAYLOADS = ("layer0_depth", "layer0_mask", "layer1_mask", "layer1_rgb", "layer1_depth")


def read_container(path: str | Path) -> tuple[ContainerContents, DSImageMetadata]:
    reader = DSContainerReader().open(path)
    contents = reader.contents()
    entry = contents.payloads.get("metadata")
    if entry is None or entry[0] != "json":
        raise FormatError("DS metadata payload is missing")
    return contents, deserialize_metadata(entry[1])


def _payload(contents: ContainerContents, meta: DSImageMetadata, label: str, verify: bool) -> bytes:
    entry = contents.payloads.get(label)
    desc = meta.payload_descriptors.get(label)
    if entry is None or desc is None:
        raise FormatError(f"required payload {label!r} is missing")
    data = entry[1]
    if desc.get("length") != len(data):
        raise FormatError(f"payload {label!r} is {len(data)} bytes, metadata declares {desc.get('length')}")
    if verify and (zlib.crc32(data) & 0xFFFFFFFF) != desc.get("crc32"):
        raise FormatError(f"payload {label!r} failed its CRC-32 check")
    return data


def read_ds_image(input_path: str | Path, config: DSImageReadConfig | None = None) -> DSAsset:
    config = config or DSImageReadConfig()
    contents, meta = read_container(input_path)
    w, h = meta.spatial_width, meta.spatial_height

    rgb0 = decode_rgb(contents.primary_jpeg)
    if rgb0.shape != (h, w, 3):
        raise FormatError(f"primary JPEG is {rgb0.shape[1]}x{rgb0.shape[0]}, metadata declares {w}x{h}")
    try:
        camera = Camera.from_dict(meta.base_camera)
        camera_math.validate(camera)
    except GeometryError as exc:
        raise FormatError(f"DS metadata describes an invalid base camera: {exc}") from exc

    get = lambda label: _payload(contents, meta, label, config.verify_checksums)  # noqa: E731
    try:
        depth_params = meta.depth_parameters
        d0_params, d1_params = depth_params["layer0"], depth_params["layer1"]
        valid_count = int(meta.layer1_parameters["valid_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FormatError(f"malformed DS metadata parameters: {exc}") from exc
    if tuple(d0_params.get("shape", ())) != (h, w):
        raise FormatError("layer 0 depth shape does not match the spatial dimensions")

    mask0 = inflate(get("layer0_mask"), mask_byte_count(w, h), "layer 0 mask")
    valid0 = unpack_mask(mask0, w, h)
    depth0 = decode_depth(get("layer0_depth"), d0_params)
    depth0 = np.where(valid0, depth0, 0).astype(np.float32)
    layer0 = Layer(w, h, rgb0, depth0, valid0)

    mask1 = inflate(get("layer1_mask"), mask_byte_count(w, h), "layer 1 mask")
    n = int(unpack_mask(mask1, w, h).sum())
    if n != valid_count:
        raise FormatError(f"layer 1 mask has {n} valid pixels, metadata declares {valid_count}")
    rgb_raw = inflate(get("layer1_rgb"), n * 3, "layer 1 RGB stream")
    depth_stream = decode_depth(get("layer1_depth"), d1_params)
    if depth_stream.shape != (n,):
        raise FormatError(f"layer 1 depth stream has {depth_stream.size} samples, mask has {n}")
    layer1 = decode_layer1(
        SparseLayer1Payload(mask1, np.frombuffer(rgb_raw, np.uint8).reshape(n, 3), depth_stream), w, h
    )

    asset = DSAsset(meta.asset_version, w, h, camera, [layer0, layer1], dict(meta.asset_metadata))
    try:
        asset.validate()
    except ValidationError as exc:
        raise FormatError(f"DS-Image decodes to an invalid asset: {exc}") from exc
    log.info("DS-Image read: %s", input_path)
    return asset
