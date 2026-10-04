"""DS-Image metadata object and its deterministic JSON serialisation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..errors import FormatError

FORMAT_VERSION = "1.0"


@dataclass
class DSImageMetadata:
    format_version: str
    asset_version: str
    spatial_width: int
    spatial_height: int
    base_view_id: int
    layer_count: int
    coordinate_convention: str
    base_camera: dict[str, Any]
    jpeg_parameters: dict[str, Any]
    depth_parameters: dict[str, Any]
    layer1_parameters: dict[str, Any]
    payload_descriptors: dict[str, dict[str, Any]]
    asset_metadata: dict[str, Any] = field(default_factory=dict)


_REQUIRED = (
    "format_version", "asset_version", "spatial_width", "spatial_height", "base_view_id",
    "layer_count", "coordinate_convention", "base_camera", "jpeg_parameters",
    "depth_parameters", "layer1_parameters", "payload_descriptors",
)


def serialize_metadata(metadata: DSImageMetadata) -> bytes:
    """Sorted keys, compact separators, UTF-8: byte-deterministic."""
    return json.dumps(metadata.__dict__, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def deserialize_metadata(data: bytes) -> DSImageMetadata:
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormatError(f"malformed DS metadata: {exc}") from exc
    if not isinstance(obj, dict):
        raise FormatError("DS metadata must be a JSON object")
    missing = [k for k in _REQUIRED if k not in obj]
    if missing:
        raise FormatError(f"DS metadata is missing keys: {missing}")
    if str(obj["format_version"]).split(".")[0] != FORMAT_VERSION.split(".")[0]:
        raise FormatError(
            f"unsupported DS format_version {obj['format_version']!r} (reader supports {FORMAT_VERSION})"
        )
    for key in ("spatial_width", "spatial_height", "base_view_id", "layer_count"):
        if not isinstance(obj[key], int) or isinstance(obj[key], bool):
            raise FormatError(f"DS metadata {key} must be an integer")
    if obj["spatial_width"] <= 0 or obj["spatial_height"] <= 0:
        raise FormatError("DS metadata declares non-positive dimensions")
    if obj["layer_count"] != 2:
        raise FormatError(f"M0 supports layer_count == 2, file declares {obj['layer_count']}")
    for key in ("base_camera", "jpeg_parameters", "depth_parameters", "layer1_parameters", "payload_descriptors"):
        if not isinstance(obj[key], dict):
            raise FormatError(f"DS metadata {key} must be an object")
    return DSImageMetadata(**{k: obj[k] for k in _REQUIRED}, asset_metadata=obj.get("asset_metadata", {}))
