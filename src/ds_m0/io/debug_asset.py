"""Internal/debug on-disk form of a DS Asset (a directory). NOT a DS-Image."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..errors import FormatError
from ..model.asset import DSAsset
from ..model.camera import Camera
from ..model.layer import Layer

MARKER = "ds_m0_debug_asset"


def save_debug_asset(asset: DSAsset, directory: str | Path) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    for i, layer in enumerate(asset.layers):
        np.save(d / f"layer{i}_rgb.npy", layer.rgb)
        np.save(d / f"layer{i}_depth.npy", layer.depth)
        np.save(d / f"layer{i}_valid.npy", layer.valid)
    meta = {
        "kind": MARKER, "version": asset.version, "spatial_width": asset.spatial_width,
        "spatial_height": asset.spatial_height, "layer_count": asset.layer_count,
        "base_camera": asset.base_camera.to_dict(), "metadata": asset.metadata,
    }
    (d / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def is_debug_asset(path: str | Path) -> bool:
    return (Path(path) / "metadata.json").is_file()


def load_debug_asset(directory: str | Path) -> DSAsset:
    d = Path(directory)
    try:
        meta = json.loads((d / "metadata.json").read_text())
        if meta.get("kind") != MARKER:
            raise FormatError(f"{d} is not a ds_m0 debug asset")
        w, h = int(meta["spatial_width"]), int(meta["spatial_height"])
        layers = [
            Layer(w, h, np.load(d / f"layer{i}_rgb.npy"), np.load(d / f"layer{i}_depth.npy"),
                  np.load(d / f"layer{i}_valid.npy"))
            for i in range(int(meta["layer_count"]))
        ]
        return DSAsset(meta["version"], w, h, Camera.from_dict(meta["base_camera"]), layers, meta["metadata"])
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise FormatError(f"cannot load debug asset {d}: {exc}") from exc
