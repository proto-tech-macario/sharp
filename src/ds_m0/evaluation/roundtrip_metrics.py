"""Round-trip comparison of two DS Assets (spec Part IV T08)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model.asset import DSAsset


@dataclass
class ComparisonConfig:
    rgb_mean_abs_tolerance: float = 12.0  # layer-0 JPEG sanity bound (documented, configurable)
    depth_abs_tolerance: float = 0.0  # 0 = must be numerically identical (lossless depth)
    layer1_rgb_exact: bool = True  # Layer-1 RGB is always lossless in M0


def _depth_stats(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> dict:
    if not valid.any():
        return {"max_abs_error": 0.0, "mean_abs_error": 0.0, "max_relative_error": 0.0, "samples": 0}
    da, db = a[valid].astype(np.float64), b[valid].astype(np.float64)
    err = np.abs(da - db)
    return {"max_abs_error": float(err.max()), "mean_abs_error": float(err.mean()),
            "max_relative_error": float((err / da).max()), "samples": int(valid.sum())}


def compare_assets(original: DSAsset, reconstructed: DSAsset, config: ComparisonConfig | None = None) -> dict:
    config = config or ComparisonConfig()
    a, b = original, reconstructed
    structural = {
        "spatial_width": a.spatial_width == b.spatial_width,
        "spatial_height": a.spatial_height == b.spatial_height,
        "layer_count": a.layer_count == b.layer_count,
        "base_camera": a.base_camera.equals(b.base_camera),
    }
    same_shape = structural["spatial_width"] and structural["spatial_height"] and structural["layer_count"]
    report: dict = {"structural": structural}
    if not same_shape:
        report.update(passed=False, failures=["asset dimensions/layer count differ"])
        return report
    l0a, l0b, l1a, l1b = a.layers[0], b.layers[0], a.layers[1], b.layers[1]
    mask_mismatch = {"layer0": int((l0a.valid != l0b.valid).sum()), "layer1": int((l1a.valid != l1b.valid).sum())}

    diff = np.abs(l0a.rgb.astype(np.int16) - l0b.rgb.astype(np.int16))
    mse = float((diff.astype(np.float64) ** 2).mean())
    layer0_rgb = {
        "max_abs_error": int(diff.max()), "mean_abs_error": float(diff.mean()),
        "psnr_db": float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse)),
    }
    v1 = l1a.valid & l1b.valid
    l1_rgb_mismatch = int((l1a.rgb[v1] != l1b.rgb[v1]).any(axis=1).sum()) if v1.any() else 0
    depth = {
        "layer0": _depth_stats(l0a.depth, l0b.depth, l0a.valid & l0b.valid),
        "layer1": _depth_stats(l1a.depth, l1b.depth, v1),
    }
    failures = [k for k, ok in structural.items() if not ok]
    if mask_mismatch["layer0"] or mask_mismatch["layer1"]:
        failures.append("validity mask mismatch")
    if layer0_rgb["mean_abs_error"] > config.rgb_mean_abs_tolerance:
        failures.append("layer 0 RGB (JPEG) error above tolerance")
    if config.layer1_rgb_exact and l1_rgb_mismatch:
        failures.append("layer 1 RGB not exact")
    for name, s in depth.items():
        if s["max_abs_error"] > config.depth_abs_tolerance:
            failures.append(f"{name} depth error above tolerance")
    report.update(
        mask_mismatch_count=mask_mismatch, layer0_rgb=layer0_rgb, layer1_rgb_mismatch_count=l1_rgb_mismatch,
        depth=depth,
        tolerances={"rgb_mean_abs_tolerance": config.rgb_mean_abs_tolerance,
                    "depth_abs_tolerance": config.depth_abs_tolerance},
        camera_mismatch=not structural["base_camera"], passed=not failures, failures=failures,
    )
    return report
