"""Full M0 reference run: build -> write -> read -> compare -> render, with reports A-E."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .. import GENERATOR_VERSION
from ..config.m0_config import M0Config
from ..creator import build_ds_asset_with_diagnostics
from ..geometry.transformation import translated_camera
from ..io import read_ds_image, read_source_dataset, write_ds_image
from ..model.asset import DSAsset
from ..presenter import render
from ..presenter.output import save_render
from .baseline_metrics import measure_miv_baseline
from .coverage_metrics import measure_coverage
from .file_metrics import measure_ds_image
from .roundtrip_metrics import ComparisonConfig, compare_assets
from .sparsity_metrics import measure_layer1

log = logging.getLogger("ds_m0.evaluation")


def legacy_jpeg_check(path: str | Path) -> dict:
    """A JPEG-only consumer must still decode the primary image of a DS-Image."""
    data = Path(path).read_bytes()
    result: dict = {"jpeg_signature": data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"}
    try:
        with Image.open(path) as im:
            im.load()
            result["pillow"] = {"ok": im.format == "JPEG", "size": list(im.size)}
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the evaluation
        result["pillow"] = {"ok": False, "error": str(exc)}
    sips = shutil.which("sips")
    if sips:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "legacy.png"
            proc = subprocess.run([sips, "-s", "format", "png", str(path), "--out", str(out)],
                                  capture_output=True, text=True)
            ok = proc.returncode == 0 and out.is_file()
            result["sips"] = {"ok": ok, "size": list(Image.open(out).size) if ok else None}
    result["passed"] = result["jpeg_signature"] and all(
        v["ok"] for k, v in result.items() if isinstance(v, dict)
    )
    return result


def _save_layer_views(asset: DSAsset, out: Path) -> None:
    l0, l1 = asset.layers
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(l0.rgb).save(out / "layer0_rgb.png")
    Image.fromarray(l1.valid.astype(np.uint8) * 255).save(out / "layer1_valid.png")
    Image.fromarray(np.where(l1.valid[..., None], l1.rgb, 0).astype(np.uint8)).save(out / "layer1_rgb.png")
    for name, layer in (("layer0", l0), ("layer1", l1)):
        if layer.valid.any():
            d = layer.depth[layer.valid]
            lo, hi = float(d.min()), float(d.max())
            norm = np.where(layer.valid, (layer.depth - lo) / max(hi - lo, 1e-12), 0)
            Image.fromarray((np.clip(norm, 0, 1) * 255).astype(np.uint8)).save(out / f"{name}_depth.png")


def _peak_memory_bytes() -> int:
    """Peak RSS of this process (ru_maxrss is bytes on macOS, kibibytes elsewhere)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def run_evaluation(config: M0Config, dataset_path: str | Path, output_dir: str | Path,
                   miv_file: str | Path | None = None) -> dict:
    config.validate()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    tick = time.perf_counter

    t = tick(); source = read_source_dataset(dataset_path); timings["source_loading"] = tick() - t
    asset_a, diag = build_ds_asset_with_diagnostics(source, config.builder)
    timings.update(diag.timings_s)
    ds_path = out / "ds_image.jpg"
    wr = write_ds_image(asset_a, ds_path, config.writer); timings["ds_image_write"] = wr.timings_s["write"]
    t = tick(); asset_b = read_ds_image(ds_path, config.reader); timings["ds_image_read"] = tick() - t
    del source  # from here on nothing may need the source views (Presenter independence)

    file_metrics = measure_ds_image(ds_path)
    sparsity = measure_layer1(asset_a.layers[1])
    if config.writer.depth_codec != "float32_zlib":
        depth_tol = config.evaluation.depth_extra_tolerance + 1e-6 * float(asset_a.layers[0].depth.max())
        d = asset_a.layers[0].depth[asset_a.layers[0].valid]
        d1 = asset_a.layers[1].depth[asset_a.layers[1].valid]
        span = float(max(d.max(), d1.max() if d1.size else 0) - min(d.min(), d1.min() if d1.size else d.min()))
        depth_tol += span / 65535.0
    else:
        depth_tol = 0.0
    roundtrip = compare_assets(asset_a, asset_b, ComparisonConfig(
        config.evaluation.rgb_mean_abs_tolerance, depth_tol))

    median_depth = float(np.median(asset_b.layers[0].depth[asset_b.layers[0].valid]))
    scale = median_depth if config.evaluation.offsets_scaled_by_median_depth else 1.0
    render_reports, determinism_ok, render_s = [], True, 0.0
    for i, off in enumerate(config.evaluation.target_offsets):
        cam = translated_camera(asset_b.base_camera, np.asarray(off, float) * scale)
        t = tick(); res = render(asset_b, cam, config.presenter); render_s += tick() - t
        again = render(asset_b, cam, config.presenter)
        same = bool((res.rgb == again.rgb).all() and (res.coverage == again.coverage).all())
        determinism_ok &= same
        res_a = render(asset_a, cam, config.presenter)
        name = f"target_{i:02d}"
        files = save_render(res, out / "renders", name)
        cov = measure_coverage(res.coverage)
        render_reports.append({
            "name": name, "offset_in_base_camera_axes": [float(x) * scale for x in off],
            "target_camera": cam.to_dict(), **cov.to_dict(), "files": files, "deterministic": same,
            "coverage_identical_to_original_asset_render": bool((res.coverage == res_a.coverage).all()),
        })
    timings["presenter_render_total"] = render_s
    base = render(asset_b, asset_b.base_camera, config.presenter)
    base_ok = bool(base.coverage[asset_b.layers[0].valid].all())
    presenter_pass = base_ok and determinism_ok

    legacy = legacy_jpeg_check(ds_path)
    _save_layer_views(asset_b, out / "layers")

    miv_path = miv_file or config.evaluation.miv_file
    miv = measure_miv_baseline(ds_path, miv_path) if miv_path else None

    asset_report = {
        "spatial_width": asset_a.spatial_width, "spatial_height": asset_a.spatial_height,
        "base_view_id": diag.base_view_id, "number_of_views": diag.number_of_views,
        "base_camera": asset_a.base_camera.to_dict(), "layer_count": asset_a.layer_count,
        "layer0_valid_pixel_count": diag.layer0_valid_pixel_count,
        "layer1_valid_pixel_count": diag.layer1_valid_pixel_count,
        "layer1_occupancy": diag.layer1_occupancy, "candidate_count": diag.candidate_count,
        "candidates_per_view": {str(k): v for k, v in diag.candidates_per_view.items()},
        "asset_content_hash": asset_a.content_hash(), "asset_metadata": asset_a.metadata,
    }
    file_report = {**file_metrics.to_dict(), "sparsity": sparsity.to_dict(), "writer_components": wr.component_bytes,
                   "writer_total_matches_file": wr.total_bytes == file_metrics.ds_total_bytes}
    reports = {"report_a_asset": asset_report, "report_b_file": file_report, "report_c_roundtrip": roundtrip,
               "report_d_render": {"targets": render_reports, "base_view_covers_layer0": base_ok,
                                   "determinism": determinism_ok}}
    if miv:
        reports["report_e_miv"] = miv
    for name, obj in reports.items():
        _dump(out / f"{name}.json", obj)
    config.save(out / "config.json")

    dataset_hash = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
    run_info = {
        "software_version": GENERATOR_VERSION, "dataset_path": str(dataset_path), "dataset_sha256": dataset_hash,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "python": platform.python_version(), "numpy": np.__version__, "timings_s": timings,
        "peak_memory_bytes": _peak_memory_bytes(),
    }
    _dump(out / "run_info.json", run_info)

    summary = {
        "dataset": str(dataset_path), "base_view": diag.base_view_id,
        "resolution": f"{asset_a.spatial_width}x{asset_a.spatial_height}",
        "layer1_occupancy": diag.layer1_occupancy, "ds_image_size": file_metrics.ds_total_bytes,
        "roundtrip_pass": roundtrip["passed"], "presenter_pass": presenter_pass, "legacy_jpeg_pass": legacy["passed"],
    }
    lines = [
        f"Dataset:             {dataset_path}",
        f"Base View:           {diag.base_view_id}",
        f"Resolution:          {summary['resolution']}",
        f"Layer 1 occupancy:   {diag.layer1_occupancy:.4%}",
        f"DS-Image size:       {file_metrics.ds_total_bytes} bytes",
        f"  Primary JPEG:      {file_metrics.primary_jpeg_bytes}",
        f"  Base Depth:        {file_metrics.base_depth_bytes}",
        f"  Layer 1 Sparse:    {file_metrics.layer1_mask_bytes + file_metrics.layer1_rgb_bytes + file_metrics.layer1_depth_bytes}"
        f" (mask {file_metrics.layer1_mask_bytes} / rgb {file_metrics.layer1_rgb_bytes} / depth {file_metrics.layer1_depth_bytes})",
        f"  Metadata/overhead: {file_metrics.metadata_bytes + file_metrics.layer0_mask_bytes + file_metrics.container_overhead_bytes}",
        f"Sparse vs dense L1:  {sparsity.sparse_payload_bytes} vs {sparsity.dense_payload_bytes} bytes raw "
        f"({sparsity.sparse_savings_ratio:.1%} saved)",
        f"Round-trip:          {'PASS' if roundtrip['passed'] else 'FAIL'}",
        f"Presenter:           {'PASS' if presenter_pass else 'FAIL'}",
        f"Legacy JPEG:         {'PASS' if legacy['passed'] else 'FAIL'}",
        f"MIV size:            {miv['miv_total_size'] if miv else 'n/a'}",
        f"DS/MIV ratio:        {miv['ds_to_miv_ratio'] if miv else 'n/a'}",
    ]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    _dump(out / "legacy_jpeg.json", legacy)
    summary["passed"] = bool(roundtrip["passed"] and presenter_pass and legacy["passed"])
    summary["output_directory"] = str(out)
    return summary
