"""ds_m0 command line: build | write | read | render | roundtrip | inspect | evaluate | synth | export-config."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from ..config.m0_config import M0Config
from ..creator import build_ds_asset_with_diagnostics
from ..errors import DSError
from ..evaluation import compare_assets, measure_coverage, measure_layer1
from ..evaluation.reports import run_evaluation
from ..geometry.transformation import translated_camera
from ..io import (
    is_debug_asset,
    load_debug_asset,
    read_container,
    read_ds_image,
    read_source_dataset,
    save_debug_asset,
    write_ds_image,
)
from ..model.asset import DSAsset
from ..model.camera import Camera
from ..presenter import render
from ..presenter.output import save_render
from .synthetic import write_synthetic_h5


def _load_config(args) -> M0Config:
    cfg = M0Config.load(args.config) if getattr(args, "config", None) else M0Config()
    if getattr(args, "base_view", None) is not None:
        cfg.builder.base_view_id = args.base_view
    for flag, section, key in (
        ("rel_threshold", cfg.builder, "relative_depth_threshold"),
        ("abs_threshold", cfg.builder, "absolute_depth_threshold"),
        ("jpeg_quality", cfg.writer, "jpeg_quality"),
        ("depth_codec", cfg.writer, "depth_codec"),
    ):
        if getattr(args, flag, None) is not None:
            setattr(section, key, getattr(args, flag))
    cfg.validate()
    return cfg


def _load_asset(path: str) -> DSAsset:
    return load_debug_asset(path) if Path(path).is_dir() and is_debug_asset(path) else read_ds_image(path)


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def cmd_build(args) -> int:
    cfg = _load_config(args)
    asset, diag = build_ds_asset_with_diagnostics(read_source_dataset(args.input), cfg.builder)
    save_debug_asset(asset, args.output)
    _print({"asset": args.output, "layer1_valid_pixel_count": diag.layer1_valid_pixel_count,
            "layer1_occupancy": diag.layer1_occupancy, "candidate_count": diag.candidate_count,
            "timings_s": diag.timings_s})
    return 0


def cmd_write(args) -> int:
    cfg = _load_config(args)
    res = write_ds_image(load_debug_asset(args.asset), args.output, cfg.writer)
    _print({"path": res.path, "total_bytes": res.total_bytes, "components": res.component_bytes})
    return 0


def cmd_read(args) -> int:
    asset = read_ds_image(args.input)
    save_debug_asset(asset, args.output)
    _print(_inspect_asset(args.input, asset))
    return 0


def _target_camera(args, asset: DSAsset) -> Camera:
    if args.camera:
        return Camera.from_dict(json.loads(Path(args.camera).read_text()))
    if args.translate is not None:
        return translated_camera(asset.base_camera, np.asarray(args.translate, float))
    return asset.base_camera


def cmd_render(args) -> int:
    asset = _load_asset(args.asset)
    cfg = _load_config(args)
    res = render(asset, _target_camera(args, asset), cfg.presenter)
    files = save_render(res, args.output, args.name)
    _print({**measure_coverage(res.coverage).to_dict(), "files": files})
    return 0


def cmd_roundtrip(args) -> int:
    cfg = _load_config(args)
    asset_a, _ = build_ds_asset_with_diagnostics(read_source_dataset(args.input), cfg.builder)
    out = Path(args.output_dir)
    write_ds_image(asset_a, out / "roundtrip.jpg", cfg.writer)
    asset_b = read_ds_image(out / "roundtrip.jpg", cfg.reader)
    from ..evaluation.roundtrip_metrics import ComparisonConfig

    tol = 0.0 if cfg.writer.depth_codec == "float32_zlib" else 1e-3 * float(asset_a.layers[0].depth.max())
    report = compare_assets(asset_a, asset_b, ComparisonConfig(cfg.evaluation.rgb_mean_abs_tolerance, tol))
    _print(report)
    return 0 if report["passed"] else 1


def _inspect_asset(path: str, asset: DSAsset | None = None) -> dict:
    from ..evaluation import measure_ds_image

    contents, meta = read_container(path)
    asset = asset or read_ds_image(path)
    fm = measure_ds_image(path)
    sp = measure_layer1(asset.layers[1])
    return {
        "format_version": meta.format_version, "asset_version": meta.asset_version,
        "width": meta.spatial_width, "height": meta.spatial_height, "layer_count": meta.layer_count,
        "base_camera": meta.base_camera, "jpeg_parameters": meta.jpeg_parameters,
        "depth_parameters": meta.depth_parameters, "layer1_occupancy": sp.occupancy,
        "sizes": fm.to_dict(), "payload_labels": sorted(contents.payloads),
    }


def cmd_inspect(args) -> int:
    _print(_inspect_asset(args.input))
    return 0


def cmd_evaluate(args) -> int:
    cfg = _load_config(args)
    summary = run_evaluation(cfg, args.input, args.output_dir, args.miv_file)
    _print(summary)
    print((Path(args.output_dir) / "summary.txt").read_text())
    return 0 if summary["passed"] else 1


def cmd_synth(args) -> int:
    path = write_synthetic_h5(args.output, args.width, args.height, args.baseline)
    _print({"written": str(path)})
    return 0


def cmd_export_config(args) -> int:
    cfg = _load_config(args)
    cfg.save(args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ds_m0", description="DS-Image M0 prototype")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, build=False, write=False):
        sp.add_argument("--config", help="M0 config JSON (see export-config)")
        if build:
            sp.add_argument("--base-view", type=int)
            sp.add_argument("--rel-threshold", type=float)
            sp.add_argument("--abs-threshold", type=float)
        if write:
            sp.add_argument("--jpeg-quality", type=int)
            sp.add_argument("--depth-codec", choices=("float32_zlib", "uint16_quant_zlib"))

    s = sub.add_parser("build", help="9-view RGBD h5 -> DS Asset (debug directory)")
    s.add_argument("--input", required=True); s.add_argument("--output", required=True)
    common(s, build=True); s.set_defaults(fn=cmd_build)

    s = sub.add_parser("write", help="DS Asset -> DS-Image")
    s.add_argument("--asset", required=True); s.add_argument("--output", required=True)
    common(s, write=True); s.set_defaults(fn=cmd_write)

    s = sub.add_parser("read", help="DS-Image -> DS Asset (debug directory)")
    s.add_argument("--input", required=True); s.add_argument("--output", required=True)
    s.set_defaults(fn=cmd_read)

    s = sub.add_parser("render", help="DS Asset or DS-Image + target camera -> RGB + coverage")
    s.add_argument("--asset", required=True, help="debug asset directory or DS-Image file")
    s.add_argument("--camera", help="target camera JSON"); s.add_argument("--output", required=True)
    s.add_argument("--translate", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="base camera moved by this offset in its own axes")
    s.add_argument("--name", default="render"); common(s); s.set_defaults(fn=cmd_render)

    s = sub.add_parser("roundtrip", help="source -> asset A -> DS-Image -> asset B -> compare")
    s.add_argument("--input", required=True); s.add_argument("--output-dir", required=True)
    common(s, build=True, write=True); s.set_defaults(fn=cmd_roundtrip)

    s = sub.add_parser("inspect", help="report DS-Image structure and sizes (no rendering)")
    s.add_argument("--input", required=True); s.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("evaluate", help="full reference run with reports A-E and a summary")
    s.add_argument("--input", required=True); s.add_argument("--output-dir", required=True)
    s.add_argument("--miv-file"); common(s, build=True, write=True); s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("synth", help="write a deterministic synthetic 9-view dataset (.h5)")
    s.add_argument("--output", required=True)
    s.add_argument("--width", type=int, default=96); s.add_argument("--height", type=int, default=96)
    s.add_argument("--baseline", type=float, default=0.35); s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("export-config", help="write the (default or overridden) config as JSON")
    s.add_argument("--output", required=True); common(s, build=True, write=True)
    s.set_defaults(fn=cmd_export_config)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        return args.fn(args)
    except DSError as exc:
        print(f"ds_m0: {type(exc).__name__}: {exc}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # noqa: BLE001 -- spec: non-zero status for fatal errors
        print(f"ds_m0: unexpected failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 6


if __name__ == "__main__":
    sys.exit(main())
