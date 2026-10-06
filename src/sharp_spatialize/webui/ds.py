"""DS-Image analysis for the web UI: build, write, read back, render, and score.

Takes a finished job's 9-view spatial photo and runs the `ds_m0` chain on it:

    9 views -> DS Asset (2 layers) -> DS-Image (JPEG + JUMBF) -> DS Asset -> Presenter

then renders the Presenter at every source camera and compares each render with
that source view, which is the reference image SHARP rendered from that camera.
Everything here is CPU-only numpy/Pillow; `ds_m0` is imported lazily so the UI
still runs (without the DS panels) where it is not installed.
"""

from __future__ import annotations

import copy
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .jobs import depth_to_rgb, encode_jpeg

DEPTH_CODECS = ("float32_zlib", "uint16_quant_zlib")
DEFAULT_DEPTH_CODEC = "float32_zlib"
#: Presenter modes rendered for every analysis; the first is the UI default.
PRESENTERS = ("mesh", "points")
DEFAULT_PRESENTER = PRESENTERS[0]
SCORE_KEYS = ("presenter_only", "ds_image", "layer0_only", "leave_one_out")
REFERENCE_VIEW = 4

#: Colour error is drawn at this gain so small differences stay visible.
ERROR_GAIN = 4
HOLE_COLOR = np.array((0, 120, 255), dtype=np.uint8)
UNCOVERED_COLOR = np.array((16, 18, 24), dtype=np.uint8)

#: One DS analysis at a time: it is CPU- and memory-heavy, and runs beside the GPU job.
DS_LOCK = threading.Lock()


def ds_available() -> bool:
    """Whether the `ds_m0` package can be imported in this environment."""
    try:
        import ds_m0  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure just hides the DS panels
        return False
    return True


# ---------------------------------------------------------------- scoring


def _ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    c1, c2, sigma = (0.01 * 255) ** 2, (0.03 * 255) ** 2, 1.5
    mu_a, mu_b = gaussian_filter(a, sigma), gaussian_filter(b, sigma)
    var_a = gaussian_filter(a * a, sigma) - mu_a**2
    var_b = gaussian_filter(b * b, sigma) - mu_b**2
    cov = gaussian_filter(a * b, sigma) - mu_a * mu_b
    numerator = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    return numerator / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def score_render(rgb: np.ndarray, coverage: np.ndarray, depth: np.ndarray | None, view) -> dict:
    """Compare one Presenter render with the source view rendered from the same camera.

    Over the pixels the source marks valid: `coverage` is the share the render fills;
    PSNR, SSIM and depth error are measured on covered pixels only, so holes are
    reported once, by coverage, instead of dragging the colour scores down.
    """
    from scipy.ndimage import binary_erosion

    valid = view.depth_valid
    covered = coverage & valid
    result: dict[str, Any] = {"coverage": float(covered.sum() / max(int(valid.sum()), 1))}
    if not covered.any():
        return result
    diff = rgb.astype(np.float64) - view.rgb.astype(np.float64)
    mse = float((diff[covered] ** 2).mean())
    result["psnr"] = float(10 * np.log10(255**2 / max(mse, 1e-12)))
    # Holes take the reference value so they do not bleed into neighbouring windows,
    # and SSIM is averaged away from hole borders.
    filled = np.where(covered[..., None], rgb, view.rgb)
    inner = binary_erosion(covered, iterations=3)
    ssim = _ssim_map(_luma(filled), _luma(view.rgb))
    result["ssim"] = float(ssim[inner if inner.any() else covered].mean())
    if depth is not None:
        d, ref = depth[covered], view.depth[covered]
        finite = np.isfinite(d) & (ref > 0)
        if finite.any():
            rel = np.abs(d[finite] - ref[finite]) / ref[finite]
            result["depth_rel_median"] = float(np.median(rel))
    return result


def error_image(rgb: np.ndarray, coverage: np.ndarray, view) -> np.ndarray:
    """Red = colour error (x ERROR_GAIN), blue = hole where the source has content."""
    err = np.abs(rgb.astype(np.int16) - view.rgb.astype(np.int16)).max(axis=-1)
    out = np.broadcast_to(UNCOVERED_COLOR, rgb.shape).copy()
    out[..., 0] = np.where(coverage, np.clip(err * ERROR_GAIN, 0, 255), out[..., 0])
    out[~coverage & view.depth_valid] = HOLE_COLOR
    return out


def render_image(rgb: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """A Presenter render for display: uncovered pixels in a flat dark colour."""
    return np.where(coverage[..., None], rgb, UNCOVERED_COLOR).astype(np.uint8)


def _mean(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if key in row]
    return float(np.mean(values)) if values else None


def summarize(per_view: list[dict], presenter: str, key: str) -> dict:
    """Mean of each metric over the 8 outer views (the reference view is trivially exact)."""
    rows = [
        row[presenter][key]
        for row in per_view
        if key in row.get(presenter, {}) and row["view"] != REFERENCE_VIEW
    ]
    return {m: _mean(rows, m) for m in ("coverage", "psnr", "ssim", "depth_rel_median")}


# ---------------------------------------------------------------- the analysis


@dataclass
class DSAnalysis:
    """The DS-Image of one job, its renders at the 9 source cameras, and their scores."""

    codec: str
    state: str = "running"  # running | done | error
    stage: str = "starting"
    progress: float = 0.0
    error: str | None = None
    timings: dict[str, float] = field(default_factory=dict)
    file_metrics: dict[str, Any] = field(default_factory=dict)
    sparsity: dict[str, Any] = field(default_factory=dict)
    roundtrip: dict[str, Any] = field(default_factory=dict)
    legacy_jpeg_ok: bool | None = None
    h5_bytes: int | None = None
    per_view: list[dict] = field(default_factory=list)  # {"view", "mesh": {...}, "points": {...}}
    loo_state: str = "idle"  # idle | running | done | error
    loo_progress: float = 0.0
    loo_error: str | None = None
    images: dict[str, bytes] = field(default_factory=dict)  # "render/mesh/3", "layer1_rgb", ...
    ds_path: Path | None = None
    asset: Any = None  # the DSAsset read back from the file, for free-view renders
    source: Any = None  # the ds_m0 SourceDataset, kept for leave-one-out
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def report(self, stage: str, progress: float) -> None:
        """Record the current stage and overall progress (0..1)."""
        with self._lock:
            self.stage, self.progress = stage, progress

    def status(self) -> dict[str, Any]:
        """A JSON-serializable snapshot for the client to poll."""
        with self._lock:
            return {
                "codec": self.codec,
                "state": self.state,
                "stage": self.stage,
                "progress": round(self.progress, 3),
                "error": self.error,
                "timings": {k: round(v, 2) for k, v in self.timings.items()},
                "file": self.file_metrics,
                "sparsity": self.sparsity,
                "h5_bytes": self.h5_bytes,
                "roundtrip": self.roundtrip,
                "legacy_jpeg_ok": self.legacy_jpeg_ok,
                "per_view": self.per_view,
                "presenters": list(PRESENTERS),
                "summary": {
                    presenter: {key: summarize(self.per_view, presenter, key) for key in SCORE_KEYS}
                    for presenter in PRESENTERS
                },
                "loo": {"state": self.loo_state, "progress": round(self.loo_progress, 3),
                        "error": self.loo_error},
                "median_depth": median_depth(self),
                "width": self.asset.spatial_width if self.asset is not None else None,
                "height": self.asset.spatial_height if self.asset is not None else None,
            }

    def cleanup(self) -> None:
        """Release the asset, source views, previews and the temp .ds.jpg."""
        self.asset = None
        self.source = None
        self.images = {}
        if self.ds_path is not None:
            shutil.rmtree(self.ds_path.parent, ignore_errors=True)
            self.ds_path = None


def source_from_result(result) -> Any:
    """A `ds_m0` SourceDataset straight from in-memory spatial photo arrays."""
    from ds_m0.geometry import camera_math
    from ds_m0.io.source_reader import normalize_source_camera
    from ds_m0.model.source_dataset import SourceDataset, SourceView

    n, h, w = result.rgb.shape[:3]
    views = []
    for i in range(n):
        depth = np.asarray(result.depth[i], np.float32)
        valid = (result.mask[i] > 0) & camera_math.valid_depth(depth)
        camera = normalize_source_camera(
            np.asarray(result.K[i], np.float64), np.asarray(result.R[i], np.float64),
            np.asarray(result.C[i], np.float64), w, h)
        rgb = np.ascontiguousarray(result.rgb[i], np.uint8)
        views.append(SourceView(i, rgb, depth, valid, camera))
    return SourceDataset(w, h, views, "meter")


def _layer_previews(asset) -> dict[str, bytes]:
    l0, l1 = asset.layers
    blank = np.zeros_like(l0.rgb)
    both = np.stack([l0.depth, l1.depth])
    valid = np.stack([l0.valid, l1.valid]).astype(np.uint8)
    depth0, depth1 = depth_to_rgb(both, valid)  # shared colour scale for both layers
    mask = np.where(l1.valid[..., None], np.array((90, 169, 255), np.uint8), np.uint8(16))
    return {
        "layer0_rgb": encode_jpeg(l0.rgb),
        "layer0_depth": encode_jpeg(depth0),
        "layer1_rgb": encode_jpeg(np.where(l1.valid[..., None], l1.rgb, blank).astype(np.uint8)),
        "layer1_depth": encode_jpeg(depth1),
        "layer1_mask": encode_jpeg(np.broadcast_to(mask, l0.rgb.shape).astype(np.uint8)),
    }


def _layer0_only(asset):
    stripped = copy.deepcopy(asset)
    stripped.layers[1].valid[:] = False
    stripped.layers[1].rgb[:] = 0
    stripped.layers[1].depth[:] = 0
    return stripped


def run_analysis(analysis: DSAnalysis, result, h5_bytes: int | None = None) -> None:
    """Build, write and read back the DS-Image of `result`, then render and score it.

    Never raises: a failure is recorded on `analysis` as `state="error"`.
    """
    temp_dir: Path | None = None
    try:
        from ds_m0.config.m0_config import BuilderConfig, DSImageWriteConfig, PresenterConfig
        from ds_m0.creator.asset_builder import build_ds_asset
        from ds_m0.evaluation.file_metrics import measure_ds_image
        from ds_m0.evaluation.reports import legacy_jpeg_check
        from ds_m0.evaluation.roundtrip_metrics import ComparisonConfig, compare_assets
        from ds_m0.evaluation.sparsity_metrics import measure_layer1
        from ds_m0.io.dsimage_reader import read_ds_image
        from ds_m0.io.dsimage_writer import write_ds_image
        from ds_m0.presenter.renderer import render

        with DS_LOCK:
            tick = time.perf_counter
            analysis.h5_bytes = h5_bytes
            analysis.report("building", 0.05)
            t = tick()
            source = source_from_result(result)
            built = build_ds_asset(source, BuilderConfig(base_view_id=REFERENCE_VIEW))
            analysis.timings["build"] = tick() - t

            analysis.report("writing", 0.3)
            temp_dir = Path(tempfile.mkdtemp(prefix="sharp-webui-ds-"))
            ds_path = temp_dir / "ds_image.jpg"
            t = tick()
            write_ds_image(built, ds_path, DSImageWriteConfig(depth_codec=analysis.codec))
            analysis.timings["write"] = tick() - t
            t = tick()
            asset = read_ds_image(ds_path)
            analysis.timings["read"] = tick() - t

            analysis.report("checking", 0.38)
            analysis.file_metrics = measure_ds_image(ds_path).to_dict()
            analysis.sparsity = measure_layer1(asset.layers[1]).to_dict()
            depth_tol = 0.0
            if analysis.codec != "float32_zlib":  # same tolerance rule as `ds_m0 evaluate`
                depths = [layer.depth[layer.valid] for layer in built.layers if layer.valid.any()]
                span = float(max(d.max() for d in depths) - min(d.min() for d in depths))
                depth_tol = span / 65535.0 + 1e-6 * float(built.layers[0].depth.max())
            report = compare_assets(built, asset, ComparisonConfig(depth_abs_tolerance=depth_tol))
            analysis.roundtrip = {
                "passed": bool(report.get("passed")),
                "layer0_rgb_psnr": report.get("layer0_rgb", {}).get("psnr_db"),
                "layer0_depth_max_abs":
                    report.get("depth", {}).get("layer0", {}).get("max_abs_error"),
                "failures": report.get("failures", []),
            }
            analysis.legacy_jpeg_ok = bool(legacy_jpeg_check(ds_path)["passed"])
            analysis.images.update(_layer_previews(asset))

            stripped = _layer0_only(asset)
            t = tick()
            for i, view in enumerate(source.views):
                analysis.report("rendering", 0.4 + 0.58 * i / len(source.views))
                row: dict[str, Any] = {"view": view.view_id}
                for presenter in PRESENTERS:
                    config = PresenterConfig(mode=presenter)
                    direct = render(built, view.camera, config)
                    from_file = render(asset, view.camera, config)
                    layer0 = render(stripped, view.camera, config)
                    row[presenter] = {
                        "presenter_only": score_render(
                            direct.rgb, direct.coverage, direct.depth, view),
                        "ds_image": score_render(
                            from_file.rgb, from_file.coverage, from_file.depth, view),
                        "layer0_only": score_render(
                            layer0.rgb, layer0.coverage, layer0.depth, view),
                    }
                    analysis.images[f"render/{presenter}/{view.view_id}"] = encode_jpeg(
                        render_image(from_file.rgb, from_file.coverage))
                    analysis.images[f"error/{presenter}/{view.view_id}"] = encode_jpeg(
                        error_image(from_file.rgb, from_file.coverage, view))
                analysis.per_view.append(row)
            analysis.timings["render_and_score"] = tick() - t

            analysis.asset = asset
            analysis.source = source
            analysis.ds_path = ds_path
            temp_dir = None  # ownership passes to the analysis
            analysis.state = "done"
            analysis.report("done", 1.0)
    except BaseException as exc:  # noqa: BLE001 - surfaced to the client verbatim
        analysis.error = f"{type(exc).__name__}: {exc}"
        analysis.state = "error"
        analysis.report("error", analysis.progress)
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_leave_one_out(analysis: DSAnalysis) -> None:
    """For each outer view, rebuild the DS-Image without it and score the render there.

    The honest novel-view test: the asset being rendered has never seen the view it is
    compared with. Each rebuild goes through the writer and reader, like a real file.
    """
    try:
        from ds_m0.config.m0_config import BuilderConfig, DSImageWriteConfig, PresenterConfig
        from ds_m0.creator.asset_builder import build_ds_asset
        from ds_m0.io.dsimage_reader import read_ds_image
        from ds_m0.io.dsimage_writer import write_ds_image
        from ds_m0.model.source_dataset import SourceDataset
        from ds_m0.presenter.renderer import render

        source = analysis.source
        if source is None:
            raise RuntimeError("this DS analysis has been released; rebuild it first")
        with DS_LOCK:
            outer = [v for v in source.views if v.view_id != REFERENCE_VIEW]
            started = time.perf_counter()
            for n, view in enumerate(outer):
                analysis.loo_progress = n / len(outer)
                subset = SourceDataset(source.width, source.height,
                                       [v for v in source.views if v.view_id != view.view_id],
                                       source.depth_unit)
                with tempfile.TemporaryDirectory(prefix="sharp-webui-loo-") as tmp:
                    path = Path(tmp) / "loo.jpg"
                    rebuilt = build_ds_asset(subset, BuilderConfig(base_view_id=REFERENCE_VIEW),
                                             expected_view_count=None)
                    write_ds_image(rebuilt, path, DSImageWriteConfig(depth_codec=analysis.codec))
                    held_out = read_ds_image(path)
                for presenter in PRESENTERS:
                    res = render(held_out, view.camera, PresenterConfig(mode=presenter))
                    for row in analysis.per_view:
                        if row["view"] == view.view_id:
                            row[presenter]["leave_one_out"] = score_render(
                                res.rgb, res.coverage, res.depth, view)
            analysis.timings["leave_one_out"] = time.perf_counter() - started
            analysis.loo_progress = 1.0
            analysis.loo_state = "done"
    except BaseException as exc:  # noqa: BLE001 - surfaced to the client verbatim
        analysis.loo_error = f"{type(exc).__name__}: {exc}"
        analysis.loo_state = "error"


def render_free_view(
    analysis: DSAnalysis,
    offset_m: tuple[float, float, float],
    presenter: str = DEFAULT_PRESENTER,
) -> tuple[bytes, float]:
    """Render the DS-Image from the reference camera moved by `offset_m` (its own axes).

    Returns the JPEG and the share of the output the render covers.
    """
    from ds_m0.config.m0_config import PresenterConfig
    from ds_m0.geometry.transformation import translated_camera
    from ds_m0.presenter.renderer import render

    asset = analysis.asset
    if asset is None:
        raise FileNotFoundError("this DS-Image is no longer available; rebuild it")
    camera = translated_camera(asset.base_camera, np.asarray(offset_m, float))
    res = render(asset, camera, PresenterConfig(mode=presenter))
    return encode_jpeg(render_image(res.rgb, res.coverage)), float(res.coverage.mean())


def median_depth(analysis: DSAnalysis) -> float | None:
    """Median Layer-0 depth in metres, which sets a sensible free-view step size."""
    if analysis.asset is None:
        return None
    layer0 = analysis.asset.layers[0]
    return float(np.median(layer0.depth[layer0.valid])) if layer0.valid.any() else None
