"""MIV correctness validation against the source spatial sequence (spec §33.3).

Decodes the MIV file with the independent reference decoder and checks:
frame count, frame rate, duration and per-frame timing (via the manifest),
the 9-view structure (count, resolution), camera metadata per frame and view
(K, R, C after converting back to Stage 1 conventions), and RGB/depth
correspondence, and reports file size and bitrate. Collects every failure
instead of stopping at the first.

RGB/depth correspondence is judged on MIV's own terms: MIV codes only
occupied samples, so texture and depth are compared where both the decoded
occupancy and the source mask are valid, and the two masks must agree.
Texture fidelity is luma PSNR, the usual measure for video coding (4:2:0 chroma
subsampling alone costs several dB of RGB PSNR on saturated edges).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .decode import decode_miv
from .tmiv import TmivInstall, run_logged
from .tmiv_config import VIEW_NAMES

_LUMA = np.array([0.2126, 0.7152, 0.0722])  # BT.709, as used for the texture video


@dataclass(frozen=True)
class Thresholds:
    """Tolerances used by `compare_to_reference`."""

    k_rel: float = 1e-3
    r_abs: float = 1e-3
    c_abs: float = 1e-3  # metres
    mask_agreement: float = 0.95  # fraction of pixels where decoded and source validity agree
    depth_median_rel: float = 0.05
    # Luma PSNR on pixels valid in both. This is a correspondence gate (a misaligned or
    # missing texture scores ~6-15 dB), not a codec quality target: VVC random access
    # codes higher temporal layers at higher QP. Codec quality is reported in the stats.
    y_psnr_db: float = 25.0
    timestamp_abs: float = 1e-6  # seconds


@dataclass
class MivReport:
    """Outcome of MIV validation: every failure found, plus statistics."""

    ok: bool
    failures: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _luma_psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR (dB, 8-bit peak) of the BT.709 luma of two RGB pixel arrays."""
    ya = a.astype(np.float64) @ _LUMA
    yb = b.astype(np.float64) @ _LUMA
    mse = float(np.mean((ya - yb) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def compare_to_reference(
    decoded, reference, manifest: dict | None = None, thresholds: Thresholds = Thresholds()
) -> MivReport:
    """Compare a decoded MIV (`DecodedMiv`-like) with the reference `SequenceReader`."""
    failures: list[str] = []
    info = reference.info
    ref_count = len(reference)

    if decoded.frame_count != ref_count:
        failures.append(f"frame count: decoded {decoded.frame_count}, reference {ref_count}")
    if abs(decoded.fps - info.fps) > 1e-3:
        failures.append(f"fps: decoded {decoded.fps}, reference {info.fps}")
    if list(decoded.view_names) != list(VIEW_NAMES):
        failures.append(f"views: decoded {list(decoded.view_names)}, expected {list(VIEW_NAMES)}")
    if tuple(decoded.resolution) != (info.width, info.height):
        failures.append(
            f"resolution: decoded {tuple(decoded.resolution)}, reference "
            f"{(info.width, info.height)}"
        )

    ref_timestamps = reference.timestamps()
    if ref_count:
        ref_duration = float(ref_timestamps[-1] - ref_timestamps[0]) + 1.0 / info.fps
        decoded_duration = decoded.frame_count / decoded.fps
        if abs(decoded_duration - ref_duration) > 0.5 / info.fps:
            failures.append(
                f"duration: decoded {decoded_duration:.4f} s, reference {ref_duration:.4f} s"
            )
    if manifest is not None:
        manifest_times = [f["timestamp"] for f in manifest.get("frames", [])]
        if len(manifest_times) != ref_count:
            failures.append(
                f"frame count: manifest lists {len(manifest_times)}, reference {ref_count}"
            )
        for i, (got, want) in enumerate(zip(manifest_times, ref_timestamps)):
            if abs(got - want) > thresholds.timestamp_abs:
                failures.append(f"timestamp of frame {i}: manifest {got}, reference {want}")

    psnrs, depth_errors, agreements = [], [], []
    camera_errors = {"K": 0.0, "R": 0.0, "C": 0.0}
    for t, (views, ref_frame) in enumerate(zip(decoded.frames(), reference)):
        for name, ref in zip(VIEW_NAMES, ref_frame.views):
            if name not in views:
                failures.append(f"frame {t} {name}: view missing from the decoded frame")
                continue
            got = views[name]
            k_err = float(np.max(np.abs(got.K - ref.K) / np.maximum(np.abs(ref.K), 1.0)))
            r_err = float(np.max(np.abs(got.R - ref.R)))
            c_err = float(np.max(np.abs(got.C - ref.C)))
            camera_errors["K"] = max(camera_errors["K"], k_err)
            camera_errors["R"] = max(camera_errors["R"], r_err)
            camera_errors["C"] = max(camera_errors["C"], c_err)
            if k_err > thresholds.k_rel:
                failures.append(f"frame {t} {name}: K differs from reference by {k_err:.2e} (rel)")
            if r_err > thresholds.r_abs:
                failures.append(f"frame {t} {name}: R differs from reference by {r_err:.2e}")
            if c_err > thresholds.c_abs:
                failures.append(f"frame {t} {name}: C differs from reference by {c_err:.2e} m")

            agreement = float(np.mean(got.mask == ref.mask))
            agreements.append(agreement)
            if agreement < thresholds.mask_agreement:
                failures.append(
                    f"frame {t} {name}: validity mask agrees with the reference on only "
                    f"{agreement:.1%} of pixels (decoded {got.mask.mean():.1%} valid, reference "
                    f"{ref.mask.mean():.1%})"
                )
            both = (got.mask == 1) & (ref.mask == 1)
            if not both.any():
                continue
            rel = float(np.median(np.abs(got.depth[both] - ref.depth[both]) / ref.depth[both]))
            depth_errors.append(rel)
            if rel > thresholds.depth_median_rel:
                failures.append(
                    f"frame {t} {name}: depth median relative error {rel:.3f} exceeds "
                    f"{thresholds.depth_median_rel}"
                )
            psnr = _luma_psnr(got.rgb[both], ref.rgb[both])
            psnrs.append(psnr)
            if psnr < thresholds.y_psnr_db:
                failures.append(
                    f"frame {t} {name}: luma PSNR {psnr:.1f} dB below {thresholds.y_psnr_db} dB"
                )

    finite = [p for p in psnrs if np.isfinite(p)]
    stats = {
        "frame_count": decoded.frame_count,
        "fps": decoded.fps,
        "duration_s": decoded.frame_count / decoded.fps if decoded.fps else 0.0,
        "views": list(decoded.view_names),
        "resolution": list(decoded.resolution),
        "y_psnr_db_min": min(psnrs) if psnrs else None,
        "y_psnr_db_mean": float(np.mean(finite)) if finite else (
            float("inf") if psnrs else None),
        "depth_median_rel_error_max": max(depth_errors) if depth_errors else None,
        "depth_median_rel_error_mean": float(np.mean(depth_errors)) if depth_errors else None,
        "mask_agreement_min": min(agreements) if agreements else None,
        "camera_max_error": camera_errors,
    }
    return MivReport(ok=not failures, failures=failures, stats=stats)


def validate_miv(
    miv_path: str | Path,
    reference,
    work_dir: str | Path,
    tmiv: TmivInstall | None = None,
    thresholds: Thresholds = Thresholds(),
    runner=run_logged,
) -> MivReport:
    """Decode `miv_path` with TmivDecoder and compare it with `reference` (a SequenceReader)."""
    miv_path = Path(miv_path)
    manifest_path = Path(str(miv_path) + ".json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    frame_count = manifest["frame_count"] if manifest else len(reference)

    decoded = decode_miv(miv_path, frame_count, work_dir, tmiv=tmiv, runner=runner)
    report = compare_to_reference(decoded, reference, manifest, thresholds)
    size = miv_path.stat().st_size
    duration = report.stats["duration_s"]
    report.stats["size_bytes"] = size
    report.stats["bitrate_bps"] = size * 8.0 / duration if duration else None
    report.stats["camera_update_frames"] = decoded.camera_update_frames
    report.stats["parser_dump"] = str(decoded.parser_dump) if decoded.parser_dump else None
    if manifest is None:
        report.stats["manifest"] = "missing: per-frame timestamps were not checked"
    return report
