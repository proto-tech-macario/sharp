"""Regenerate the golden artifacts. Run deliberately (and bump DATASET_ID) when behaviour changes:

    PYTHONPATH=src python tests/ds_m0/golden/generate_golden.py

Writes next to this file: reference_scene.h5 (immutable reference dataset), golden_asset/ (debug asset),
golden.ds.jpg (golden DS-Image), golden_renders/*.png (+ coverage masks) and metrics.json.
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from ds_m0.config.m0_config import BuilderConfig, DSImageWriteConfig
from ds_m0.creator import build_ds_asset
from ds_m0.evaluation import measure_coverage, measure_ds_image
from ds_m0.geometry.transformation import translated_camera
from ds_m0.io import read_ds_image, save_debug_asset, write_ds_image
from ds_m0.presenter import render
from ds_m0.presenter.output import save_render
from ds_m0.tools.synthetic import make_synthetic_arrays, make_synthetic_dataset, write_synthetic_h5

HERE = Path(__file__).parent
DATASET_ID = "synthetic-v1-96x96-baseline0.35"
OFFSETS = {"base": [0, 0, 0], "right": [0.2, 0, 0], "up": [0, -0.2, 0]}


def arrays_hash(arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def collect(out_dir: Path | None = None) -> dict:
    """Recompute every golden value from the code; optionally write the artifact files to out_dir."""
    scene = (out_dir or HERE) / "reference_scene.h5"
    if out_dir is not None:
        write_synthetic_h5(scene, 96, 96, 0.35)
    asset = build_ds_asset(make_synthetic_dataset(96, 96, 0.35), BuilderConfig())
    image = (out_dir or HERE) / "golden.ds.jpg"
    if out_dir is not None:
        write_ds_image(asset, image, DSImageWriteConfig())
        save_debug_asset(asset, out_dir / "golden_asset")
    else:
        import tempfile

        image = Path(tempfile.mkdtemp()) / "g.jpg"
        write_ds_image(asset, image, DSImageWriteConfig())
    fm = measure_ds_image(image)
    back = read_ds_image(image)
    renders = {}
    for name, off in OFFSETS.items():
        res = render(back, translated_camera(back.base_camera, off))
        if out_dir is not None:
            save_render(res, out_dir / "golden_renders", name)
        renders[name] = {"covered_pixels": measure_coverage(res.coverage).covered_pixel_count,
                         "coverage_sha256": hashlib.sha256(res.coverage.tobytes()).hexdigest()}
    return {
        "dataset_id": DATASET_ID,
        "dataset_arrays_sha256": arrays_hash(make_synthetic_arrays(96, 96, 0.35)),
        "layer0_valid_pixels": asset.layers[0].valid_pixel_count(),
        "layer1_valid_pixels": asset.layers[1].valid_pixel_count(),
        "asset_content_hash": asset.content_hash(),
        "ds_image_total_bytes": fm.ds_total_bytes,
        "ds_image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "renders": renders,
    }


if __name__ == "__main__":
    shutil.rmtree(HERE / "golden_renders", ignore_errors=True)
    shutil.rmtree(HERE / "golden_asset", ignore_errors=True)
    metrics = collect(HERE)
    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
