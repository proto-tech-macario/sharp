import ast
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import ds_m0
import numpy as np
import pytest
from ds_m0.config.m0_config import BuilderConfig, DSImageWriteConfig
from ds_m0.creator import build_ds_asset
from ds_m0.geometry.transformation import translated_camera
from ds_m0.io import read_ds_image, write_ds_image
from ds_m0.model import DSAsset, Layer, SourceDataset
from ds_m0.presenter import render
from ds_m0.tools.command_line import main as cli
from ds_m0.tools.synthetic import make_synthetic_arrays, make_synthetic_dataset, write_synthetic_h5
from PIL import Image

PKG = Path(ds_m0.__file__).parent
GOLDEN = Path(__file__).parents[1] / "golden"


# ---- T11 determinism --------------------------------------------------------------------------
def test_build_is_deterministic_across_runs_and_view_order(dataset):
    hashes = {build_ds_asset(dataset, BuilderConfig()).content_hash() for _ in range(3)}
    shuffled = SourceDataset(dataset.width, dataset.height, list(reversed(dataset.views)))
    hashes.add(build_ds_asset(shuffled, BuilderConfig()).content_hash())
    rng = np.random.default_rng(3)
    mixed = SourceDataset(dataset.width, dataset.height, [dataset.views[i] for i in rng.permutation(9)])
    hashes.add(build_ds_asset(mixed, BuilderConfig()).content_hash())
    assert len(hashes) == 1


def test_write_read_render_are_deterministic(built_asset, tmp_path):
    cfg = DSImageWriteConfig()
    write_ds_image(built_asset, tmp_path / "a.jpg", cfg)
    write_ds_image(built_asset, tmp_path / "b.jpg", cfg)
    assert (tmp_path / "a.jpg").read_bytes() == (tmp_path / "b.jpg").read_bytes()
    b1, b2 = read_ds_image(tmp_path / "a.jpg"), read_ds_image(tmp_path / "b.jpg")
    assert b1.content_hash() == b2.content_hash()
    cam = translated_camera(b1.base_camera, [0.1, 0.05, 0.0])
    r1, r2 = render(b1, cam), render(b2, cam)
    assert (r1.rgb == r2.rgb).all() and (r1.coverage == r2.coverage).all()


# ---- architecture ------------------------------------------------------------------------------
def imports_of(rel_dir: str) -> set[str]:
    found = set()
    for f in (PKG / rel_dir).rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.ImportFrom):
                base = "." * node.level + (node.module or "")
                found.add(base)
                found.update(f"{base}.{a.name}" for a in node.names)
            elif isinstance(node, ast.Import):
                found.update(a.name for a in node.names)
    return found


def mentions(imports: set[str], *needles: str) -> list[str]:
    """Imports containing any needle as a whole dotted-name token."""
    return sorted(i for i in imports if any(n in i.split(".") for n in needles))


def test_presenter_has_no_forbidden_dependencies():
    imp = imports_of("presenter")
    assert not mentions(imp, "creator", "source_dataset", "source_reader", "sparse_layer1", "jumbf",
                        "dsimage", "torch", "gsplat", "plyfile", "h5py", "miv", "evaluation", "io", "dsimage_writer",
                        "dsimage_reader", "debug_asset")


def test_builder_does_not_know_the_container_or_presenter():
    imp = imports_of("creator")
    assert not mentions(imp, "jumbf", "jpeg", "dsimage", "PIL", "presenter", "sparse_layer1", "evaluation")


def test_geometry_and_model_are_leaf_layers():
    assert not mentions(imports_of("geometry"), "creator", "presenter", "ds_m0.io", "io", "evaluation")
    assert not mentions(imports_of("model"), "creator", "presenter", "io", "evaluation")


def test_jumbf_details_stay_inside_the_container_module():
    for f in PKG.rglob("*.py"):
        if f.name == "jumbf_container.py":
            continue
        assert "APP11" not in f.read_text() and "0xEB" not in f.read_text(), f


def test_camera_math_is_not_duplicated():
    for f in PKG.rglob("*.py"):
        if f.parent.name == "geometry" or f.name == "synthetic.py":
            continue
        text = f.read_text()
        assert "/ camera.fx" not in text and "camera.fy *" not in text, f


def test_sparse_and_dense_paths_render_identically(built_asset, tmp_path):
    write_ds_image(built_asset, tmp_path / "s.jpg")
    via_sparse = read_ds_image(tmp_path / "s.jpg")
    dense = DSAsset(via_sparse.version, via_sparse.spatial_width, via_sparse.spatial_height, via_sparse.base_camera,
                    [Layer(l.width, l.height, l.rgb.copy(), l.depth.copy(), l.valid.copy()) for l in via_sparse.layers],
                    dict(via_sparse.metadata))
    for off in ([0, 0, 0], [0.2, 0, 0], [-0.1, 0.1, 0.05]):
        cam = translated_camera(via_sparse.base_camera, off)
        a, b = render(via_sparse, cam), render(dense, cam)
        assert (a.rgb == b.rgb).all() and (a.coverage == b.coverage).all()


def test_presenter_renders_after_the_source_is_gone(tmp_path):
    src = write_synthetic_h5(tmp_path / "scene.h5", 32, 32)
    assert cli(["roundtrip", "--input", str(src), "--output-dir", str(tmp_path / "rt")]) == 0
    src.unlink()
    out = tmp_path / "r"
    assert cli(["render", "--asset", str(tmp_path / "rt" / "roundtrip.jpg"), "--output", str(out),
                "--translate", "0.1", "0", "0"]) == 0
    assert (out / "render_rgb.png").is_file() and (out / "render_coverage.png").is_file()


def test_no_ai_dependencies_are_loaded():
    code = ("import sys; from ds_m0.tools.command_line import main; "
            "from ds_m0.tools.synthetic import make_synthetic_arrays, make_synthetic_dataset, write_synthetic_h5; import tempfile, os; "
            "d=tempfile.mkdtemp(); p=str(write_synthetic_h5(os.path.join(d,'s.h5'),24,24)); "
            "assert main(['evaluate','--input',p,'--output-dir',os.path.join(d,'o')])==0; "
            "bad=[m for m in ('torch','gsplat','sharp','sharp_spatialize','tensorflow','onnxruntime') if m in sys.modules]; "
            "assert not bad, bad")
    env = {"PYTHONPATH": str(PKG.parent), "PATH": "/usr/bin:/bin"}
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ---- T12 legacy JPEG compatibility ---------------------------------------------------------------
def test_legacy_jpeg_decoders_read_the_primary_image(built_asset, tmp_path):
    path = tmp_path / "ds.jpg"
    write_ds_image(built_asset, path)
    data = path.read_bytes()
    assert data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"
    with Image.open(path) as im:
        im.load()
        assert im.format == "JPEG" and im.size == (48, 48)
        legacy = np.asarray(im.convert("RGB"))
    assert (legacy == read_ds_image(path).layers[0].rgb).all()
    if shutil.which("sips"):
        out = tmp_path / "sips.png"
        proc = subprocess.run(["sips", "-s", "format", "png", str(path), "--out", str(out)], capture_output=True)
        assert proc.returncode == 0 and Image.open(out).size == (48, 48)
    if shutil.which("file"):
        assert b"JPEG" in subprocess.run(["file", str(path)], capture_output=True).stdout


# ---- CLI + T16 real-data-style end-to-end regression against golden artifacts --------------------
def test_cli_error_codes(tmp_path):
    assert cli(["inspect", "--input", str(tmp_path / "missing.jpg")]) == 5
    assert cli(["build", "--input", str(tmp_path / "missing.h5"), "--output", str(tmp_path / "a")]) == 3
    src = write_synthetic_h5(tmp_path / "s.h5", 16, 16)
    assert cli(["build", "--input", str(src), "--output", str(tmp_path / "a"), "--base-view", "-3"]) == 2
    assert cli(["build", "--input", str(src), "--output", str(tmp_path / "a"), "--base-view", "42"]) == 3


def test_cli_stage_by_stage_matches_roundtrip(tmp_path, capsys):
    src = write_synthetic_h5(tmp_path / "s.h5", 32, 32)
    a, img, back = tmp_path / "asset", tmp_path / "x.jpg", tmp_path / "back"
    assert cli(["build", "--input", str(src), "--output", str(a)]) == 0
    assert cli(["write", "--asset", str(a), "--output", str(img)]) == 0
    assert cli(["read", "--input", str(img), "--output", str(back)]) == 0
    assert cli(["inspect", "--input", str(img)]) == 0
    assert cli(["roundtrip", "--input", str(src), "--output-dir", str(tmp_path / "rt")]) == 0
    from ds_m0.evaluation import compare_assets
    from ds_m0.io import load_debug_asset
    assert compare_assets(load_debug_asset(a), load_debug_asset(back))["passed"]
    assert cli(["export-config", "--output", str(tmp_path / "cfg.json"), "--jpeg-quality", "75"]) == 0
    assert json.loads((tmp_path / "cfg.json").read_text())["writer"]["jpeg_quality"] == 75


def test_full_evaluation_reports_and_golden_regression(tmp_path):
    src = write_synthetic_h5(tmp_path / "scene.h5", 96, 96, 0.35)
    out = tmp_path / "eval"
    miv = tmp_path / "fake.miv"
    miv.write_bytes(b"\0" * 100_000)
    assert cli(["evaluate", "--input", str(src), "--output-dir", str(out), "--miv-file", str(miv)]) == 0
    for name in ("report_a_asset", "report_b_file", "report_c_roundtrip", "report_d_render", "report_e_miv"):
        assert (out / f"{name}.json").is_file(), name
    summary = (out / "summary.txt").read_text()
    assert summary.count("PASS") == 3 and "FAIL" not in summary and "DS/MIV ratio" in summary
    rep_b = json.loads((out / "report_b_file.json").read_text())
    assert rep_b["sparsity"]["sparse_savings_bytes"] > 0 and rep_b["writer_total_matches_file"]
    rep_e = json.loads((out / "report_e_miv.json").read_text())
    assert rep_e["miv_total_size"] == 100_000 and 0 < rep_e["ds_to_miv_ratio"] < 1
    run_info = json.loads((out / "run_info.json").read_text())
    assert run_info["dataset_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()

    sys.path.insert(0, str(GOLDEN))
    try:
        import generate_golden
    finally:
        sys.path.remove(str(GOLDEN))
    golden = json.loads((GOLDEN / "metrics.json").read_text())
    now = generate_golden.collect()
    assert now["dataset_id"] == golden["dataset_id"]
    for key in ("layer0_valid_pixels", "layer1_valid_pixels", "asset_content_hash", "renders"):
        assert now[key] == golden[key], key
    # the exact file bytes depend on the Pillow/libjpeg build, so size is checked as a band
    assert abs(now["ds_image_total_bytes"] - golden["ds_image_total_bytes"]) < 0.10 * golden["ds_image_total_bytes"]


def test_golden_artifacts_on_disk(tmp_path):
    """The stored golden DS-Image, asset, renders and reference dataset must stay valid and equivalent."""
    sys.path.insert(0, str(GOLDEN))
    try:
        import generate_golden as gg
    finally:
        sys.path.remove(str(GOLDEN))
    from ds_m0.evaluation import compare_assets
    from ds_m0.io import load_debug_asset, read_source_dataset

    golden = json.loads((GOLDEN / "metrics.json").read_text())
    # reference dataset: stored file == what the generator produces now
    ref = read_source_dataset(GOLDEN / "reference_scene.h5")
    fresh = make_synthetic_dataset(96, 96, 0.35)
    for a, b in zip(ref.views, fresh.views):
        assert (a.rgb == b.rgb).all() and (a.depth == b.depth).all() and a.camera.equals(b.camera)
    assert gg.arrays_hash(make_synthetic_arrays(96, 96, 0.35)) == golden["dataset_arrays_sha256"]
    # golden DS-Image reads back, matches the golden asset, and re-renders the golden masks exactly
    image = GOLDEN / "golden.ds.jpg"
    assert hashlib.sha256(image.read_bytes()).hexdigest() == golden["ds_image_sha256"]  # the stored file is untouched
    back = read_ds_image(image)
    asset = load_debug_asset(GOLDEN / "golden_asset")
    assert asset.content_hash() == golden["asset_content_hash"]
    assert compare_assets(asset, back)["passed"]
    for name, off in gg.OFFSETS.items():
        res = render(back, translated_camera(back.base_camera, off))
        stored = np.asarray(Image.open(GOLDEN / "golden_renders" / f"{name}_coverage.png")) > 0
        assert (res.coverage == stored).all(), name
    # and the golden asset built fresh from the reference dataset is the same asset
    assert build_ds_asset(ref, BuilderConfig()).content_hash() == golden["asset_content_hash"]


def test_builder_and_presenter_agree_on_conventions(dataset, built_asset):
    """Render the DS Asset from another *source* camera: wherever it lands it must reproduce that view's
    colours, which fails on any coordinate / depth / pixel-centre mismatch between Builder and Presenter."""
    from ds_m0.config.m0_config import PresenterConfig

    cfg = PresenterConfig(depth_epsilon=1e-3)
    for vid in (0, 2, 6, 8):
        src = dataset.view(vid)
        res = render(built_asset, src.camera, cfg)
        both = res.coverage & src.depth_valid
        assert both.mean() > 0.6, vid
        close = (np.abs(res.rgb.astype(int) - src.rgb.astype(int)).max(axis=-1) <= 40) & both
        assert close.sum() / both.sum() > 0.85, (vid, close.sum() / both.sum())
        depth_ok = np.abs(res.depth[both] - src.depth[both]) < 0.1 * src.depth[both]
        assert depth_ok.mean() > 0.9, vid


@pytest.mark.parametrize("w,h", [(7, 5), (8, 8), (13, 9)])
def test_pixel_centre_convention_is_identity_at_base_camera(w, h):
    """Odd/even sizes (principal point at W/2) still map every Layer-0 sample onto itself."""
    from ds_helpers import make_asset

    asset = make_asset(w, h)
    res = render(asset, asset.base_camera)
    assert res.coverage.all() and (res.rgb == asset.layers[0].rgb).all()
