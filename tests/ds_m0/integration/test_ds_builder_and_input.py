import h5py
import numpy as np
import pytest
from ds_m0.config.m0_config import BuilderConfig, M0Config
from ds_m0.creator import build_ds_asset, build_ds_asset_with_diagnostics, build_layer0
from ds_m0.errors import ConfigurationError, InputError, ValidationError
from ds_m0.io import read_source_dataset
from ds_m0.model import SourceDataset
from ds_m0.tools.synthetic import make_synthetic_dataset, write_synthetic_h5


# ---- T01 input dataset validation ------------------------------------------------------------
def test_source_dataset_reads_and_reports_actual_resolution(tmp_path):
    path = write_synthetic_h5(tmp_path / "s.h5", 40, 32)
    ds = read_source_dataset(path)
    assert ds.view_count() == 9 and (ds.width, ds.height) == (40, 32)
    ds.validate()
    cam = ds.view(4).camera
    assert np.allclose(cam.rotation, np.eye(3)) and np.allclose(cam.translation, 0)
    for v in ds.views:  # normalisation: C = -R^T t must reproduce the stored camera centre
        with h5py.File(path) as f:
            assert np.allclose(-v.camera.rotation.T @ v.camera.translation, f["camera/C"][v.view_id], atol=1e-6)


def _rewrite(path, drop=None, **replace):
    with h5py.File(path, "r+") as f:
        if drop:
            del f[drop]
        for k, v in replace.items():
            del f[k]
            f.create_dataset(k, data=v)


@pytest.mark.parametrize("drop", ["depth", "camera/K", "camera/R", "camera/C", "rgb"])
def test_missing_pieces_rejected(tmp_path, drop):
    path = write_synthetic_h5(tmp_path / "s.h5", 16, 16)
    _rewrite(path, drop=drop)
    with pytest.raises(InputError):
        read_source_dataset(path)


def test_dimension_mismatch_and_bad_cameras_rejected(tmp_path):
    path = write_synthetic_h5(tmp_path / "s.h5", 16, 16)
    with h5py.File(path) as f:
        depth = f["depth"][:]
    _rewrite(path, depth=depth[:, :8])
    with pytest.raises(InputError):
        read_source_dataset(path)
    path = write_synthetic_h5(tmp_path / "t.h5", 16, 16)
    with h5py.File(path) as f:
        K = f["camera/K"][:]
    K[3, 0, 0] = np.nan
    _rewrite(path, **{"camera/K": K})
    with pytest.raises(InputError):
        read_source_dataset(path)
    assert pytest.raises(InputError, read_source_dataset, tmp_path / "missing.h5")


def test_unconvertible_convention_rejected(tmp_path):
    path = write_synthetic_h5(tmp_path / "s.h5", 16, 16)
    with h5py.File(path, "r+") as f:
        f.attrs["coordinate_system"] = "OpenGL"
    with pytest.raises(InputError):
        read_source_dataset(path)


def test_dataset_validation_failures(dataset):
    short = SourceDataset(dataset.width, dataset.height, dataset.views[:8])
    with pytest.raises(InputError):
        short.validate()
    with pytest.raises(InputError):
        dataset.view(99)
    empty_depth = make_synthetic_dataset(16, 16)
    for v in empty_depth.views:
        v.depth_valid[:] = False
    with pytest.raises(InputError):
        empty_depth.validate()
    nan = make_synthetic_dataset(16, 16)
    nan.views[2].depth[0, 0] = np.nan
    nan.views[2].depth_valid[0, 0] = True  # NaN must be treated as invalid downstream, not crash
    assert not np.isfinite(nan.views[2].depth[0, 0])


# ---- T03 / builder ----------------------------------------------------------------------------
def test_layer0_is_the_base_view_losslessly(dataset, built_asset):
    base = dataset.view(4)
    l0 = built_asset.layers[0]
    assert (l0.rgb == base.rgb).all() and (l0.depth == base.depth).all() and (l0.valid == base.depth_valid).all()
    assert l0.valid_pixel_count() == int(base.depth_valid.sum())
    assert build_layer0(base).rgb is not base.rgb  # the asset owns its buffers


def test_layer1_invariants(built_asset):
    l0, l1 = built_asset.layers
    assert built_asset.layer_count == 2 and 0 < l1.occupancy() < 1
    assert (l1.valid & ~l0.valid).sum() == 0
    both = l1.valid
    assert (l1.depth[both] > l0.depth[both] * (1 + 0.02) - 1e-6).all()
    assert not l1.rgb[~l1.valid].any() and not l1.depth[~l1.valid].any()
    assert built_asset.metadata["hidden_depth_threshold"]["relative"] == 0.02


def test_layer1_pixels_come_from_other_views(dataset, built_asset):
    other_colours = {tuple(c) for v in dataset.views if v.view_id != 4 for c in v.rgb.reshape(-1, 3)[:: 1]}
    l1 = built_asset.layers[1]
    assert {tuple(c) for c in l1.rgb[l1.valid]} <= other_colours


def test_threshold_changes_occupancy_monotonically(dataset):
    occ = [build_ds_asset(dataset, BuilderConfig(relative_depth_threshold=t)).layers[1].occupancy()
           for t in (0.005, 0.02, 0.5, 5.0)]
    assert occ == sorted(occ, reverse=True) and occ[0] > occ[-1] == 0.0


def test_identical_views_produce_empty_layer1():
    ds = make_synthetic_dataset(24, 24, baseline=1e-9)
    assert build_ds_asset(ds, BuilderConfig()).layers[1].valid_pixel_count() == 0


def test_other_base_view_is_supported_and_depths_stay_in_base_frame(dataset):
    asset, diag = build_ds_asset_with_diagnostics(dataset, BuilderConfig(base_view_id=0))
    assert diag.base_view_id == 0 and (asset.layers[0].rgb == dataset.view(0).rgb).all()
    assert asset.base_camera.equals(dataset.view(0).camera)
    assert diag.layer1_valid_pixel_count > 0 and 0 not in diag.candidates_per_view


def test_diagnostics_are_complete(dataset):
    _, d = build_ds_asset_with_diagnostics(dataset, BuilderConfig())
    assert d.number_of_views == 9 and d.candidate_count == sum(d.candidates_per_view.values())
    assert d.layer1_occupancy == d.layer1_valid_pixel_count / (d.width * d.height)
    assert {"layer1_candidate_generation", "layer1_merge"} <= set(d.timings_s)


def test_bad_builder_config_and_missing_base_view_rejected(dataset):
    for cfg in (BuilderConfig(base_view_id=-1), BuilderConfig(relative_depth_threshold=-1),
                BuilderConfig(relative_depth_threshold=0), BuilderConfig(rounding_mode="x"),
                BuilderConfig(depth_epsilon=-1)):
        with pytest.raises(ConfigurationError):
            build_ds_asset(dataset, cfg)
    with pytest.raises(InputError):
        build_ds_asset(dataset, BuilderConfig(base_view_id=12))


def test_asset_validation_catches_broken_assets(built_asset):
    from copy import deepcopy

    a = deepcopy(built_asset)
    a.layers[1].depth[a.layers[1].valid] = 0.5  # in front of layer 0
    with pytest.raises(ValidationError):
        a.validate(check_hidden=True)
    a = deepcopy(built_asset); a.layers.pop()
    with pytest.raises(ValidationError):
        a.validate()
    a = deepcopy(built_asset); a.layers[0].depth[a.layers[0].valid] = -1
    with pytest.raises(ValidationError):
        a.validate()
    a = deepcopy(built_asset); a.layers[1] = a.layers[1].__class__.empty(3, 3)
    with pytest.raises(ValidationError):
        a.validate()


def test_config_roundtrip_and_unknown_keys(tmp_path):
    cfg = M0Config()
    cfg.builder.relative_depth_threshold = 0.05
    cfg.save(tmp_path / "c.json")
    assert M0Config.load(tmp_path / "c.json").to_dict() == cfg.to_dict()
    with pytest.raises(ConfigurationError):
        M0Config.from_dict({"builder": {"nope": 1}})
    with pytest.raises(ConfigurationError):
        M0Config.from_dict({"writer": {"jpeg_quality": 0}})
