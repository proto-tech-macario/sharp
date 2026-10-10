import json
import zlib
from copy import deepcopy

import numpy as np
import pytest
from ds_helpers import make_asset
from ds_m0.config.m0_config import DSImageReadConfig, DSImageWriteConfig
from ds_m0.errors import FormatError
from ds_m0.evaluation import compare_assets, measure_ds_image
from ds_m0.evaluation.roundtrip_metrics import ComparisonConfig
from ds_m0.io import read_container, read_ds_image, write_ds_image
from ds_m0.io.jumbf_container import DSContainerWriter


def rt(asset, tmp_path, **cfg):
    path = tmp_path / "a.jpg"
    res = write_ds_image(asset, path, DSImageWriteConfig(**cfg))
    return read_ds_image(path), path, res


def lossless_report(a, b, mean_tol=12.0):
    return compare_assets(a, b, ComparisonConfig(rgb_mean_abs_tolerance=mean_tol))


def test_roundtrip_builder_asset(built_asset, tmp_path):
    b, path, res = rt(built_asset, tmp_path)
    rep = lossless_report(built_asset, b)
    assert rep["passed"], rep
    assert rep["mask_mismatch_count"] == {"layer0": 0, "layer1": 0} and rep["layer1_rgb_mismatch_count"] == 0
    assert rep["depth"]["layer0"]["max_abs_error"] == 0.0 and rep["depth"]["layer1"]["max_abs_error"] == 0.0
    assert b.base_camera.equals(built_asset.base_camera) and b.metadata == built_asset.metadata
    assert b.content_hash() != "" and res.total_bytes == path.stat().st_size


def test_roundtrip_empty_sparse_and_dense_layer1(tmp_path):
    for name, pts in (
        ("empty", []),
        ("sparse", [(1, 1, 5.0, (1, 2, 3)), (7, 7, 6.5, (250, 0, 9))]),
        ("dense", [(x, y, 3.0 + 0.01 * (x + y), ((x * 30) % 256, (y * 30) % 256, 7)) for x in range(8) for y in range(8)]),
    ):
        a = make_asset(layer1_points=pts)
        b, _, _ = rt(a, tmp_path, jpeg_quality=100, jpeg_subsampling="4:4:4")
        assert (b.layers[1].valid == a.layers[1].valid).all(), name
        assert lossless_report(a, b)["passed"], name
        assert b.layers[1].valid_pixel_count() == len(pts)


@pytest.mark.parametrize("w,h", [(1, 1), (7, 5), (16, 33), (64, 8)])
def test_roundtrip_varying_sizes(w, h, tmp_path):
    a = make_asset(w, h, layer1_points=[(w - 1, h - 1, 4.0, (9, 9, 9))] if w * h > 1 else [])
    b, _, _ = rt(a, tmp_path, jpeg_quality=95, jpeg_subsampling="4:4:4")
    assert (b.spatial_width, b.spatial_height) == (w, h) and (b.layers[1].valid == a.layers[1].valid).all()


def test_roundtrip_mixed_depth_ranges_with_invalid_layer0_pixels(tmp_path):
    a = make_asset(layer1_points=[(2, 2, 900.0, (1, 1, 1))])
    a.layers[0].depth[:] = np.geomspace(0.01, 500.0, 64, dtype=np.float32).reshape(8, 8)
    a.layers[0].valid[0, :3] = False
    a.layers[0].depth[0, :3] = 0.0
    a.layers[0].valid[5, 5] = False  # invalid but with a positive stored depth: validity must still survive
    b, _, _ = rt(a, tmp_path)
    assert (b.layers[0].valid == a.layers[0].valid).all()
    assert (b.layers[0].depth[b.layers[0].valid] == a.layers[0].depth[a.layers[0].valid]).all()


def test_quantised_depth_error_is_bounded_and_file_is_smaller(built_asset, tmp_path):
    b, path_q, _ = rt(built_asset, tmp_path, depth_codec="uint16_quant_zlib")
    lossless_size = (rt(built_asset, tmp_path / "..", depth_codec="float32_zlib")[1]).stat().st_size
    span = float(built_asset.layers[0].depth.max() - built_asset.layers[0].depth[built_asset.layers[0].valid].min())
    rep = compare_assets(built_asset, b, ComparisonConfig(depth_abs_tolerance=span / 65535))
    assert rep["passed"], rep
    assert 0 < rep["depth"]["layer0"]["max_abs_error"] <= span / 65535 / 2 * 1.01
    assert path_q.stat().st_size < lossless_size


def test_layer0_rgb_error_is_reported(built_asset, tmp_path):
    b, _, _ = rt(built_asset, tmp_path, jpeg_quality=60)
    rep = lossless_report(built_asset, b)
    assert rep["layer0_rgb"]["max_abs_error"] > 0 and rep["layer0_rgb"]["psnr_db"] > 20


def test_file_metrics_reconcile_with_actual_size(built_asset, tmp_path):
    _, path, res = rt(built_asset, tmp_path)
    fm = measure_ds_image(path)
    parts = [fm.primary_jpeg_bytes, fm.metadata_bytes, fm.base_depth_bytes, fm.layer0_mask_bytes,
             fm.layer1_mask_bytes, fm.layer1_rgb_bytes, fm.layer1_depth_bytes, fm.container_overhead_bytes]
    assert sum(parts) == fm.ds_total_bytes == path.stat().st_size
    assert 0 < fm.container_overhead_bytes < 0.1 * fm.ds_total_bytes
    assert res.component_bytes["layer1_rgb"] == fm.layer1_rgb_bytes and res.component_bytes["primary_jpeg"] == fm.primary_jpeg_bytes


# ---- corrupted / malformed files ---------------------------------------------------------------
@pytest.fixture()
def good_file(built_asset, tmp_path):
    path = tmp_path / "good.jpg"
    write_ds_image(built_asset, path)
    return path


def rewrite(path, out, mutate_payloads=None, mutate_meta=None):
    contents, meta = read_container(path)
    payloads = {k: v for k, v in contents.payloads.items()}
    if mutate_meta:
        obj = json.loads(payloads["metadata"][1])
        mutate_meta(obj)
        payloads["metadata"] = ("json", json.dumps(obj, sort_keys=True).encode())
    if mutate_payloads:
        mutate_payloads(payloads)
    w = DSContainerWriter().create()
    w.add_primary_jpeg(contents.primary_jpeg)
    for label, (kind, data) in payloads.items():
        w.add_ds_payload(label, kind, data)
    out.write_bytes(w.finalize())
    return out


def test_truncated_files_never_produce_an_asset(good_file, tmp_path):
    data = good_file.read_bytes()
    for frac in (0.02, 0.1, 0.3, 0.5, 0.8, 0.95, 0.999):
        p = tmp_path / f"t{frac}.jpg"
        p.write_bytes(data[: int(len(data) * frac)])
        with pytest.raises(FormatError):
            read_ds_image(p)


def test_altered_metadata_rejected(good_file, tmp_path):
    cases = {
        "width": lambda o: o.update(spatial_width=o["spatial_width"] + 1),
        "count": lambda o: o["layer1_parameters"].update(valid_count=o["layer1_parameters"]["valid_count"] + 1),
        "codec": lambda o: o["depth_parameters"]["layer0"].update(codec="lzw9000"),
        "descriptor_len": lambda o: o["payload_descriptors"]["layer1_rgb"].update(length=1),
        "version": lambda o: o.update(format_version="9.0"),
        "layers": lambda o: o.update(layer_count=1),
        "shape": lambda o: o["depth_parameters"]["layer0"].update(shape=[3, 3]),
        "camera": lambda o: o["base_camera"].update(fx=-1.0),
    }
    for name, mutate in cases.items():
        p = rewrite(good_file, tmp_path / f"{name}.jpg", mutate_meta=mutate)
        with pytest.raises(FormatError):
            read_ds_image(p)


def test_corrupted_payloads_rejected(good_file, tmp_path):
    def sized(label, fn):
        def go(payloads):
            kind, data = payloads[label]
            payloads[label] = (kind, fn(data))
        return go

    def fix_meta(label):
        def go(o):
            data = o_payloads[label][1]
            o["payload_descriptors"][label].update(length=len(data), crc32=zlib.crc32(data) & 0xFFFFFFFF)
        return go

    # flipped bit: CRC must catch it (and the bypass flag must still fail structurally or decode)
    for label in ("layer1_rgb", "layer1_depth", "layer1_mask", "layer0_depth", "layer0_mask"):
        p = rewrite(good_file, tmp_path / f"flip_{label}.jpg",
                    mutate_payloads=sized(label, lambda d: d[:5] + bytes([d[5] ^ 1]) + d[6:]))
        with pytest.raises(FormatError):
            read_ds_image(p)
    # truncated streams with *consistent* descriptors: the codec/length checks must catch them
    for label in ("layer1_rgb", "layer1_depth", "layer1_mask"):
        contents, _ = read_container(good_file)
        o_payloads = dict(contents.payloads)
        cut = o_payloads[label][1][:-4]
        p = rewrite(good_file, tmp_path / f"cut_{label}.jpg",
                    mutate_payloads=lambda pl, label=label, cut=cut: pl.__setitem__(label, ("bidb", cut)),
                    mutate_meta=lambda o, label=label, cut=cut: o["payload_descriptors"][label].update(
                        length=len(cut), crc32=zlib.crc32(cut) & 0xFFFFFFFF))
        with pytest.raises(FormatError):
            read_ds_image(p)
    # missing payload
    p = rewrite(good_file, tmp_path / "missing.jpg", mutate_payloads=lambda pl: pl.pop("layer1_depth"))
    with pytest.raises(FormatError):
        read_ds_image(p)
    # skipping CRC verification must not make a damaged file silently "work" with wrong structure
    p = rewrite(good_file, tmp_path / "nocrc.jpg", mutate_payloads=sized("layer1_mask", lambda d: d[:-2]))
    with pytest.raises(FormatError):
        read_ds_image(p, DSImageReadConfig(verify_checksums=False))


def test_jpeg_primary_dimension_mismatch_rejected(good_file, tmp_path):
    p = rewrite(good_file, tmp_path / "dims.jpg", mutate_meta=lambda o: (
        o.update(spatial_width=10, spatial_height=10),
        o["base_camera"].update(width=10, height=10)))
    with pytest.raises(FormatError):
        read_ds_image(p)


def test_plain_jpeg_and_garbage_are_not_ds_images(tmp_path):
    from ds_m0.io.jpeg_codec import JPEGConfig, encode_rgb

    plain = tmp_path / "plain.jpg"
    plain.write_bytes(encode_rgb(np.zeros((8, 8, 3), np.uint8), JPEGConfig()))
    for blob in (plain.read_bytes(), b"hello", b""):
        p = tmp_path / "x.jpg"
        p.write_bytes(blob)
        with pytest.raises(FormatError):
            read_ds_image(p)
    with pytest.raises(FormatError):
        read_ds_image(tmp_path / "does_not_exist.jpg")


def test_writer_never_mutates_asset_and_cleans_tmp(built_asset, tmp_path):
    before = built_asset.content_hash()
    snapshot = deepcopy(built_asset.layers[1].depth)
    write_ds_image(built_asset, tmp_path / "sub" / "x.jpg")
    assert built_asset.content_hash() == before and (built_asset.layers[1].depth == snapshot).all()
    assert not list((tmp_path / "sub").glob("*.tmp"))
    bad = deepcopy(built_asset)
    bad.layers[0].depth[bad.layers[0].valid] = np.nan
    with pytest.raises(Exception):
        write_ds_image(bad, tmp_path / "bad.jpg")
    assert not (tmp_path / "bad.jpg").exists()
