import json
import struct

import numpy as np
import pytest
from ds_m0.errors import FormatError
from ds_m0.io import jumbf_container as jc
from ds_m0.io.depth_codec import DepthCodecConfig, decode_depth, encode_depth
from ds_m0.io.sparse_layer1 import SparseLayer1Payload, decode_layer1, encode_layer1, pack_mask
from ds_m0.model import Layer
from ds_m0.model.metadata import DSImageMetadata, deserialize_metadata, serialize_metadata


def layer_with(valid, w, h):
    layer = Layer.empty(w, h)
    layer.valid[:] = valid
    n = int(layer.valid.sum())
    layer.rgb[layer.valid] = (np.arange(n * 3) % 256).reshape(n, 3).astype(np.uint8)
    layer.depth[layer.valid] = 1.0 + np.arange(n, dtype=np.float32)
    return layer


def roundtrip(layer):
    out = decode_layer1(encode_layer1(layer), layer.width, layer.height)
    assert (out.valid == layer.valid).all()
    assert (out.rgb[layer.valid] == layer.rgb[layer.valid]).all()
    assert (out.depth[layer.valid] == layer.depth[layer.valid]).all()
    return out


def test_sparse_empty_layer():
    layer = Layer.empty(7, 5)
    p = encode_layer1(layer)
    assert p.valid_count == 0 and p.rgb_stream.size == 0 and p.depth_stream.size == 0
    assert len(p.mask) == 5 and not roundtrip(layer).valid.any()


def test_sparse_fully_dense_layer():
    layer = layer_with(np.ones((6, 9), bool), 9, 6)
    assert encode_layer1(layer).valid_count == 54
    roundtrip(layer)


@pytest.mark.parametrize("where", ["tl", "center", "br"])
def test_sparse_single_sample_positions(where):
    valid = np.zeros((9, 13), bool)
    y, x = {"tl": (0, 0), "center": (4, 6), "br": (8, 12)}[where]
    valid[y, x] = True
    out = roundtrip(layer_with(valid, 13, 9))
    assert out.valid[y, x] and out.valid.sum() == 1


def test_sparse_checkerboard_keeps_raster_order():
    yy, xx = np.mgrid[0:6, 0:11]
    roundtrip(layer_with((yy + xx) % 2 == 0, 11, 6))


def test_sparse_corrupt_streams_rejected():
    layer = layer_with(np.eye(6, dtype=bool), 6, 6)
    p = encode_layer1(layer)
    bad = [
        SparseLayer1Payload(p.mask, p.rgb_stream[:-1], p.depth_stream),
        SparseLayer1Payload(p.mask, p.rgb_stream, p.depth_stream[:-1]),
        SparseLayer1Payload(p.mask[:-1], p.rgb_stream, p.depth_stream),
        SparseLayer1Payload(pack_mask(np.ones((6, 6), bool)), p.rgb_stream, p.depth_stream),
    ]
    for payload in bad:
        with pytest.raises(FormatError):
            decode_layer1(payload, 6, 6)
    with pytest.raises(FormatError):  # non-zero padding bits
        decode_layer1(SparseLayer1Payload(b"\xff\xff\xff\xff\xff", p.rgb_stream, p.depth_stream), 6, 6)


def test_depth_codec_lossless_and_quantised():
    rng = np.random.default_rng(1)
    d = (rng.random((17, 23)) * 9 + 0.1).astype(np.float32)
    enc = encode_depth(d, DepthCodecConfig("float32_zlib"))
    assert (decode_depth(enc.data, enc.params) == d).all() and enc.params["lossless"]
    q = encode_depth(d, DepthCodecConfig("uint16_quant_zlib"))
    err = np.abs(decode_depth(q.data, q.params).astype(np.float64) - d)
    assert err.max() <= q.params["max_abs_error"] * 1.001 + 1e-6 and len(q.data) < len(enc.data)
    assert q.params["depth_min"] == float(d.min()) and q.params["depth_max"] == float(d.max())


def test_depth_codec_invalid_inputs():
    enc = encode_depth(np.ones((4, 4), np.float32), DepthCodecConfig())
    with pytest.raises(FormatError):
        decode_depth(enc.data[:-3], enc.params)
    with pytest.raises(FormatError):
        decode_depth(enc.data, {**enc.params, "codec": "nope"})
    with pytest.raises(FormatError):
        decode_depth(enc.data, {**enc.params, "shape": [5, 5]})
    with pytest.raises(FormatError):
        decode_depth(b"garbage", enc.params)


def test_depth_codec_is_deterministic():
    d = np.linspace(1, 5, 64, dtype=np.float32).reshape(8, 8)
    assert encode_depth(d, DepthCodecConfig()).data == encode_depth(d, DepthCodecConfig()).data


def sample_metadata(**over):
    base = dict(format_version="1.0", asset_version="1.0", spatial_width=4, spatial_height=4, base_view_id=4,
                layer_count=2, coordinate_convention="x", base_camera={}, jpeg_parameters={},
                depth_parameters={}, layer1_parameters={}, payload_descriptors={}, asset_metadata={"a": 1})
    base.update(over)
    return DSImageMetadata(**base)


def test_metadata_serialisation_is_deterministic_and_validated():
    m = sample_metadata()
    blob = serialize_metadata(m)
    assert blob == serialize_metadata(m) and deserialize_metadata(blob) == m
    obj = json.loads(blob)
    for mutate in (
        lambda o: o.pop("spatial_width"), lambda o: o.update(spatial_width=0), lambda o: o.update(layer_count=3),
        lambda o: o.update(format_version="2.0"), lambda o: o.update(base_camera=[]),
    ):
        o = json.loads(blob)
        mutate(o)
        with pytest.raises(FormatError):
            deserialize_metadata(json.dumps(o).encode())
    for junk in (b"\xff\xfe", b"[]", b"{"):
        with pytest.raises(FormatError):
            deserialize_metadata(junk)
    assert obj["asset_metadata"] == {"a": 1}


def fake_jpeg():
    from ds_m0.io.jpeg_codec import JPEGConfig, encode_rgb

    return encode_rgb(np.full((8, 8, 3), 100, np.uint8), JPEGConfig())


def test_container_roundtrip_with_multi_packet_payload():
    big = bytes(range(256)) * 1000  # 256 kB -> several 64 kB APP11 packets
    w = jc.DSContainerWriter().create()
    w.add_primary_jpeg(fake_jpeg())
    w.add_ds_payload("metadata", "json", b'{"x":1}')
    w.add_ds_payload("blob", "bidb", big)
    blob = w.finalize()
    c = jc.DSContainerReader.parse(blob)
    assert c.payloads["blob"] == ("bidb", big) and c.payloads["metadata"] == ("json", b'{"x":1}')
    assert c.primary_jpeg == fake_jpeg() and c.app11_bytes == len(blob) - len(c.primary_jpeg)
    # every APP11 segment must respect the 64 KiB JPEG segment limit
    for marker, s, e in jc._segments(blob):
        assert e - s <= 65537


def test_container_rejects_damage():
    w = jc.DSContainerWriter().create()
    w.add_primary_jpeg(fake_jpeg())
    w.add_ds_payload("blob", "bidb", bytes(200_000))
    blob = w.finalize()
    segs = [s for s in jc._segments(blob) if s[0] == 0xEB]
    last = segs[-1]
    with pytest.raises(FormatError):  # a middle packet removed
        jc.DSContainerReader.parse(blob[:segs[1][1]] + blob[segs[1][2]:])
    with pytest.raises(FormatError):  # last packet removed (truncated payload)
        jc.DSContainerReader.parse(blob[:last[1]] + blob[last[2]:])
    with pytest.raises(FormatError):
        jc.DSContainerReader.parse(fake_jpeg())  # plain JPEG, not a DS-Image
    with pytest.raises(FormatError):
        jc.DSContainerReader.parse(b"not a jpeg")
    with pytest.raises(FormatError):
        jc.parse_boxes(struct.pack(">I", 99) + b"jumb")
