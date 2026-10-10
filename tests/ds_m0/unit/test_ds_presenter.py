import itertools

import numpy as np
from ds_helpers import make_asset, make_camera
from ds_m0.config.m0_config import PresenterConfig
from ds_m0.geometry.transformation import translated_camera
from ds_m0.presenter import DepthBuffer, render

BLUE = (0, 0, 255)


def test_identity_render_reproduces_layer0_and_ignores_hidden_layer1():
    asset = make_asset(layer1_points=[(3, 3, 4.0, BLUE), (0, 0, 9.0, BLUE)])
    res = render(asset, asset.base_camera)
    assert res.coverage.all() and (res.rgb == asset.layers[0].rgb).all()
    assert np.allclose(res.depth, 2.0)


def test_nearer_sample_beats_farther_regardless_of_order():
    buf_args = dict(x=np.array([1, 1, 1]), y=np.array([0, 0, 0]), depth=np.array([5.0, 2.0, 3.0]))
    lid, order = np.array([0, 1, 0]), np.array([0, 1, 2])
    winners = set()
    for perm in itertools.permutations(range(3)):
        p = list(perm)
        buf = DepthBuffer(4, 1, 1e-6)
        win = buf.update_many(buf_args["x"][p], buf_args["y"][p], buf_args["depth"][p], lid[p], order[p])
        assert len(win) == 1
        winners.add(int(order[p][win[0]]))
    assert winners == {1}  # the depth-2.0 sample always wins


def test_depth_ties_prefer_layer0_then_lower_raster_order():
    buf = DepthBuffer(2, 1, 1e-6)
    x, y = np.array([0, 0, 0]), np.array([0, 0, 0])
    win = buf.update_many(x, y, np.array([2.0, 2.0 + 1e-8, 2.0]), np.array([1, 0, 0]), np.array([0, 5, 3]))
    assert win.tolist() == [2]  # layer 0, order 3 beats layer 0 order 5 beats layer 1
    assert buf.test_and_update(0, 0, 2.0, (0, 1)) and not buf.test_and_update(0, 0, 2.0, (1, 0))
    assert buf.test_and_update(0, 0, 1.0, (9, 9)) and not buf.test_and_update(0, 0, 1.5, (0, 0))


def two_depth_asset():
    """Left half is a near object (depth 2) with hidden background (depth 5, blue) behind it."""
    pts = [(x, y, 5.0, BLUE) for x in range(4) for y in range(8)]
    asset = make_asset(layer1_points=pts)
    asset.layers[0].depth[:, 4:] = 5.0
    return asset


def test_small_translation_reveals_layer1_and_leaves_unsupported_uncovered():
    asset = two_depth_asset()
    cam = translated_camera(asset.base_camera, [0.5, 0, 0])
    with_l1 = render(asset, cam)
    without = make_asset()
    without.layers[0].depth[:] = asset.layers[0].depth
    without.layers[0].rgb[:] = asset.layers[0].rgb
    bare = render(without, cam)
    revealed = with_l1.coverage & ~bare.coverage
    assert revealed.any() and (with_l1.rgb[revealed] == BLUE).all()
    assert (~bare.coverage).any()  # holes exist without Layer 1 -> they are not filled
    uncovered = ~with_l1.coverage
    assert (with_l1.rgb[uncovered] == 0).all()
    # nearer Layer 0 stays in front wherever both land on the same pixel
    both = with_l1.coverage & bare.coverage
    assert (with_l1.rgb[both] == bare.rgb[both]).all()


def test_no_projected_sample_means_no_coverage():
    asset = make_asset()
    far = translated_camera(asset.base_camera, [500.0, 0, 0])
    res = render(asset, far)
    assert not res.coverage.any() and not res.rgb.any()


def test_target_camera_may_differ_in_size_and_focal_length():
    asset = make_asset()
    cam = make_camera(width=16, height=12, f=16.0)
    res = render(asset, cam)
    assert res.rgb.shape == (12, 16, 3) and res.coverage.shape == (12, 16)
    cfg = PresenterConfig(output_width=5, output_height=4)
    assert render(asset, cam, cfg).rgb.shape == (4, 5, 3)


def test_layer0_invalid_samples_are_not_rendered():
    asset = make_asset()
    asset.layers[0].valid[:, :4] = False
    res = render(asset, asset.base_camera)
    assert not res.coverage[:, :4].any() and res.coverage[:, 4:].all()
