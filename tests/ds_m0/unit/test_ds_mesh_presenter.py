"""Tests for the mesh Presenter (PresenterConfig.mode="mesh")."""

import numpy as np
import pytest
from ds_helpers import make_asset
from ds_m0.config.m0_config import PresenterConfig
from ds_m0.errors import ConfigurationError
from ds_m0.geometry.transformation import translated_camera
from ds_m0.presenter import render

MESH = PresenterConfig(mode="mesh")
RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)


def test_identity_mesh_render_reproduces_layer0_exactly():
    """At the base camera the mesh returns Layer 0 pixel for pixel."""
    asset = make_asset(layer1_points=[(3, 3, 4.0, BLUE)])
    res = render(asset, asset.base_camera, MESH)
    assert res.coverage.all() and (res.rgb == asset.layers[0].rgb).all()
    assert np.allclose(res.depth, 2.0)


def test_mesh_closes_the_pinholes_a_magnified_surface_leaves_between_points():
    """Moving closer spreads the points apart; triangles fill the gaps."""
    asset = make_asset(16, 16)
    closer = translated_camera(asset.base_camera, [0, 0, 0.6])  # 1.43x magnification
    points = render(asset, closer)
    mesh = render(asset, closer, MESH)
    assert (~points.coverage).any()  # the point splat leaves gaps
    assert mesh.coverage.sum() > points.coverage.sum()
    # every pixel inside the projected surface is covered
    ys, xs = np.nonzero(points.coverage)
    assert mesh.coverage[ys.min():ys.max() + 1, xs.min():xs.max() + 1].all()


def test_mesh_interpolates_colour_between_samples():
    """A sub-pixel shift gives interpolated colours, not copies of samples."""
    asset = make_asset(16, 16)
    ramp = np.linspace(0, 240, 16).astype(np.uint8)
    asset.layers[0].rgb[:] = ramp[None, :, None]
    shifted = translated_camera(asset.base_camera, [0.0625, 0, 0])  # a quarter pixel at depth 2
    mesh = render(asset, shifted, MESH)
    inner = mesh.rgb[4:12, 4:12, 0].astype(int)
    steps = np.diff(inner, axis=1)
    assert (steps > 0).all() and (inner % 16 != 0).any()  # smooth ramp, off the sample values


def test_mesh_does_not_stitch_across_a_depth_edge():
    """No triangle joins a near object to the far wall behind it."""
    asset = make_asset(16, 8)
    l0 = asset.layers[0]
    l0.rgb[:, :8], l0.rgb[:, 8:] = RED, GREEN
    l0.depth[:, 8:] = 6.0  # near red object on the left, far green wall on the right
    res = render(asset, translated_camera(asset.base_camera, [-0.4, 0, 0]), MESH)
    colours = {tuple(c) for c in res.rgb[res.coverage]}
    assert colours <= {RED, GREEN}  # no red/green blend from a stretched triangle
    assert (~res.coverage).any()  # the disocclusion stays a hole, no Layer 1 here


def test_mesh_depth_threshold_controls_the_break():
    """A step below the threshold is joined, one above it is broken."""
    asset = make_asset(16, 8)
    asset.layers[0].depth[:, 8:] = 2.1  # a 5% step
    cam = translated_camera(asset.base_camera, [-0.2, 0, 0])
    loose = render(asset, cam, PresenterConfig(mode="mesh", mesh_depth_threshold=0.1))
    tight = render(asset, cam, PresenterConfig(mode="mesh", mesh_depth_threshold=0.01))

    def blended(res):  # depths strictly between the two surfaces = a stitched triangle
        d = res.depth[res.coverage]
        return ((d > 2.0 + 1e-4) & (d < 2.1 - 1e-4)).any()

    assert blended(loose) and not blended(tight)


def test_isolated_layer1_samples_still_render_as_points():
    """A sample in no triangle still shows up, as a one-pixel point."""
    pts = [(x, y, 5.0, BLUE) for x in range(4) for y in range(8)]
    pts.append((6, 6, 5.0, BLUE))  # alone: in no triangle
    asset = make_asset(layer1_points=pts)
    asset.layers[0].depth[:, 4:] = 5.0
    res = render(asset, translated_camera(asset.base_camera, [0.5, 0, 0]), MESH)
    assert (res.rgb[res.coverage] == BLUE).all(axis=-1).any()


def test_mesh_render_is_deterministic_and_covers_at_least_the_points(built_asset):
    """Same input, same output; and never less coverage than the point splat."""
    cam = translated_camera(built_asset.base_camera, [0.05, -0.03, 0.02])
    first = render(built_asset, cam, MESH)
    second = render(built_asset, cam, MESH)
    assert (first.rgb == second.rgb).all() and (first.coverage == second.coverage).all()
    assert first.coverage.sum() >= render(built_asset, cam).coverage.sum()


@pytest.mark.parametrize("bad", [{"mode": "voxels"}, {"mesh_depth_threshold": 0.0}])
def test_presenter_config_rejects_bad_mesh_settings(bad):
    """Unknown modes and non-positive thresholds are configuration errors."""
    with pytest.raises(ConfigurationError):
        PresenterConfig(**bad).validate()
