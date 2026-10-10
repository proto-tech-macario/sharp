import numpy as np
import pytest
from ds_helpers import make_camera
from ds_m0.errors import ConfigurationError, GeometryError
from ds_m0.geometry import camera_math, projection, transformation, visibility
from ds_m0.model import Camera

TOL = 1e-9


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


@pytest.mark.parametrize("u,v", [(32.0, 24.0), (0.0, 0.0), (63.0, 47.0), (10.5, 3.25)])
@pytest.mark.parametrize("depth", [0.5, 2.0, 37.0])
@pytest.mark.parametrize("skew", [0.0, 0.7])
def test_backproject_project_identity(u, v, depth, skew):
    cam = make_camera(64, 48, f=50.0, fy=61.0, cx=30.5, cy=20.25, skew=skew)  # non-square, off-centre
    p = camera_math.backproject(cam, (u, v), depth)
    out = projection.project(cam, p)
    assert out.valid and abs(out.u - u) < TOL and abs(out.v - v) < TOL and abs(out.depth - depth) < TOL


def test_pose_roundtrip_and_identity():
    cam = make_camera(R=rot_y(0.3), C=(0.4, -0.2, 1.0))
    p = np.array([0.3, -1.2, 4.0])
    assert np.allclose(transformation.camera_to_world(cam, transformation.world_to_camera(cam, p)), p, atol=TOL)
    ident = make_camera()
    assert np.allclose(transformation.world_to_camera(ident, p), p)
    assert np.allclose(transformation.camera_to_world(ident, p), p)


def test_camera_center_and_forward():
    cam = make_camera(R=rot_y(0.5), C=(1.0, 2.0, 3.0))
    assert np.allclose(camera_math.camera_center(cam), [1, 2, 3])
    assert np.allclose(camera_math.forward_direction(cam), rot_y(0.5)[2])


def test_known_translation_gives_expected_displacement():
    base = make_camera(width=64, height=64, f=100.0)
    moved = transformation.translated_camera(base, [0.1, 0.0, 0.0])  # camera moves right -> content shifts left
    p = camera_math.backproject(base, (32.0, 32.0), 4.0)
    u = projection.project(moved, transformation.world_to_camera(moved, transformation.camera_to_world(base, p))).u
    assert abs(u - (32.0 - 100.0 * 0.1 / 4.0)) < TOL


def test_points_behind_camera_rejected():
    cam = make_camera()
    assert not projection.project(cam, [0.0, 0.0, -1.0]).valid
    assert not projection.project(cam, [0.0, 0.0, 0.0]).valid
    _, _, _, ok = projection.project_many(cam, np.array([[0, 0, -2.0], [0, 0, 2.0]]))
    assert ok.tolist() == [False, True]


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_invalid_depth_rejected(bad):
    with pytest.raises(GeometryError):
        camera_math.backproject(make_camera(), (1, 1), bad)


def test_invalid_camera_rejected():
    good = make_camera()
    for mutate in (
        lambda c: setattr(c, "fx", 0.0), lambda c: setattr(c, "fy", np.nan),
        lambda c: setattr(c, "width", 0), lambda c: setattr(c, "rotation", np.eye(3) * 2),
        lambda c: setattr(c, "translation", np.array([np.nan, 0, 0])),
        lambda c: setattr(c, "projection_type", "fisheye"),
    ):
        cam = Camera.from_dict(good.to_dict())
        mutate(cam)
        with pytest.raises(GeometryError):
            camera_math.validate(cam)
    with pytest.raises(GeometryError):
        Camera.from_dict({"fx": 1})


def test_view_angle():
    assert camera_math.view_angle(make_camera(), make_camera()) == 0.0
    assert abs(camera_math.view_angle(make_camera(), make_camera(R=rot_y(0.25))) - 0.25) < 1e-12


def test_rounding_is_deterministic_and_validated():
    x = np.array([0.5, 1.5, 2.5, -0.5, 2.4999])
    assert projection.round_to_pixel(x, "half_up").tolist() == [1, 2, 3, 0, 2]
    assert projection.round_to_pixel(x, "half_even").tolist() == [0, 2, 2, 0, 2]
    with pytest.raises(ConfigurationError):
        projection.round_to_pixel(x, "banker")


def test_hidden_threshold_just_below_and_above():
    z0 = 2.0  # relative 2% -> threshold 0.04
    assert not visibility.is_hidden(2.039, z0, 0.02, 0.0)
    assert visibility.is_hidden(2.041, z0, 0.02, 0.0)
    assert visibility.is_hidden(2.041, z0, 0.0, 0.04)  # absolute-only form
    assert not visibility.is_hidden(2.041, z0, 0.02, 0.05)  # max(abs, rel)


def test_select_winners_is_order_independent():
    rng = np.random.default_rng(0)
    n = 300
    group = rng.integers(0, 20, n)
    depth = np.round(rng.random(n) * 3, 1)  # many exact ties
    k1, k2 = rng.integers(0, 3, n), np.arange(n)
    ref = {int(group[i]): int(k2[i]) for i in visibility.select_winners(group, depth, [k1, k2], 1e-6)}
    for _ in range(5):
        p = rng.permutation(n)
        got = {int(group[p][i]): int(k2[p][i]) for i in visibility.select_winners(group[p], depth[p], [k1[p], k2[p]], 1e-6)}
        assert got == ref
