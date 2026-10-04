import numpy as np
from ds_helpers import make_camera
from ds_m0.creator import (
    CandidateGenerationConfig,
    Layer1Candidates,
    MergeConfig,
    generate_candidates,
    merge_layer1_candidates,
)
from ds_m0.model import Layer, SourceView

W = H = 4
CFG = CandidateGenerationConfig(relative_depth_threshold=0.02)


def base_layer(depth=2.0):
    layer = Layer.empty(W, H)
    layer.valid[:] = True
    layer.depth[:] = depth
    return layer


def view(depth, camera=None, valid=None, view_id=0):
    d = np.full((H, W), depth, np.float32)
    return SourceView(view_id, np.full((H, W, 3), 9, np.uint8), d,
                      np.ones((H, W), bool) if valid is None else valid, camera or make_camera(W, H, f=4.0))


def gen(v, layer0=None):
    return generate_candidates(v, layer0 or base_layer(), make_camera(W, H, f=4.0), CFG)


def test_fully_visible_surface_makes_no_candidate():
    assert len(gen(view(2.0))) == 0


def test_hidden_surface_makes_candidate_at_base_depth():
    c = gen(view(3.0))
    assert len(c) == W * H and np.allclose(c.depth_base, 3.0)
    assert (c.target_x < W).all() and (c.rgb == 9).all()


def test_equal_depth_and_just_below_threshold_not_hidden_just_above_is():
    assert len(gen(view(2.039))) == 0
    assert len(gen(view(2.041))) == W * H


def test_invalid_source_pixels_make_no_candidates():
    valid = np.zeros((H, W), bool)
    valid[1, 2] = True
    c = gen(view(3.0, valid=valid))
    assert len(c) == 1 and (c.target_x[0], c.target_y[0]) == (2, 1)
    v = view(3.0)
    v.depth[0, 0], v.depth[0, 1] = np.nan, -1.0
    assert len(gen(v)) == W * H - 2


def test_behind_camera_and_outside_image_discarded():
    flip = make_camera(W, H, f=4.0, R=np.diag([-1.0, 1.0, -1.0]))  # looks the other way
    assert len(gen(view(3.0, camera=flip))) == 0
    far_right = make_camera(W, H, f=4.0, C=(100.0, 0.0, 0.0))
    assert len(gen(view(3.0, camera=far_right))) == 0


def test_invalid_layer0_pixel_is_never_a_target():
    l0 = base_layer()
    l0.valid[:, :2] = False
    c = gen(view(3.0), l0)
    assert (c.target_x >= 2).all()


def cands(rows):
    """rows: (x, y, depth, view_id, angle, source_index)"""
    a = np.array(rows, dtype=float)
    return Layer1Candidates(a[:, 0].astype(np.int64), a[:, 1].astype(np.int64), a[:, 2],
                            np.tile((a[:, 5] % 256).astype(np.uint8)[:, None], (1, 3)),
                            a[:, 3].astype(np.int64), a[:, 4], a[:, 5].astype(np.int64))


def merge(c):
    return merge_layer1_candidates(c, W, H, MergeConfig(1e-6))


def test_merge_nearest_hidden_wins():
    layer = merge(cands([(1, 1, 5.0, 2, 0.1, 0), (1, 1, 3.0, 5, 0.9, 1)]))
    assert layer.valid.sum() == 1 and layer.depth[1, 1] == 3.0 and layer.rgb[1, 1, 0] == 1


def test_merge_equal_depth_smaller_angle_then_lower_view_id():
    layer = merge(cands([(0, 0, 3.0, 1, 0.5, 0), (0, 0, 3.0, 2, 0.2, 1)]))
    assert layer.rgb[0, 0, 0] == 1  # angle 0.2 wins
    layer = merge(cands([(0, 0, 3.0, 7, 0.3, 0), (0, 0, 3.0, 2, 0.3, 1)]))
    assert layer.rgb[0, 0, 0] == 1  # view id 2 wins


def test_merge_invalid_ignored_and_empty_stays_invalid():
    layer = merge(cands([(0, 0, np.nan, 1, 0.1, 0), (0, 0, -1.0, 1, 0.1, 1), (2, 2, 4.0, 1, 0.1, 2)]))
    assert layer.valid.sum() == 1 and layer.valid[2, 2]
    assert merge(Layer1Candidates.empty()).valid.sum() == 0


def test_merge_does_not_depend_on_order():
    rows = [(x % 3, 0, 3.0 + (x % 2) * 1e-9, x % 4, 0.1 * (x % 3), x) for x in range(30)]
    ref = merge(cands(rows))
    for seed in range(5):
        perm = np.random.default_rng(seed).permutation(len(rows))
        got = merge(cands([rows[i] for i in perm]))
        assert (got.valid == ref.valid).all() and (got.depth == ref.depth).all() and (got.rgb == ref.rgb).all()
