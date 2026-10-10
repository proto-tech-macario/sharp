"""Mesh Presenter: each layer is drawn as a triangle surface instead of one-pixel points.

Neighbouring valid samples of a layer (on the base pixel grid) are joined into
triangles, two per 2x2 quad. A triangle is dropped when its corner depths differ
by more than `mesh_depth_threshold` (relative), so foreground and background are
never stitched together across a depth edge. Triangles are projected into the
target camera and rasterised at integer pixel positions (the DS pixel convention),
with colour interpolated across each triangle and depth interpolated as 1/z. A
depth buffer keeps the nearest fragment per pixel.

Every sample is also splatted as a one-pixel point, as in the point Presenter. A
point is used only where no triangle landed, where it is clearly in front of the
triangle surface there (by more than the same threshold), or where it is in front
of a triangle from a later layer (Layer 1 holds what is hidden behind Layer 0).
That keeps the half-pixel rim triangles cannot reach -- along the image border and
on each side of a depth edge -- and isolated samples that belong to no triangle.

Compared with the point splat this closes the pinholes a stretched surface leaves
between samples, and replaces nearest-pixel resampling with interpolation.
Deterministic: the result depends only on the asset and the camera.
"""

from __future__ import annotations

import numpy as np

from ..config.m0_config import PresenterConfig
from ..geometry import camera_math, projection, transformation
from ..model.asset import DSAsset
from ..model.camera import Camera
from ..model.layer import Layer

#: Barycentric slack, so pixels exactly on a shared edge are not lost to float error.
_EDGE_EPS = 1e-7
#: Upper bound on candidate pixels examined per rasterisation batch (memory cap).
_BATCH_PIXELS = 4_000_000


def _layer_triangles(layer: Layer, threshold: float):
    """Triangles as three corner arrays of flat grid indices, split along the flatter diagonal."""
    h, w = layer.height, layer.width
    if h < 2 or w < 2:
        empty = np.empty(0, np.int64)
        return empty, empty, empty
    z = np.where(layer.valid, layer.depth.astype(np.float64), np.nan).ravel()
    idx = np.arange(h * w, dtype=np.int64).reshape(h, w)
    tl, tr = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    bl, br = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    ztl, ztr, zbl, zbr = z[tl], z[tr], z[bl], z[br]

    def ok(za, zb, zc):  # NaN (invalid corner) makes the comparison False
        lo = np.minimum(np.minimum(za, zb), zc)
        hi = np.maximum(np.maximum(za, zb), zc)
        with np.errstate(invalid="ignore"):
            return hi <= lo * (1.0 + threshold)

    # The four triangles of a quad, each leaving out one corner.
    corners = {
        "no_br": (tl, tr, bl),  # } diagonal tr-bl
        "no_tl": (tr, br, bl),  # }
        "no_bl": (tl, tr, br),  # } diagonal tl-br
        "no_tr": (tl, br, bl),  # }
    }
    good = {
        "no_br": ok(ztl, ztr, zbl), "no_tl": ok(ztr, zbr, zbl),
        "no_bl": ok(ztl, ztr, zbr), "no_tr": ok(ztl, zbr, zbl),
    }
    with np.errstate(invalid="ignore"):
        prefer_trbl = ~(np.abs(ztr - zbl) > np.abs(ztl - zbr))  # NaN -> prefer tr-bl

    split_a = good["no_br"] & good["no_tl"]
    split_b = good["no_bl"] & good["no_tr"]
    use_a = split_a & (prefer_trbl | ~split_b)
    use_b = split_b & ~use_a
    rest = ~use_a & ~use_b  # at most one triangle: the first continuous one
    taken = np.zeros_like(rest)
    picks = {}
    for key, split in (("no_br", use_a), ("no_tl", use_a), ("no_bl", use_b), ("no_tr", use_b)):
        single = rest & good[key] & ~taken
        taken |= single
        picks[key] = split | single
    return tuple(
        np.concatenate([corners[k][i][picks[k]] for k in corners]) for i in range(3)
    )


def _rasterize(u, v, width: int, height: int):
    """Rasterise triangles given per-corner screen coordinates u = (u0, u1, u2), v likewise.

    Returns fragments: pixel index, triangle index, and the barycentric weights b0, b1, b2.
    """
    u0, u1, u2 = u
    v0, v1, v2 = v
    x0 = np.ceil(np.minimum(np.minimum(u0, u1), u2) - _EDGE_EPS).astype(np.int64)
    x1 = np.floor(np.maximum(np.maximum(u0, u1), u2) + _EDGE_EPS).astype(np.int64)
    y0 = np.ceil(np.minimum(np.minimum(v0, v1), v2) - _EDGE_EPS).astype(np.int64)
    y1 = np.floor(np.maximum(np.maximum(v0, v1), v2) + _EDGE_EPS).astype(np.int64)
    np.maximum(x0, 0, out=x0)
    np.maximum(y0, 0, out=y0)
    np.minimum(x1, width - 1, out=x1)
    np.minimum(y1, height - 1, out=y1)
    area = (u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0)
    live = np.flatnonzero((x1 >= x0) & (y1 >= y0) & (np.abs(area) > 1e-12))
    out_pix, out_tri, out_b0, out_b1 = [], [], [], []
    if len(live):
        u0, u1, u2, v0, v1, v2, a = (arr[live] for arr in (u0, u1, u2, v0, v1, v2, area))
        x0, x1, y0, y1 = x0[live], x1[live], y0[live], y1[live]
        # Barycentric weights are linear in (x, y): w = e + dx * ox + dy * oy, where e is
        # the weight at the bounding box's top-left pixel and (ox, oy) the offset from it.
        bx, by = x0.astype(np.float64), y0.astype(np.float64)
        e0 = ((u1 - bx) * (v2 - by) - (u2 - bx) * (v1 - by)) / a
        e1 = ((u2 - bx) * (v0 - by) - (u0 - bx) * (v2 - by)) / a
        dx0, dy0 = (v1 - v2) / a, (u2 - u1) / a
        dx1, dy1 = (v2 - v0) / a, (u0 - u2) / a
        bw, bh = x1 - x0, y1 - y0
        # Bucket by bounding-box size (powers of two) so each batch is a dense grid.
        bucket = np.ceil(np.log2(np.maximum(bw, bh) + 1)).astype(np.int64)
        for b in np.unique(bucket):
            s = 1 << int(b)
            rows_b = np.flatnonzero(bucket == b)
            per = max(1, _BATCH_PIXELS // (s * s))
            oy, ox = np.divmod(np.arange(s * s, dtype=np.int64), s)
            oxf, oyf = ox.astype(np.float64), oy.astype(np.float64)
            for start in range(0, len(rows_b), per):
                r = rows_b[start:start + per]
                w0 = e0[r, None] + dx0[r, None] * oxf + dy0[r, None] * oyf
                w1 = e1[r, None] + dx1[r, None] * oxf + dy1[r, None] * oyf
                inside = (w0 >= -_EDGE_EPS) & (w1 >= -_EDGE_EPS) & (w0 + w1 <= 1.0 + _EDGE_EPS)
                if s > 1:
                    inside &= (ox <= bw[r, None]) & (oy <= bh[r, None])
                ri, k = np.nonzero(inside)
                if len(ri) == 0:
                    continue
                rr = r[ri]
                out_pix.append((y0[rr] + oy[k]) * width + (x0[rr] + ox[k]))
                out_tri.append(live[rr])
                out_b0.append(w0[ri, k])
                out_b1.append(w1[ri, k])
    if not out_pix:
        e = np.empty(0)
        return np.empty(0, np.int64), np.empty(0, np.int64), e, e, e
    b0, b1 = np.concatenate(out_b0), np.concatenate(out_b1)
    return np.concatenate(out_pix), np.concatenate(out_tri), b0, b1, 1.0 - b0 - b1


def _nearest(pix, z, rank, n_pix: int):
    """Indices of the nearest fragment per pixel (ties: lowest rank, which is unique)."""
    zbuf = np.full(n_pix, np.inf)
    np.minimum.at(zbuf, pix, z)
    cand = np.flatnonzero(z == zbuf[pix])
    rbuf = np.full(n_pix, np.iinfo(np.int64).max)
    np.minimum.at(rbuf, pix[cand], rank[cand])
    return cand[rank[cand] == rbuf[pix[cand]]]


def render_mesh(
    asset: DSAsset, target_camera: Camera, config: PresenterConfig, width: int, height: int
):
    """Mesh render of every layer; returns (rgb uint8, coverage bool, depth float32)."""
    n_pix = width * height
    frag_pix, frag_z, frag_rgb, frag_rank, frag_layer = [], [], [], [], []
    pt_pix, pt_z, pt_rgb, pt_layer = [], [], [], []
    rank_base = 0  # unique, deterministic tie-break across layers and primitives
    for layer_id, layer in enumerate(asset.layers):
        if not layer.valid.any():
            continue
        py, px = np.nonzero(layer.valid)
        z = layer.depth[py, px].astype(np.float64)
        p_base = camera_math.backproject_many(asset.base_camera, px, py, z)
        p_world = transformation.camera_to_world_many(asset.base_camera, p_base)
        p_tgt = transformation.world_to_camera_many(target_camera, p_world)
        u_s, v_s, z_t, ok = projection.project_many(target_camera, p_tgt)
        n = layer.width * layer.height
        flat = py * layer.width + px
        u = np.full(n, np.nan)
        v = np.full(n, np.nan)
        iz = np.full(n, np.nan)  # NaN marks samples that did not project
        u[flat[ok]], v[flat[ok]], iz[flat[ok]] = u_s[ok], v_s[ok], 1.0 / z_t[ok]
        colours = layer.rgb.reshape(-1, 3)

        a, b, c = _layer_triangles(layer, config.mesh_depth_threshold)
        projected = ~np.isnan(iz[a]) & ~np.isnan(iz[b]) & ~np.isnan(iz[c])
        a, b, c = a[projected], b[projected], c[projected]
        if len(a):
            pix, tri, b0, b1, b2 = _rasterize((u[a], u[b], u[c]), (v[a], v[b], v[c]), width, height)
            ia, ib, ic = a[tri], b[tri], c[tri]
            za, zb, zc = iz[ia], iz[ib], iz[ic]
            izf = b0 * za + b1 * zb + b2 * zc  # 1/z is linear in screen space
            # Perspective-correct colour: interpolate colour/z, then divide by 1/z.
            wa, wb, wc = (b0 * za / izf), (b1 * zb / izf), (b2 * zc / izf)
            frag_rgb.append(
                wa[:, None] * colours[ia] + wb[:, None] * colours[ib] + wc[:, None] * colours[ic]
            )
            frag_pix.append(pix)
            frag_z.append(1.0 / izf)
            frag_rank.append(rank_base + tri)
            frag_layer.append(np.full(len(tri), layer_id, np.int8))
            rank_base += len(a)

        # Every sample as a one-pixel point (raster order within the layer).
        pts = flat[ok]
        tx, ty, inside = projection.to_pixels(u[pts], v[pts], width, height, config.rounding_mode)
        pts, tx, ty = pts[inside], tx[inside], ty[inside]
        pt_pix.append(ty * width + tx)
        pt_z.append(1.0 / iz[pts])
        pt_rgb.append(colours[pts].astype(np.float64))
        pt_layer.append(np.full(len(pts), layer_id, np.int8))

    out_rgb = np.zeros((n_pix, 3), np.float64)
    out_z = np.full(n_pix, np.inf)
    out_layer = np.full(n_pix, np.iinfo(np.int8).max, np.int8)
    if frag_pix:
        pix, fz = np.concatenate(frag_pix), np.concatenate(frag_z)
        win = _nearest(pix, fz, np.concatenate(frag_rank), n_pix)
        out_rgb[pix[win]] = np.concatenate(frag_rgb)[win]
        out_z[pix[win]] = fz[win]
        out_layer[pix[win]] = np.concatenate(frag_layer)[win]
    if pt_pix:
        pix, pz = np.concatenate(pt_pix), np.concatenate(pt_z)
        win = _nearest(pix, pz, np.arange(len(pix)), n_pix)
        pix, pz = pix[win], pz[win]
        pc, pl = np.concatenate(pt_rgb)[win], np.concatenate(pt_layer)[win]
        mesh_z = out_z[pix]
        use = pz * (1.0 + config.mesh_depth_threshold) < mesh_z  # also true where no triangle
        use |= (pl < out_layer[pix]) & (pz < mesh_z)
        out_rgb[pix[use]] = pc[use]
        out_z[pix[use]] = pz[use]

    covered = np.isfinite(out_z).reshape(height, width)
    rgb = np.clip(np.floor(out_rgb + 0.5), 0, 255).astype(np.uint8).reshape(height, width, 3)
    rgb[~covered] = 0
    return rgb, covered, out_z.astype(np.float32).reshape(height, width)
