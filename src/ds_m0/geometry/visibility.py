"""Depth comparison and the deterministic, order-independent winner selection
shared by the Layer-1 candidate merge and the Presenter depth buffer."""

from __future__ import annotations

import numpy as np


def depths_equal(a, b, epsilon: float):
    return np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)) <= epsilon


def hidden_threshold(z_base, relative: float, absolute: float):
    """threshold = max(absolute, relative * Z0)."""
    return np.maximum(absolute, relative * np.asarray(z_base, np.float64))


def is_hidden(z_candidate, z_base, relative: float, absolute: float):
    """Zc > Z0 + threshold."""
    z0 = np.asarray(z_base, np.float64)
    return np.asarray(z_candidate, np.float64) > z0 + hidden_threshold(z0, relative, absolute)


def select_winners(group: np.ndarray, depth: np.ndarray, tie_keys: list[np.ndarray], epsilon: float) -> np.ndarray:
    """Pick one row per `group` value. Returns the winning row indices, ordered by group.

    1. smallest depth wins; rows within `epsilon` of that minimum stay in contention;
    2. contenders are ordered lexicographically by `tie_keys` (smaller wins).
    The last tie key must be unique within a group so the result never depends on
    input order. The outcome depends only on the *set* of rows.
    """
    n = len(group)
    if n == 0:
        return np.empty(0, np.int64)
    order = np.lexsort((depth, group))
    g, d = group[order], depth[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    counts = np.diff(np.r_[starts, n])
    group_min = np.repeat(d[starts], counts)
    contenders = order[d <= group_min + epsilon]
    keys = tuple(k[contenders] for k in reversed(tie_keys)) + (group[contenders],)
    o2 = np.lexsort(keys)
    cg = group[contenders][o2]
    first = np.r_[True, cg[1:] != cg[:-1]]
    return contenders[o2][first]
