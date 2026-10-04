"""Target-image depth buffer with an explicit, deterministic priority tuple."""

from __future__ import annotations

import numpy as np

from ..geometry import visibility

_INT_MAX = np.iinfo(np.int64).max


class DepthBuffer:
    """Smaller depth wins; depths within `epsilon` tie-break on (layer_id, source_order)."""

    def __init__(self, width: int, height: int, epsilon: float = 1e-5):
        self.width, self.height, self.epsilon = width, height, epsilon
        self.clear()

    def clear(self) -> None:
        shape = (self.height, self.width)
        self.depth = np.full(shape, np.inf)
        self.layer_id = np.full(shape, _INT_MAX, np.int64)
        self.source_order = np.full(shape, _INT_MAX, np.int64)

    def test_and_update(self, x: int, y: int, depth: float, priority: tuple[int, int]) -> bool:
        cur = self.depth[y, x]
        accept = depth < cur - self.epsilon
        if not accept and abs(depth - cur) <= self.epsilon:
            accept = priority < (int(self.layer_id[y, x]), int(self.source_order[y, x]))
        if accept:
            self.depth[y, x] = depth
            self.layer_id[y, x], self.source_order[y, x] = priority
        return bool(accept)

    def update_many(self, x, y, depth, layer_id, source_order) -> np.ndarray:
        """Vectorised batch update. Returns indices of the samples that became visible.
        The result depends only on the set of samples, not their order."""
        group = y * self.width + x
        win = visibility.select_winners(group, depth, [layer_id, source_order], self.epsilon)
        wx, wy, wd = x[win], y[win], depth[win]
        cur = self.depth[wy, wx]
        better = wd < cur - self.epsilon
        tie = np.abs(wd - cur) <= self.epsilon
        pri_better = (layer_id[win] < self.layer_id[wy, wx]) | (
            (layer_id[win] == self.layer_id[wy, wx]) & (source_order[win] < self.source_order[wy, wx])
        )
        accepted = better | (tie & pri_better)
        win = win[accepted]
        self.depth[y[win], x[win]] = depth[win]
        self.layer_id[y[win], x[win]] = layer_id[win]
        self.source_order[y[win], x[win]] = source_order[win]
        return win
