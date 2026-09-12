"""Temporal frame scheduling: which source frames get spatialized (spec §4.1, §36)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameSelection:
    """Selects frames by presentation-order index.

    `start_frame` is inclusive, `end_frame` exclusive (None = to the end), and
    `max_frames` caps how many frames are selected after `start_frame`.
    """

    start_frame: int = 0
    end_frame: int | None = None
    max_frames: int | None = None

    def validate(self) -> FrameSelection:
        """Raise ValueError for a selection that could never select a frame."""
        if self.start_frame < 0:
            raise ValueError(f"start_frame must be >= 0, got {self.start_frame}")
        if self.end_frame is not None and self.end_frame <= self.start_frame:
            raise ValueError(
                f"end_frame ({self.end_frame}) must be greater than start_frame "
                f"({self.start_frame})"
            )
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {self.max_frames}")
        return self

    def contains(self, index: int, selected_so_far: int) -> bool:
        """Whether the frame at `index` is selected, given how many already were."""
        if index < self.start_frame:
            return False
        if self.end_frame is not None and index >= self.end_frame:
            return False
        return self.max_frames is None or selected_so_far < self.max_frames

    def exhausted(self, index: int, selected_so_far: int) -> bool:
        """True once no frame at `index` or later can be selected (decoding may stop)."""
        if self.end_frame is not None and index >= self.end_frame:
            return True
        return self.max_frames is not None and selected_so_far >= self.max_frames
