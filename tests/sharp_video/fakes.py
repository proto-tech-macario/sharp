"""Test doubles for the two external systems Stage 2 drives: Stage 1 (CUDA) and TMIV."""

from __future__ import annotations

import threading

from synthetic import make_stage1_result

from sharp_video.contract import SpatialFrame
from sharp_video.miv.tmiv_config import bitstream_output_path
from sharp_video.spatializer import FrameTimings


class FakeRunner:
    """Stands in for `run_logged` + TMIV: records commands, writes the MIV bitstream."""

    def __init__(self, payload: bytes = b"\x00MIV" * 250):
        self.commands = []
        self.payload = payload

    def __call__(self, cmd, log_path, cwd=None, env=None):
        cmd = [str(c) for c in cmd]
        self.commands.append(cmd)
        if "-o" in cmd:
            from pathlib import Path

            target = Path(cmd[cmd.index("-o") + 1]) / bitstream_output_path()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.payload)


class FakeSpatializer:
    """Stands in for `Stage1Spatializer`: a consistent synthetic 9-view frame per input frame.

    The texture shifts with the source frame index, so frames differ from each other.
    """

    def __init__(self, width: int = 32, height: int = 32, fail_on=()):
        self.width, self.height = width, height
        self.fail_on = set(fail_on)
        self.calls = []
        self._lock = threading.Lock()

    def __call__(self, decoded):
        with self._lock:
            self.calls.append(decoded.index)
        if decoded.index in self.fail_on:
            raise RuntimeError(f"stage 1 failed on frame {decoded.index}")
        result = make_stage1_result(self.width, self.height, phase=0.1 * decoded.index)
        frame = SpatialFrame.from_stage1(result, decoded.timestamp, pts=decoded.pts,
                                         source_frame_index=decoded.index)
        return frame, FrameTimings(0.5, 0.2, 0.05, 0.8, 3 * 2**30)
