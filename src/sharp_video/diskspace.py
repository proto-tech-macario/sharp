"""How much disk a video-to-MIV run needs, and whether it has it.

Stage 2 is disk-hungry: at 1280x720 views every frame leaves ~45 MB in the
Stage 1 frame cache, ~50 MB of raw views for TMIV, and ~100 MB of raw atlases
(texture + geometry) while TMIV and VVenC run. A 15 s clip is tens of
gigabytes, so the run is refused up front rather than dying hours in.

On WSL2 the Linux filesystem lives in a growable virtual disk (ext4.vhdx) on
a Windows drive. `df` inside Linux reports the virtual disk's maximum size,
not what the Windows drive can still hold, and when that drive fills the
whole VM stops. So on WSL the space that counts is what the Windows drive has
left, plus the room the virtual disk already holds but Linux is not using.
"""

from __future__ import annotations

import glob
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .miv.tmiv_config import atlas_budget

NUM_VIEWS = 9
#: Measured Stage 1 cache size (gzip'd uint8 RGB + float32 depth + mask): 4.6-5.5
#: bytes per view pixel; rounded up.
STAGE1_BYTES_PER_VIEW_PIXEL = 6.0
#: TMIV's raw input per view pixel: 10-bit texture + 16-bit geometry, 4:2:0, 2 bytes a sample.
MIV_INPUT_BYTES_PER_VIEW_PIXEL = 2 * 1.5 * 2
#: Raw atlas bytes per atlas luma sample: 4:2:0 at 2 bytes a sample.
ATLAS_BYTES_PER_SAMPLE = 1.5 * 2
#: Headroom on top of the estimate: logs, bitstreams, HDF5 overhead, the filesystem's own.
SAFETY_FACTOR = 1.15

_VHDX_GLOBS = (
    "/mnt/c/Users/*/AppData/Local/Packages/*/LocalState/ext4.vhdx",
    "/mnt/c/Users/*/AppData/Local/wsl/*/ext4.vhdx",
)


class InsufficientDiskSpaceError(RuntimeError):
    """The run would fill the disk; the message says how much it needs and has."""


@dataclass(frozen=True)
class DiskEstimate:
    """Bytes a run writes, by where they go."""

    stage1_cache: int  # Stage 1 frame cache (kept until the run succeeds)
    sequence: int  # two-step mode's temporal HDF5 (0 when streaming without a dump)
    miv_input: int  # raw views TMIV reads
    atlases: int  # raw atlases TMIV writes for VVenC, plus VVenC's reconstruction

    @property
    def total(self) -> int:
        """Everything at once: the peak, since nothing is deleted until TMIV finishes."""
        return self.stage1_cache + self.sequence + self.miv_input + self.atlases


def estimate(width: int, height: int, fps: float, frames: int, cached_frames: int = 0,
             sequence: bool = False) -> DiskEstimate:
    """Disk needed to convert `frames` frames of `width` x `height` views.

    `cached_frames` are already in the Stage 1 cache from an interrupted run,
    so they take no new space there.
    """
    view_pixels = NUM_VIEWS * width * height
    max_atlases, picture_size, _ = atlas_budget(width, height, fps)
    atlas_frame = max_atlases * picture_size * ATLAS_BYTES_PER_SAMPLE
    # Texture and geometry atlases, plus VVenC's reconstruction of the largest one.
    atlases = frames * (2 * atlas_frame + picture_size * ATLAS_BYTES_PER_SAMPLE)
    stage1_frame = view_pixels * STAGE1_BYTES_PER_VIEW_PIXEL
    return DiskEstimate(
        stage1_cache=int(max(0, frames - cached_frames) * stage1_frame),
        sequence=int(frames * stage1_frame) if sequence else 0,
        miv_input=int(frames * view_pixels * MIV_INPUT_BYTES_PER_VIEW_PIXEL),
        atlases=int(atlases),
    )


def _existing(path: Path) -> Path:
    path = Path(path).absolute()
    while not path.exists():
        path = path.parent
    return path


def _is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        return False


def _wsl_vhdx() -> Path | None:
    """This distro's virtual disk, if exactly one is found where WSL puts them."""
    found = [path for pattern in _VHDX_GLOBS for path in glob.glob(pattern)]
    return Path(found[0]) if len(found) == 1 else None


def free_bytes(path: str | Path) -> tuple[int, str]:
    """Bytes that can still be written at `path`, and a note on where the limit is."""
    path = _existing(Path(path))
    usage = shutil.disk_usage(path)
    if not _is_wsl() or str(path).startswith("/mnt/"):
        return usage.free, str(path)
    vhdx = _wsl_vhdx()
    host = vhdx.parent if vhdx else Path("/mnt/c")
    if not host.exists():
        return usage.free, str(path)
    host_free = shutil.disk_usage(host).free
    # Blocks the virtual disk already holds but Linux has freed can be reused
    # without it growing. Without the file, count only the Windows drive.
    slack = max(0, os.stat(vhdx).st_size - usage.used) if vhdx else 0
    free = min(usage.free, host_free + slack)
    drive = f"{str(host)[5:6].upper()}:" if str(host).startswith("/mnt/") else str(host)
    return free, f"the Windows drive {drive} holding WSL's virtual disk"


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


def check(path: str | Path, needed: DiskEstimate, frames: int) -> None:
    """Raise InsufficientDiskSpaceError if `needed` (with headroom) does not fit at `path`."""
    free, where = free_bytes(path)
    want = needed.total * SAFETY_FACTOR
    if want <= free:
        return
    per_frame = want / max(frames, 1)
    raise InsufficientDiskSpaceError(
        f"Not enough disk space: converting {frames} frames needs about {_gb(want)} "
        f"(Stage 1 cache {_gb(needed.stage1_cache)}, TMIV input {_gb(needed.miv_input)}, "
        f"atlases {_gb(needed.atlases)}"
        + (f", sequence {_gb(needed.sequence)}" if needed.sequence else "")
        + f"), but only {_gb(free)} is free on {where}. "
        f"That is about {int(free // per_frame)} frames at this size; convert fewer "
        f"frames, use a smaller view size, or free some space."
    )
