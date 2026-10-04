"""Tests for sharp_video.diskspace: estimating and checking a run's disk use."""

from __future__ import annotations

import pytest
from sharp_video import diskspace
from sharp_video.diskspace import DiskEstimate, InsufficientDiskSpaceError


def test_estimate_matches_what_a_1280x720_run_wrote():
    """Measured on a real 383-frame run: 45 MB cache, 50 MB TMIV input, one 1280x6480 atlas."""
    needed = diskspace.estimate(1280, 720, 25.0, frames=100)
    assert needed.stage1_cache / 100 == pytest.approx(45e6, rel=0.25)
    assert needed.miv_input / 100 == 9 * 1280 * 720 * 6
    # texture + geometry atlases + VVenC's reconstruction, 3 bytes a sample each
    assert needed.atlases / 100 == 3 * 1280 * 6480 * 3
    assert needed.sequence == 0


def test_cached_frames_and_the_sequence_are_accounted_for():
    """Resumed frames need no new cache space; two-step mode adds a sequence of the same size."""
    fresh = diskspace.estimate(64, 64, 30.0, frames=10)
    resumed = diskspace.estimate(64, 64, 30.0, frames=10, cached_frames=4)
    assert resumed.stage1_cache == fresh.stage1_cache * 6 // 10
    assert diskspace.estimate(64, 64, 30.0, frames=10, sequence=True).sequence == \
        fresh.stage1_cache


def test_check_refuses_with_the_numbers(monkeypatch, tmp_path):
    """The refusal says what is needed, what is free, and how many frames would fit."""
    monkeypatch.setattr(diskspace, "free_bytes", lambda path: (50e9, "the Windows drive C:"))
    needed = DiskEstimate(stage1_cache=20e9, sequence=0, miv_input=20e9, atlases=50e9)
    with pytest.raises(InsufficientDiskSpaceError) as error:
        diskspace.check(tmp_path, needed, frames=400)
    message = str(error.value)
    assert "400 frames" in message and "50.0 GB is free on the Windows drive C:" in message
    assert "about 193 frames" in message

    diskspace.check(tmp_path, DiskEstimate(10e9, 0, 10e9, 20e9), frames=400)  # fits


def test_free_bytes_off_wsl_is_the_filesystems(monkeypatch, tmp_path):
    """Outside WSL the filesystem's own free space is the answer, even for a path not made yet."""
    monkeypatch.setattr(diskspace, "_is_wsl", lambda: False)
    free, where = diskspace.free_bytes(tmp_path / "not" / "made")
    assert free > 0 and where == str(tmp_path)


def test_free_bytes_on_wsl_counts_the_windows_drive(monkeypatch, tmp_path):
    """On WSL: the host drive's free space plus what the virtual disk holds but Linux freed."""
    host = tmp_path / "host"
    host.mkdir()
    vhdx = host / "ext4.vhdx"
    with open(vhdx, "wb") as f:
        f.truncate(10**9)
    usage = {str(host): (0, 0, 2 * 10**9), str(tmp_path): (10**12, 4 * 10**8, 10**11)}

    def disk_usage(path):
        from collections import namedtuple

        return namedtuple("usage", "total used free")(*usage[str(path)])

    monkeypatch.setattr(diskspace, "_is_wsl", lambda: True)
    monkeypatch.setattr(diskspace, "_wsl_vhdx", lambda: vhdx)
    monkeypatch.setattr(diskspace.shutil, "disk_usage", disk_usage)
    free, where = diskspace.free_bytes(tmp_path)
    assert free == 2 * 10**9 + (10**9 - 4 * 10**8)  # host free + slack inside the vhdx
    assert "virtual disk" in where
