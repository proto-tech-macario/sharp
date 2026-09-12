"""Tests for sharp_video.miv.tmiv (locating a patched TMIV build, running it)."""

from __future__ import annotations

import json
import sys

import pytest
from sharp_video.miv.tmiv import (
    ENV_VAR,
    PATCH_NAME,
    TmivError,
    TmivNotFoundError,
    find_tmiv,
    run_logged,
)


def _fake_tmiv(root, patched=True):
    (root / "install" / "bin").mkdir(parents=True)
    (root / "src" / "scripts").mkdir(parents=True)
    for name in ("TmivEncoder", "TmivDecoder", "TmivMultiplexer", "TmivParser"):
        (root / "install" / "bin" / name).write_text("")
    (root / "src" / "scripts" / "encode.py").write_text("")
    marker = {"version": "v24.0", "patch": PATCH_NAME if patched else None}
    (root / "sharp_tmiv.json").write_text(json.dumps(marker))
    return root


def test_explicit_path_wins(tmp_path, monkeypatch):
    """Explicit path wins."""
    explicit = _fake_tmiv(tmp_path / "a")
    monkeypatch.setenv(ENV_VAR, str(_fake_tmiv(tmp_path / "b")))
    tmiv = find_tmiv(explicit)
    assert tmiv.root == explicit
    assert tmiv.exe("TmivEncoder") == explicit / "install" / "bin" / "TmivEncoder"
    assert tmiv.encode_script == explicit / "src" / "scripts" / "encode.py"


def test_environment_variable_then_default(tmp_path, monkeypatch):
    """Environment variable then default."""
    env_root = _fake_tmiv(tmp_path / "env")
    monkeypatch.setenv(ENV_VAR, str(env_root))
    assert find_tmiv().root == env_root

    monkeypatch.delenv(ENV_VAR)
    monkeypatch.chdir(tmp_path)
    _fake_tmiv(tmp_path / ".tmiv")
    assert find_tmiv().root == (tmp_path / ".tmiv").resolve()


def test_missing_install_explains_how_to_build(tmp_path, monkeypatch):
    """Missing install explains how to build."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(TmivNotFoundError) as info:
        find_tmiv()
    assert ENV_VAR in str(info.value) and "build_tmiv.sh" in str(info.value)


def test_unpatched_install_rejected(tmp_path):
    """Unpatched install rejected."""
    root = _fake_tmiv(tmp_path / "t", patched=False)
    with pytest.raises(TmivNotFoundError, match="patch"):
        find_tmiv(root)


def test_missing_executable_named(tmp_path):
    """Missing executable named."""
    tmiv = find_tmiv(_fake_tmiv(tmp_path / "t"))
    with pytest.raises(TmivNotFoundError, match="TmivRenderer"):
        tmiv.exe("TmivRenderer")


def test_run_logged_writes_log(tmp_path):
    """Run logged writes log."""
    log = tmp_path / "logs" / "ok.log"
    run_logged([sys.executable, "-c", "print('hello'); import sys; print('err', file=sys.stderr)"],
               log)
    text = log.read_text()
    assert "hello" in text and "err" in text


def test_run_logged_failure_points_at_log(tmp_path):
    """Run logged failure points at log."""
    log = tmp_path / "bad.log"
    with pytest.raises(TmivError) as info:
        run_logged([sys.executable, "-c", "print('nope'); raise SystemExit(3)"], log)
    assert info.value.log_path == log
    assert "exit 3" in str(info.value) and str(log) in str(info.value)
