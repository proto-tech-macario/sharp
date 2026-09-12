"""Locating the patched TMIV build and running its tools.

Stage 2 drives TMIV v24.0 (MPEG's reference MIV software) with a small patch
that lets view poses change per frame (`third_party/tmiv/`). `scripts/build_tmiv.sh`
produces the expected layout::

    <root>/src             TMIV source checkout (patched) -- encode.py, configs
    <root>/install/bin     TmivEncoder, TmivMultiplexer, TmivDecoder, vvencFFapp, ...
    <root>/sharp_tmiv.json build marker: {"version": ..., "patch": ...}

The root is taken from an explicit argument, then `$SHARP_TMIV_DIR`, then
`./.tmiv`.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ENV_VAR = "SHARP_TMIV_DIR"
PATCH_NAME = "sharp-per-frame-cameras"
TMIV_VERSION = "v24.0"
_BUILD_HINT = (
    f"Build it with scripts/build_tmiv.sh (installs into ./.tmiv), or point {ENV_VAR} "
    "at an existing build."
)


class TmivNotFoundError(RuntimeError):
    """No usable (patched) TMIV build was found."""


class TmivError(RuntimeError):
    """A TMIV tool exited with an error; its output is in `log_path`."""

    def __init__(self, message: str, log_path: Path):
        super().__init__(message)
        self.log_path = log_path


@dataclass(frozen=True)
class TmivInstall:
    """A TMIV build laid out by scripts/build_tmiv.sh."""

    root: Path

    @property
    def source_dir(self) -> Path:
        return self.root / "src"

    @property
    def bin_dir(self) -> Path:
        return self.root / "install" / "bin"

    @property
    def encode_script(self) -> Path:
        return self.source_dir / "scripts" / "encode.py"

    @property
    def marker(self) -> dict:
        path = self.root / "sharp_tmiv.json"
        return json.loads(path.read_text()) if path.exists() else {}

    @property
    def patched(self) -> bool:
        return self.marker.get("patch") == PATCH_NAME

    def exe(self, name: str) -> Path:
        path = self.bin_dir / name
        if not path.exists():
            raise TmivNotFoundError(f"TMIV executable {name} not found in {self.bin_dir}")
        return path


def _looks_installed(root: Path) -> bool:
    return (root / "install" / "bin" / "TmivEncoder").exists()


def find_tmiv(explicit: str | Path | None = None) -> TmivInstall:
    """Return the first TMIV build found (explicit, $SHARP_TMIV_DIR, ./.tmiv)."""
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    if os.environ.get(ENV_VAR):
        candidates.append(Path(os.environ[ENV_VAR]))
    candidates.append(Path.cwd() / ".tmiv")

    for root in candidates:
        root = root.expanduser().resolve()
        if not _looks_installed(root):
            if explicit is not None and root == Path(explicit).expanduser().resolve():
                raise TmivNotFoundError(f"No TMIV build at {root}. {_BUILD_HINT}")
            continue
        tmiv = TmivInstall(root)
        if not tmiv.patched:
            raise TmivNotFoundError(
                f"The TMIV build at {root} lacks the {PATCH_NAME} patch, so it cannot carry "
                f"per-frame cameras. {_BUILD_HINT}"
            )
        return tmiv
    raise TmivNotFoundError(f"No TMIV build found. {_BUILD_HINT}")


def run_logged(
    cmd: Sequence[str | Path],
    log_path: str | Path,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run `cmd`, sending stdout+stderr to `log_path`; raise TmivError on failure."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    args = [str(part) for part in cmd]
    with open(log_path, "w") as log:
        log.write("$ " + " ".join(args) + "\n")
        log.flush()
        returncode = subprocess.run(
            args, stdout=log, stderr=subprocess.STDOUT, cwd=cwd, env=env
        ).returncode
    if returncode != 0:
        raise TmivError(
            f"{Path(args[0]).name} failed (exit {returncode}); see {log_path}", log_path
        )
