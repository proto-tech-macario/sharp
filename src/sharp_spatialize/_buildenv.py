"""Makes gsplat's JIT toolchain reachable however Python was launched.

gsplat compiles its CUDA extension on the first render by shelling out to
`ninja`, which it looks for on PATH. But a virtualenv's `bin/` directory is
only on PATH if the environment was *activated*: launching the interpreter by
absolute path (`.venv/bin/python -m sharp_spatialize.webui`, an editor's
configured interpreter, a systemd unit, a container ENTRYPOINT) leaves `ninja`
installed but invisible, and the first render dies with

    RuntimeError: Ninja is required to load C++ extensions (pip install ninja
    to get it)

which is thoroughly misleading -- ninja is right there in `.venv/bin`.

So we put the running interpreter's own `bin/` directory on PATH ourselves,
and do the same for the CUDA toolkit's `bin/` (for `nvcc`), setting CUDA_HOME
if the caller has not. All of it is best-effort and idempotent: anything the
caller already configured wins, and a machine with no CUDA toolkit is left
untouched so the CPU/MPS inference path still imports cleanly.
"""

from __future__ import annotations

import os
import shutil
import sys
import sysconfig
from pathlib import Path

#: Where a CUDA toolkit usually lands when it is not already on PATH.
_CUDA_HOME_CANDIDATES = ("/usr/local/cuda", "/opt/cuda", "/usr/lib/cuda")


def _prepend_to_path(directory: Path) -> None:
    """Put `directory` first on PATH, unless it is already somewhere on it."""
    entry = str(directory)
    existing = os.environ.get("PATH", "")
    if entry in existing.split(os.pathsep):
        return
    os.environ["PATH"] = f"{entry}{os.pathsep}{existing}" if existing else entry


def _find_cuda_home() -> Path | None:
    """Locate the CUDA toolkit: caller's setting, then `nvcc`, then the usual spots."""
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        configured = os.environ.get(variable)
        if configured and (Path(configured) / "bin").is_dir():
            return Path(configured)

    nvcc = shutil.which("nvcc")
    if nvcc:
        return Path(nvcc).resolve().parent.parent

    for candidate in _CUDA_HOME_CANDIDATES:
        if (Path(candidate) / "bin" / "nvcc").is_file():
            return Path(candidate)
    return None


def _script_directories() -> list[Path]:
    """The directories console scripts (`ninja`) live in for this interpreter.

    Deliberately *not* resolved: in a virtualenv `sys.executable` is a symlink
    to the base interpreter, so resolving it would walk us out of `.venv/bin`
    and straight past the very `ninja` we are looking for. `sysconfig` names
    the scripts directory properly (`Scripts` on Windows, `bin` elsewhere);
    the executable's own directory is the belt-and-braces fallback.
    """
    directories = [Path(sys.executable).parent]
    scripts = sysconfig.get_path("scripts")
    if scripts:
        directories.insert(0, Path(scripts))
    return [directory for directory in directories if directory.is_dir()]


def ensure_build_toolchain() -> None:
    """Make `ninja` and `nvcc` findable for gsplat's first-render JIT compile.

    Safe to call repeatedly and safe to call on a machine with no CUDA
    toolkit; it only ever adds to PATH and never overrides CUDA_HOME.
    """
    for directory in reversed(_script_directories()):
        _prepend_to_path(directory)

    cuda_home = _find_cuda_home()
    if cuda_home is None:
        return
    _prepend_to_path(cuda_home / "bin")
    os.environ.setdefault("CUDA_HOME", str(cuda_home))
