#!/usr/bin/env bash
#
# Builds the .venv this project is developed against and checks the pieces
# that only fail later, at the first render, when they are missing.
#
#   ./setup.sh          build the venv and verify the toolchain
#   ./setup.sh --check  verify an existing venv, build nothing
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

VENV=".venv"
LOCK="requirements-cu130.txt"
CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
warn()  { printf '\033[33m%s\033[0m\n' "$*"; }
step()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# --------------------------------------------------------------- prerequisites

step "Checking prerequisites"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  red "nvidia-smi not found. SHARP's renderer (gsplat) has no CPU or MPS kernel,"
  red "so a working NVIDIA driver is required. Install the driver and re-run."
  exit 1
fi
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
green "GPU: ${GPU}"

# gsplat compiles its rasterizer from source on the first render, which needs a
# full CUDA toolkit (nvcc) -- not just the driver, and not just the CUDA runtime
# that ships inside the torch wheel. This is the single most common reason a
# fresh install gets all the way to "rendering" and then dies.
NVCC=""
for candidate in "${CUDA_HOME:-}/bin/nvcc" "$(command -v nvcc || true)" \
                 /usr/local/cuda/bin/nvcc /opt/cuda/bin/nvcc; do
  [[ -n "${candidate}" && -x "${candidate}" ]] && { NVCC="${candidate}"; break; }
done

if [[ -z "${NVCC}" ]]; then
  red "No nvcc found. gsplat JIT-compiles its CUDA extension on first render and"
  red "needs the CUDA toolkit, which the NVIDIA driver alone does not provide."
  red ""
  red "  Ubuntu/Debian:  sudo apt install cuda-toolkit-13-0"
  red "  or download:    https://developer.nvidia.com/cuda-downloads"
  red ""
  red "Then re-run, or set CUDA_HOME to an existing toolkit."
  exit 1
fi
NVCC_VER=$("${NVCC}" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
green "nvcc: ${NVCC} (CUDA ${NVCC_VER})"

# ---------------------------------------------------------------------- install

if [[ ${CHECK_ONLY} -eq 0 ]]; then
  step "Creating ${VENV} and installing pinned dependencies"

  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.13 "${VENV}"
    uv pip install --python "${VENV}/bin/python" -r "${LOCK}"
  else
    warn "uv not found, falling back to python3 -m venv + pip (slower)."
    command -v python3.13 >/dev/null 2>&1 \
      && python3.13 -m venv "${VENV}" \
      || python3 -m venv "${VENV}"
    "${VENV}/bin/python" -m pip install --upgrade pip
    "${VENV}/bin/python" -m pip install -r "${LOCK}"
  fi
fi

[[ -x "${VENV}/bin/python" ]] || { red "No ${VENV}/bin/python -- run without --check first."; exit 1; }

# --------------------------------------------------------------- verification

step "Verifying the installed environment"

"${VENV}/bin/python" - <<'PY'
import shutil
import sys

# Importing the package puts the venv's bin/ and the CUDA toolkit on PATH; see
# sharp_spatialize/_buildenv.py for why that is not something we can leave to
# the caller's shell.
import sharp_spatialize  # noqa: F401
import torch

ok = True

print(f"  torch          {torch.__version__}")
print(f"  torch CUDA     {torch.version.cuda}")

if not torch.cuda.is_available():
    print("  FAIL: torch cannot see a CUDA device.")
    ok = False
else:
    print(f"  device         {torch.cuda.get_device_name(0)}")

ninja = shutil.which("ninja")
nvcc = shutil.which("nvcc")
print(f"  ninja          {ninja}")
print(f"  nvcc           {nvcc}")
if not ninja:
    print("  FAIL: ninja is not on PATH even after the _buildenv fixup.")
    ok = False
if not nvcc:
    print("  FAIL: nvcc is not on PATH even after the _buildenv fixup.")
    ok = False

# gsplat builds against the CUDA major version torch was compiled for. A
# toolkit from a different major line usually fails to compile, and when it
# does compile the kernel tends to misbehave at runtime -- so say so now.
if nvcc and torch.version.cuda:
    import re
    import subprocess

    released = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
    match = re.search(r"release (\d+)\.", released)
    if match and match.group(1) != torch.version.cuda.split(".")[0]:
        print(
            f"  WARNING: nvcc is CUDA {match.group(1)}.x but torch was built "
            f"against CUDA {torch.version.cuda}. gsplat compiles against torch, "
            "so install a matching toolkit if the first render fails."
        )

sys.exit(0 if ok else 1)
PY

step "Ready"
cat <<'EOF'
Start the web UI:

    .venv/bin/sharp-spatialize-webui --open

The first render compiles gsplat's CUDA extension and takes about 5 minutes
with no output; every render afterwards is seconds. The compiled kernel is
cached in ~/.cache/torch_extensions.
EOF
