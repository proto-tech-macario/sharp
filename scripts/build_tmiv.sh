#!/usr/bin/env bash
#
# Builds the MPEG MIV reference software (TMIV v24.0) with this repo's
# per-frame camera patch, for sharp_video's MIV encoder and validator.
#
#   scripts/build_tmiv.sh                 build into ./.tmiv
#   scripts/build_tmiv.sh --prefix DIR    build into DIR (then export SHARP_TMIV_DIR=DIR)
#   scripts/build_tmiv.sh -j 16           parallel build jobs
#
# Layout produced (what sharp_video.miv.tmiv expects):
#   DIR/src             patched TMIV checkout (encode.py, configs)
#   DIR/install/bin     TmivEncoder, TmivMultiplexer, TmivDecoder, vvencFFapp, ...
#   DIR/sharp_tmiv.json build marker
#
# Needs git, a C++17 compiler (GCC 13+/Clang on Linux, Xcode clang on macOS),
# Python 3.10+, and network access (TMIV downloads VVenC, VVdeC, HM, fmt, ...).
# No GPU is needed: MIV encoding and decoding run on the CPU.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="$REPO/.tmiv"
JOBS="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu || echo 4)"
TMIV_URL="https://gitlab.com/mpeg-i-visual/tmiv.git"
TMIV_REF="v24.0"
PATCH_NAME="sharp-per-frame-cameras"
PATCH="$REPO/third_party/tmiv/$PATCH_NAME.patch"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$(mkdir -p "$2" && cd "$2" && pwd)"; shift 2 ;;
    -j) JOBS="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die()  { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

for tool in git python3; do
  command -v "$tool" >/dev/null || die "$tool is required"
done

SRC="$PREFIX/src"
INSTALL="$PREFIX/install"
mkdir -p "$PREFIX"

step "Fetching TMIV $TMIV_REF"
if [[ ! -d "$SRC/.git" ]]; then
  git clone --quiet --depth 1 --branch "$TMIV_REF" "$TMIV_URL" "$SRC"
fi

step "Applying $PATCH_NAME"
if git -C "$SRC" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "already applied"
else
  # The checkout may carry an older revision of the patch: restore the pristine
  # $TMIV_REF sources first (this discards any hand edits to tracked files).
  git -C "$SRC" checkout --quiet -- .
  git -C "$SRC" apply "$PATCH"
fi

step "Preparing build tools"
VENV="$PREFIX/venv"
[[ -x "$VENV/bin/python" ]] || python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet cmake ninja
export PATH="$VENV/bin:$PATH"

# Our own presets: the stock ones plus tests disabled (not needed to run TMIV,
# and on newer toolchains its unit tests are the first thing to break).
DEPS="build_dependencies.json"
case "$(uname -s)" in
  Darwin)
    # TMIV ships Linux/Windows presets only. Apple clang also needs a newer fmt
    # than TMIV pins and a few new warnings silenced in the dependencies'
    # -Werror builds (HM: 'register'; VVdeC: nontrivial memcall; VVenC's
    # bundled nlohmann-json: literal operator spacing).
    PRESET="sharp-macos-release"
    python3 - "$SRC" <<'EOF'
import json, sys
src = sys.argv[1]
deps = json.load(open(f"{src}/build_dependencies.json"))
for dep in deps:
    if dep["name"] == "FMT":
        dep["git_ref"] = "11.2.0"
json.dump(deps, open(f"{src}/build_dependencies_sharp.json", "w"), indent=2)
EOF
    DEPS="build_dependencies_sharp.json"
    export CXXFLAGS="-Wno-register -Wno-nontrivial-memcall -Wno-deprecated-literal-operator -Wno-unknown-warning-option"
    cat > "$SRC/CMakeUserPresets.json" <<'EOF'
{
  "version": 3,
  "configurePresets": [
    {"name": "sharp-macos-release", "inherits": ["release"],
     "cacheVariables": {"CMAKE_CXX_COMPILER": "clang++", "CMAKE_C_COMPILER": "clang",
                        "BUILD_TESTING": "OFF"}}
  ]
}
EOF
    ;;
  Linux)
    # TMIV's gcc preset compiles with -Werror; a newer GCC than TMIV's CI adds
    # warnings that would stop the build, so keep its warnings but not -Werror.
    # HM (a dependency) still uses the C++17-removed 'register' keyword.
    PRESET="sharp-linux-release"
    export CXXFLAGS="-Wno-register"
    cat > "$SRC/CMakeUserPresets.json" <<'EOF'
{
  "version": 3,
  "configurePresets": [
    {"name": "sharp-linux-release", "inherits": ["gcc-release"],
     "cacheVariables": {
       "BUILD_TESTING": "OFF",
       "CMAKE_CXX_FLAGS": "-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wno-maybe-uninitialized -Wold-style-cast -Wno-alloc-size-larger-than"
     }}
  ]
}
EOF
    ;;
  *) die "unsupported OS $(uname -s); build TMIV by hand and set SHARP_TMIV_DIR" ;;
esac

step "Building TMIV ($PRESET, $JOBS jobs) -- this takes a while the first time"
(cd "$SRC" && python scripts/install.py "$PRESET" \
   --build-dependencies-file "$DEPS" --skip-tests -i "$INSTALL" -j "$JOBS")

for exe in TmivEncoder TmivMultiplexer TmivDecoder TmivParser vvencFFapp; do
  [[ -x "$INSTALL/bin/$exe" ]] || die "build finished but $INSTALL/bin/$exe is missing"
done

cat > "$PREFIX/sharp_tmiv.json" <<EOF
{"version": "$TMIV_REF", "patch": "$PATCH_NAME", "source": "$SRC", "install": "$INSTALL"}
EOF

step "Done"
echo "TMIV installed in $INSTALL"
[[ "$PREFIX" == "$REPO/.tmiv" ]] || echo "export SHARP_TMIV_DIR=$PREFIX"
