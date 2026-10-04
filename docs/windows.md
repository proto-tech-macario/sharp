# Running Stage 2 on Windows (WSL2)

This guide takes a Windows PC with an NVIDIA GPU from nothing to converting a
video into an MIV spatial video. It also covers getting back to Stage 1 at any
time (see [Going back to Stage 1](#going-back-to-stage-1)).

**Why WSL2 and not native Windows.** The setup scripts are bash. Stage 1's
renderer (gsplat) compiles CUDA code on first use, and MPEG's MIV software
(TMIV) is built by `scripts/build_tmiv.sh`. Neither step is set up for native
Windows. WSL2 runs a real Ubuntu that uses your Windows NVIDIA driver, so the
Linux path works unchanged.

**What you need:** Windows 11 (or Windows 10 21H2+), an NVIDIA GPU (8 GB+
recommended), about 40 GB free disk, admin rights once, and an internet
connection.

---

## 1. Install WSL2 with Ubuntu 24.04

Open **PowerShell as Administrator**:

```powershell
wsl --install -d Ubuntu-24.04
```

Reboot when asked. Ubuntu then opens and asks you to create a Linux user name
and password. Later, open it any time from the Start menu (**Ubuntu 24.04**).

Use Ubuntu **24.04**: its default compiler (GCC 13) is what TMIV requires.

If WSL was already installed, bring it up to date:

```powershell
wsl --update
```

## 2. NVIDIA driver: Windows side only

Install the latest NVIDIA driver **on Windows** (GeForce Experience / NVIDIA
App, or nvidia.com/drivers). Do **not** install an NVIDIA driver inside Ubuntu.
WSL uses the Windows driver.

Check from the Ubuntu terminal:

```bash
nvidia-smi
```

It must list your GPU. If it doesn't, see [Troubleshooting](#troubleshooting).

## 3. CUDA toolkit inside Ubuntu (for `nvcc`)

gsplat compiles its CUDA kernel on the first render and needs the full toolkit,
not just the driver. Use NVIDIA's **WSL-Ubuntu** repository, which contains the
toolkit but no driver:

```bash
cd ~
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-13-0
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version        # should report release 13.x
```

CUDA 13 matches the PyTorch build pinned in `requirements-cu130.txt`.

## 4. Build tools and uv

```bash
sudo apt install -y build-essential git curl python3 python3-venv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

`uv` fetches Python 3.13 for the project automatically.

## 5. Get the code

Clone **inside Ubuntu**, into your Linux home directory, **not** under `/mnt/c`.
Two reasons:

- A clone made with Windows Git converts line endings to CRLF, which breaks the bash scripts.
- Builds under `/mnt/c` are many times slower.

```bash
cd ~
git clone https://github.com/proto-tech-macario/sharp.git
cd sharp
git switch stage2-video-to-miv
```

| Ref | What it is |
|---|---|
| `stage2-video-to-miv` | Stage 1 + Stage 2 (this guide) |
| `main` | Stage 1 only, unchanged |
| tag `stage1` | frozen snapshot of Stage 1 (same commit as `main` when Stage 2 was added) |

## 6. Stage 1 environment

```bash
./setup.sh
```

This checks `nvidia-smi` and `nvcc`, builds `.venv` from the pinned lockfile, and
verifies that PyTorch sees the GPU and that Stage 2's packages import. It
fails early with a specific message if anything is missing.

## 7. TMIV (the MIV encoder/decoder)

```bash
scripts/build_tmiv.sh
```

This downloads MPEG's TMIV v24.0, applies this repo's patch
(`third_party/tmiv/sharp-per-frame-cameras.patch`), and builds it into `./.tmiv`.
It takes 20–40 minutes the first time and uses the CPU only. Re-running is
incremental.

> [!NOTE]
> `build_tmiv.sh` has been verified end to end on macOS. Its Linux/WSL path uses
> TMIV's own GCC preset (without `-Werror`) but had not been run on Linux when
> this guide was written. If it fails, see [Troubleshooting](#troubleshooting).

## 8. Verify

```bash
.venv/bin/python -m pytest tests/sharp_video -q -rs
```

This runs the unit tests plus the real-TMIV tests (encode → independent decode →
compare) and the real-SHARP end-to-end test (`test_video_e2e.py`: video →
Stage 1 on the GPU → MIV → TMIV decoder). Expect every test to pass with no
"skipped" lines. A skip names what's missing: the GPU or the TMIV build.

The first GPU test compiles gsplat's CUDA extension (about 5 minutes, silent)
and downloads the SHARP checkpoint (about 2.6 GB). Both are cached afterwards.

Stage 1's own tests still pass on this branch:

```bash
.venv/bin/python -m pytest tests/sharp_spatialize -q
```

`test_e2e.py` runs real SHARP on the GPU and takes a few minutes. For a quick
check, add `-m "not slow"` to skip it.

## 9. Convert a video

Windows drives appear under `/mnt/c`. Your outputs are visible from Windows
Explorer at `\\wsl$\Ubuntu-24.04\home\<linux-user>\`.

```bash
mkdir -p ~/out
.venv/bin/sharp_video_to_miv \
    -i /mnt/c/Users/<you>/Videos/clip.mp4 \
    -o ~/out/clip.miv \
    --dump-hdf5 ~/out/clip.h5 \
    --output-resolution 1280x720 \
    --max-frames 60 \
    --validate
```

Start small (`--max-frames 60`, 720p). Each frame is a full SHARP run plus 9
renders. You get:

| File | What |
|---|---|
| `clip.miv` | the MIV spatial video |
| `clip.miv.json` | manifest: every frame's timestamp, conventions, depth range |
| `clip.miv.report.md` / `.json` | performance report (per-frame SHARP / render time, GPU memory, bitrate, FPS) |
| `clip.h5` | the 9-view spatial sequence (debugging, re-encoding) |
| `clip.miv.work/` | intermediate files, per-frame cache, journal |

If a run stops (a frame fails, a crash, Ctrl+C), fix the cause and add `--resume`.
Finished frames are reused. Useful tools:

```bash
.venv/bin/sharp-video-validate ~/out/clip.h5 --temporal-json ~/out/clip.temporal.json
.venv/bin/sharp-video-inspect  ~/out/clip.h5 --frame 0 --frame 30 --out ~/out/inspect
.venv/bin/sharp-miv-validate   ~/out/clip.miv --reference ~/out/clip.h5
```

### The spec §34 test matrix

Put one clip per category in a folder, named by category:

```
static_office.mp4  camera_motion_street.mp4  moving_object_dog.mp4
camera_object_skate.mp4  difficult_foliage.mp4
```

```bash
.venv/bin/python scripts/stage2/run_test_matrix.py ~/clips ~/matrix -- \
    --max-frames 120 --output-resolution 1280x720
```

`~/matrix/matrix_report.md` summarizes every clip: pass/fail, frames, MIV size,
bitrate, processing FPS, and flicker. The `.miv` files it writes are the
representative MIV test files of spec §37.4.

---

## Going back to Stage 1

Stage 2 was added **only** on the branch `stage2-video-to-miv`. `main` and the
tag `stage1` are Stage 1, untouched, and Stage 2 changes nothing under
`src/sharp/` or `src/sharp_spatialize/`. You can confirm that at any time; this
prints nothing:

```bash
git diff stage1 stage2-video-to-miv -- src/sharp src/sharp_spatialize
```

To switch this checkout back to Stage 1:

```bash
cd ~/sharp
git switch main                  # or: git checkout stage1  (read-only snapshot)
rm -rf .venv && ./setup.sh       # rebuild the environment from Stage 1's lockfile
```

To return to Stage 2, run `git switch stage2-video-to-miv`, then
`rm -rf .venv && ./setup.sh`.

What Stage 2 adds, and how to remove each part:

| Added by Stage 2 | Remove with |
|---|---|
| code, docs, scripts (branch only) | `git switch main` |
| `av` (PyAV) in `.venv` | `rm -rf .venv && ./setup.sh` on `main` |
| TMIV build in `.tmiv/` (several GB) | `rm -rf .tmiv` |
| run outputs, `*.work/` | delete them |

On GitHub, Stage 1 is also always available as the `stage1` tag and the `main` branch.

---

## Troubleshooting

**`nvidia-smi` not found / no GPU inside Ubuntu.** Update the Windows NVIDIA
driver, run `wsl --update` in PowerShell, then `wsl --shutdown` and reopen
Ubuntu.

**`nvcc` not found.** Run `source ~/.bashrc`, or `export CUDA_HOME=/usr/local/cuda`,
then re-run `./setup.sh`.

**`./setup.sh: /bin/bash^M: bad interpreter`.** The repo was cloned with
Windows Git (CRLF line endings). Delete it and clone again inside Ubuntu
(step 5).

**`build_tmiv.sh` fails while compiling.** Check the compiler first:
`gcc --version` must be 13 or newer (Ubuntu 24.04's default is). Re-running
continues where it stopped. If it still fails, the first `error:` line in the
output names the file and the problem.

**Out of GPU memory.** Keep `--worker-count 1` (the default), lower
`--output-resolution`, and keep `--precision fp16` (the default).

**First render seems stuck.** gsplat is compiling its CUDA kernel. That takes
about 5 minutes with no output, once.

**WSL uses too much RAM/disk.** Create `C:\Users\<you>\.wslconfig` with, for
example, `[wsl2]` and `memory=16GB` on the next line. Then run `wsl --shutdown`.
