# Sharp Monocular View Synthesis in Less Than a Second

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://apple.github.io/ml-sharp/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.10685-b31b1b.svg)](https://arxiv.org/abs/2512.10685)

This software project accompanies the research paper: _Sharp Monocular View Synthesis in Less Than a Second_
by _Lars Mescheder, Wei Dong, Shiwei Li, Xuyang Bai, Marcel Santos, Peiyun Hu, Bruno Lecouat, Mingmin Zhen, Amaël Delaunoy,
Tian Fang, Yanghai Tsin, Stephan Richter and Vladlen Koltun_.

![](data/teaser.jpg)

We present SHARP, an approach to photorealistic view synthesis from a single image. Given a single photograph, SHARP regresses the parameters of a 3D Gaussian representation of the depicted scene. This is done in less than a second on a standard GPU via a single feedforward pass through a neural network. The 3D Gaussian representation produced by SHARP can then be rendered in real time, yielding high-resolution photorealistic images for nearby views. The representation is metric, with absolute scale, supporting metric camera movements. Experimental results demonstrate that SHARP delivers robust zero-shot generalization across datasets. It sets a new state of the art on multiple datasets, reducing LPIPS by 25–34% and DISTS by 21–43% versus the best prior model, while lowering the synthesis time by three orders of magnitude.

> [!NOTE]
> **This is a fork of [`apple/ml-sharp`](https://github.com/apple/ml-sharp).**
> The README below is upstream Apple's, describing the SHARP model and its `sharp` CLI.
> This fork adds `sharp_spatialize`, a browser UI and CLI that turn a single photo into a
> 9-view spatial photo — see **[Spatial photo authoring](#spatial-photo-authoring-fork-addition)**.

## Getting started

There are two ways to install this repo, and they are **not** interchangeable — pick based
on what you want:

- **[Quick start](#quick-start-the-web-ui-and-spatial-photos)** — this fork's
  `sharp_spatialize` tools, including the web UI. Requires an NVIDIA GPU.
  **Start here if you want the web app.**
- **[Upstream install](#upstream-install-the-sharp-model-only)** — Apple's original SHARP
  model and `sharp` CLI only. Runs on CPU, CUDA, or MPS.

### Quick start: the web UI and spatial photos

#### 1. Check the requirements

| What | Why |
| --- | --- |
| **An NVIDIA GPU** | `gsplat`, the renderer, ships no CPU or MPS kernel. This is not optional. |
| **NVIDIA driver** | `nvidia-smi` must run. |
| **CUDA toolkit (`nvcc`)** | **The driver alone is not enough.** `gsplat` compiles its rasterizer from source on the first render, which needs the full toolkit. Ubuntu/Debian: `sudo apt install cuda-toolkit-13-0`; otherwise [download it](https://developer.nvidia.com/cuda-downloads). |
| **Python 3.13** | |
| [`uv`](https://docs.astral.sh/uv/) | Optional. Used if present; otherwise the installer falls back to `venv` + `pip`, which is slower. |

A missing CUDA toolkit is the single most common reason a fresh install gets all the way to
its first render and only then dies, so `setup.sh` checks for it before installing anything.

#### 2. Install

```
git clone https://github.com/proto-tech-macario/sharp.git
cd sharp
./setup.sh
```

`setup.sh` verifies the driver and `nvcc`, builds `.venv` from the pinned
`requirements-cu130.txt`, and then confirms that torch can actually see your GPU and that
`ninja` and `nvcc` are reachable. It fails early with a specific message rather than
letting you discover a broken toolchain later.

To re-verify an environment you already built, installing nothing:

```
./setup.sh --check
```

#### 3. Run the web UI

```
.venv/bin/sharp-spatialize-webui --open
```

This opens <http://127.0.0.1:8737/>. Drop in a photo, press **Generate spatial photo**, and
download the resulting `.h5`. There is a bundled sample image if you just want to see it
work. Every option is documented under
[Spatial photo authoring](#spatial-photo-authoring-fork-addition).

> [!IMPORTANT]
> **The first render takes roughly 5 minutes and prints nothing while it works.** That is
> `gsplat` compiling its CUDA extension; it is cached in `~/.cache/torch_extensions`, so
> every render after that takes seconds. The model checkpoint also downloads once, to
> `~/.cache/torch/hub/checkpoints/`. Nothing is stuck — let it finish.

To avoid prefixing every command with `.venv/bin/`, activate the environment:

```
source .venv/bin/activate
```

#### Why this fork pins its own dependencies

Upstream's `requirements.txt` pins torch 2.8 with the CUDA 12 runtime. This fork is
developed and tested against CUDA 13 / PyTorch 2.14, locked in `requirements-cu130.txt`,
because `gsplat` compiles its rasterizer against whatever torch and `nvcc` it finds at
runtime — mixing the two toolchains is exactly how a working checkout stops working on
another machine. Use `setup.sh` (which uses the CUDA 13 lockfile) rather than the upstream
instructions below if you want the spatial photo tools.

### Upstream install: the SHARP model only

Apple's original instructions, kept verbatim. These give you the `sharp` CLI and the model
on CPU, CUDA, or MPS — but **not** the `sharp_spatialize` tools above.

We recommend to first create a python environment:

```
conda create -n sharp python=3.13
```

Afterwards, you can install the project using

```
pip install -r requirements.txt
```

To test the installation, run

```
sharp --help
```

## Using the CLI

To run prediction:

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians
```

The model checkpoint will be downloaded automatically on first run and cached locally at `~/.cache/torch/hub/checkpoints/`.

Alternatively, you can download the model directly:

```
wget https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt
```

To use a manually downloaded checkpoint, specify it with the `-c` flag:

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians -c sharp_2572gikvuh.pt
```

The results will be 3D gaussian splats (3DGS) in the output folder. The 3DGS `.ply` files are compatible to various public 3DGS renderers. We follow the OpenCV coordinate convention (x right, y down, z forward). The 3DGS scene center is roughly at (0, 0, +z). When dealing with 3rdparty renderers, please scale and rotate to re-center the scene accordingly.

### Rendering trajectories (CUDA GPU only)

Additionally you can render videos with a camera trajectory. While the gaussians prediction works for all CPU, CUDA, and MPS, rendering videos via the `--render` option currently requires a CUDA GPU. The gsplat renderer takes a while to initialize at the first launch.

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians --render

# Or from the intermediate gaussians:
sharp render -i /path/to/output/gaussians -o /path/to/output/renderings
```

## Spatial photo authoring (fork addition)

> Not part of upstream `apple/ml-sharp`. Added in this fork as the `sharp_spatialize` package.

`sharp_spatialize` turns a single photo into a **spatial photo**: a 9-view capture — the
original viewpoint plus 8 cameras orbiting around it — written to one HDF5 file. It uses
SHARP for the 3D Gaussian prediction and `gsplat` to render the extra views.

**Rendering requires a CUDA GPU.** `gsplat` ships no CPU or MPS kernel, and every entry
point below renders, so a working NVIDIA driver is a hard requirement here even though
plain Gaussian prediction runs anywhere.

### Web UI

The quickest way in — drop a photo in the browser, set the camera angle, preview the
result, download the `.h5`:

```
sharp-spatialize-webui --open
```

That serves <http://127.0.0.1:8737/>. It is a standard-library `ThreadingHTTPServer`:
no framework, nothing extra to install, and it binds to loopback only. Generation runs on
a worker thread serialized by a GPU lock, so status stays responsive while the GPU is busy.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Bind address. |
| `--port` | `8737` | Port. |
| `-c`, `--checkpoint` | auto-download | SHARP checkpoint path. |
| `--device` | `default` | `cpu`, `mps`, `cuda`, or auto-detect. |
| `--precision` | `fp16` | `fp32`, `fp16`, or `bf16`. |
| `--open` | off | Open a browser on startup. |

In the page you can set the camera angle (2–20°), preview resolution (640–2048 px,
default 1280), and precision. Uploads accept JPEG, PNG, and HEIC up to 64 MB. Each finished
job shows the 9 rendered views, their depth maps, timings, and a validation badge, and
offers the `spatial_photo.h5` as a download. A bundled sample photo is offered when running
from a source checkout, so you can try it without supplying an image.

`python -m sharp_spatialize.webui` is equivalent to the `sharp-spatialize-webui` command.

### CLI

One image in, one spatial photo out:

```
sharp-spatialize -i photo.jpg -o photo_spatial.h5
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `-i`, `--input` | *required* | Input JPEG/PNG image. |
| `-o`, `--output` | *required* | Output `.h5` path. |
| `--angle` | `10.0` | Camera angle in degrees. Any positive float; 5, 10, and 15 are the validated values. |
| `--width` / `--height` | source size | Output dimensions in pixels. |
| `-c`, `--checkpoint` | auto-download | SHARP checkpoint path. |
| `--device` | `default` | `cpu`, `mps`, `cuda`, or auto-detect. |
| `--precision` | `fp32` | `fp32`, `fp16`, or `bf16`. `fp16` roughly halves peak VRAM. |

Note the default precision differs by entry point: the CLI defaults to `fp32` (the
reference), the web UI to `fp16` (faster, about half the VRAM).

To check an existing file — no SHARP or CUDA needed, so it works anywhere:

```
sharp-spatialize-validate photo_spatial.h5
```

It exits non-zero and lists each failure if the file is not a valid spatial photo.

### Installing

See the [Quick start](#quick-start-the-web-ui-and-spatial-photos) above: `./setup.sh`
builds the environment and checks the CUDA toolchain these tools need.

## Evaluation

Please refer to the paper for both quantitative and qualitative evaluations.
Additionally, please check out this [qualitative examples page](https://apple.github.io/ml-sharp/) containing several video comparisons against related work.

## Citation

If you find our work useful, please cite the following paper:

```bibtex
@inproceedings{Sharp2025:arxiv,
  title      = {Sharp Monocular View Synthesis in Less Than a Second},
  author     = {Lars Mescheder and Wei Dong and Shiwei Li and Xuyang Bai and Marcel Santos and Peiyun Hu and Bruno Lecouat and Mingmin Zhen and Ama\"{e}l Delaunoy and Tian Fang and Yanghai Tsin and Stephan R. Richter and Vladlen Koltun},
  journal    = {arXiv preprint arXiv:2512.10685},
  year       = {2025},
  url        = {https://arxiv.org/abs/2512.10685},
}
```

## Acknowledgements

Our codebase is built using multiple opensource contributions, please see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for more details.

## License

Please check out the repository [LICENSE](LICENSE) before using the provided code and
[LICENSE_MODEL](LICENSE_MODEL) for the released models.
