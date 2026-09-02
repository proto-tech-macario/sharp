"""Renders RGB, depth, and validity masks for a camera rig.

Uses SHARP's Gaussian rasterizer (`sharp.utils.gsplat`).

Requires a CUDA GPU -- gsplat has no CPU/MPS kernel, the same constraint
`sharp.cli.render` already enforces for `sharp render`.
"""

from __future__ import annotations

import numpy as np
import torch
from sharp.utils.gaussians import Gaussians3D
from sharp.utils.gsplat import GSplatRenderer

from .cameras import CameraPose

DEFAULT_MASK_ALPHA_THRESHOLD = 0.5


def render_views(
    gaussians: Gaussians3D,
    rig: list[CameraPose],
    output_width: int,
    output_height: int,
    mask_alpha_threshold: float = DEFAULT_MASK_ALPHA_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render every camera in `rig` from the same `gaussians` scene.

    Returns (rgb [N,H,W,3] uint8, depth [N,H,W] float32, mask [N,H,W] uint8),
    where N = len(rig). Depth is camera-space Z; invalid (low-alpha) pixels
    get depth forced to 0.0 and mask 0, so no output pixel is ever NaN/Inf.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "sharp-spatialize requires a CUDA GPU for view rendering "
            "(gsplat has no CPU/MPS kernel)."
        )
    device = torch.device("cuda")
    gaussians_cuda = gaussians.to(device)
    renderer = GSplatRenderer(color_space="linearRGB")

    rgb_frames, depth_frames, mask_frames = [], [], []
    for pose in rig:
        r_t = torch.from_numpy(pose.R).to(device=device, dtype=torch.float32)
        c_t = torch.from_numpy(pose.C).to(device=device, dtype=torch.float32)
        k_t = torch.from_numpy(pose.K).to(device=device, dtype=torch.float32)

        extrinsics = torch.eye(4, dtype=torch.float32, device=device)
        extrinsics[:3, :3] = r_t
        extrinsics[:3, 3] = -(r_t @ c_t)

        intrinsics = torch.eye(4, dtype=torch.float32, device=device)
        intrinsics[:3, :3] = k_t

        output = renderer(
            gaussians_cuda,
            extrinsics=extrinsics[None],
            intrinsics=intrinsics[None],
            image_width=output_width,
            image_height=output_height,
        )

        alpha = output.alpha[0, 0]
        mask = alpha > mask_alpha_threshold
        depth = torch.where(mask, output.depth[0, 0], torch.zeros_like(output.depth[0, 0]))
        rgb = (output.color[0].permute(1, 2, 0).clamp(0.0, 1.0) * 255.0).to(torch.uint8)

        rgb_frames.append(rgb.cpu().numpy())
        depth_frames.append(depth.cpu().numpy().astype(np.float32))
        mask_frames.append(mask.cpu().numpy().astype(np.uint8))

    return np.stack(rgb_frames), np.stack(depth_frames), np.stack(mask_frames)
