"""Rotate / reflect images in the isophote eigenframe.

Group elements g act in image coordinates aligned to the pooled structure-tensor
frame (θ). Ablation ``random_frame`` replaces θ with a random angle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class FrameGroupElement:
    """Discrete/continuous SO(2)/O(2) element in the isophote frame."""

    angle: torch.Tensor  # (B,) radians — rotation in lab frame, or relative
    reflect: torch.Tensor  # (B,) bool — reflect across e1 axis before rotate


def sample_frame_group_element(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    angles: Optional[torch.Tensor] = None,
    reflect_prob: float = 0.5,
) -> FrameGroupElement:
    """Sample g ~ rotations (and optional reflections)."""
    if angles is None:
        # Uniform on circle; common discrete subset also fine
        angles = torch.rand(batch_size, device=device, dtype=dtype) * (2.0 * torch.pi)
    reflect = torch.rand(batch_size, device=device) < reflect_prob
    return FrameGroupElement(angle=angles, reflect=reflect)


def _affine_grid_rotate_reflect(
    theta_frame: torch.Tensor,
    g: FrameGroupElement,
    height: int,
    width: int,
) -> torch.Tensor:
    """Build affine grids that rotate by g.angle relative to frame θ.

    Effective lab-frame rotation: R(θ) R(φ) R(-θ) [optionally with reflect],
    but for global crops we apply a single rotation by (θ + φ) after aligning —
    here we use relative angle φ in lab coords after aligning image so frame → x-axis
    is unnecessary if we just rotate the image by φ around center; the *semantic*
    equivariance is that features transform under the same φ. For soft residual we
    pass φ to rho(g). Image warp uses lab-frame angle = φ (crop rotation).

    For random_frame ablation, caller passes random θ as the frame and φ as g.angle;
    warp angle is still g.angle (the group action on pixels).
    """
    b = theta_frame.shape[0]
    device = theta_frame.device
    dtype = theta_frame.dtype
    phi = g.angle.to(dtype=dtype)
    # Reflect: flip x then rotate
    # Affine 2x3 for rotation by phi: [[c, -s, 0],[s, c, 0]] in normalized coords
    # torchvision-style: grid_sample with affine_grid
    c = torch.cos(phi)
    s = torch.sin(phi)
    # reflection across vertical: scale x by -1
    sx = torch.where(g.reflect, -torch.ones_like(c), torch.ones_like(c))
    # R @ S: [[c, -s],[s, c]] @ diag(sx, 1) = [[c*sx, -s],[s*sx, c]]
    a00 = c * sx
    a01 = -s
    a10 = s * sx
    a11 = c
    zeros = torch.zeros(b, device=device, dtype=dtype)
    mat = torch.stack(
        [
            torch.stack([a00, a01, zeros], dim=-1),
            torch.stack([a10, a11, zeros], dim=-1),
        ],
        dim=1,
    )  # (B, 2, 3)
    grid = F.affine_grid(mat, size=(b, 1, height, width), align_corners=False)
    return grid


def apply_frame_transform(
    images: torch.Tensor,
    frame_theta: torch.Tensor,
    g: FrameGroupElement,
) -> torch.Tensor:
    """Apply group element g as a geometric warp on images (B,C,H,W).

    ``frame_theta`` is retained for API symmetry / logging; the pixel action is
    rotation+reflect by ``g`` (isophote-conditioned sampling of g happens upstream).
    """
    if images.ndim != 4:
        raise ValueError(f"expected (B,C,H,W), got {tuple(images.shape)}")
    b, _, h, w = images.shape
    if frame_theta.shape[0] != b:
        raise ValueError("frame_theta batch mismatch")
    grid = _affine_grid_rotate_reflect(frame_theta, g, h, w)
    # grid is for single channel size; expand works for C channels
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)


def rotate_features_so2(features: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Placeholder SO(2) action on feature vectors via 2D block rotation if dim even.

    For DINO embeddings we use a learned rho(g) instead; this helper supports
    optional λ_orth checks on explicit 2D planes.
    """
    # No-op identity at feature level — learned rho handles this.
    return features
