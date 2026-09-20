"""Structure tensor / isophote eigenframe + coherence.

J_ρ = G_ρ * (∇I ∇I^T), with Gaussian smoothing scale σ_ρ (default 1.5).
Eigenvalues λ1 ≥ λ2 ≥ 0; coherence C = (λ1 - λ2) / (λ1 + λ2 + ε).
Cbar = stopgrad(mean(C)) over spatial dims (per-image or batch as configured).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class StructureTensorResult:
    """Per-pixel (or pooled) structure-tensor quantities."""

    e1: torch.Tensor  # (B, 2, H, W) major eigenvector (edge-normal / gradient dir)
    e2: torch.Tensor  # (B, 2, H, W) minor eigenvector (isophote / tangent)
    lam1: torch.Tensor  # (B, 1, H, W)
    lam2: torch.Tensor  # (B, 1, H, W)
    C: torch.Tensor  # (B, 1, H, W) coherence in [0, 1]
    Cbar: torch.Tensor  # (B,) or scalar stopgrad mean coherence
    theta: torch.Tensor  # (B, 1, H, W) orientation of e1 in radians


def _gaussian_kernel1d(sigma: float, truncate: float = 3.0, device=None, dtype=None) -> torch.Tensor:
    radius = max(1, int(truncate * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    return k


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on (B, C, H, W)."""
    if sigma <= 0:
        return x
    dtype = x.dtype
    device = x.device
    k1d = _gaussian_kernel1d(sigma, device=device, dtype=dtype)
    # depthwise: reshape to (B*C, 1, H, W)
    b, c, h, w = x.shape
    x_ = x.reshape(b * c, 1, h, w)
    kx = k1d.view(1, 1, 1, -1)
    ky = k1d.view(1, 1, -1, 1)
    pad_x = kx.shape[-1] // 2
    pad_y = ky.shape[-2] // 2
    x_ = F.pad(x_, (pad_x, pad_x, 0, 0), mode="reflect")
    x_ = F.conv2d(x_, kx)
    x_ = F.pad(x_, (0, 0, pad_y, pad_y), mode="reflect")
    x_ = F.conv2d(x_, ky)
    return x_.reshape(b, c, h, w)


def _to_gray(images: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, 1, H, W) luminance-ish."""
    if images.shape[1] == 1:
        return images
    # Simple mean; medical CXR often replicated grayscale
    return images.mean(dim=1, keepdim=True)


def compute_structure_tensor(
    images: torch.Tensor,
    sigma_rho: float = 1.5,
    eps: float = 1e-6,
    pool: str = "mean",
) -> StructureTensorResult:
    """Compute isophote eigenframe and coherence from images.

    Args:
        images: (B, C, H, W) float tensor, any range (gradients scale-invariant in C).
        sigma_rho: Gaussian integration scale for J_ρ.
        eps: numerical stabilizer in C.
        pool: how to form Cbar — ``mean`` over spatial dims (default).

    Returns:
        StructureTensorResult with e1, e2, λ1, λ2, C, Cbar (stopgrad), theta.
    """
    if images.ndim != 4:
        raise ValueError(f"expected (B,C,H,W), got {tuple(images.shape)}")

    gray = _to_gray(images)
    # Sobel gradients
    kx = torch.tensor(
        [[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]],
        device=images.device,
        dtype=images.dtype,
    )
    ky = torch.tensor(
        [[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]],
        device=images.device,
        dtype=images.dtype,
    )
    Ix = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), kx)
    Iy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), ky)

    Jxx = _gaussian_blur(Ix * Ix, sigma_rho)
    Jxy = _gaussian_blur(Ix * Iy, sigma_rho)
    Jyy = _gaussian_blur(Iy * Iy, sigma_rho)

    # Analytic 2x2 eigen-decomposition
    # λ = 0.5 * (trace ± sqrt((Jxx-Jyy)^2 + 4 Jxy^2))
    trace = Jxx + Jyy
    diff = Jxx - Jyy
    disc = torch.sqrt(diff * diff + 4.0 * Jxy * Jxy + eps)
    lam1 = 0.5 * (trace + disc)
    lam2 = 0.5 * (trace - disc)
    lam2 = torch.clamp(lam2, min=0.0)

    # e1 (major): direction of strongest variation
    # For [[a,b],[b,c]], eigenvector for λ1: (b, λ1-a) or (λ1-c, b)
    ex = Jxy
    ey = lam1 - Jxx
    # Fallback when Jxy ~ 0 and Jxx is dominant / not
    alt_ex = lam1 - Jyy
    alt_ey = Jxy
    use_alt = (ex.abs() + ey.abs()) < (alt_ex.abs() + alt_ey.abs())
    ex = torch.where(use_alt, alt_ex, ex)
    ey = torch.where(use_alt, alt_ey, ey)
    # Normalize without biasing length via eps-inside-sqrt; default e1=(1,0) if degenerate
    norm = torch.sqrt(ex * ex + ey * ey)
    degenerate = norm < 1e-6
    norm_safe = norm.clamp_min(1e-6)
    e1x = ex / norm_safe
    e1y = ey / norm_safe
    e1x = torch.where(degenerate, torch.ones_like(e1x), e1x)
    e1y = torch.where(degenerate, torch.zeros_like(e1y), e1y)
    # e2 = rotate e1 by 90° (isophote / tangent): (-e1y, e1x)
    e2x = -e1y
    e2y = e1x

    e1 = torch.cat([e1x, e1y], dim=1)
    e2 = torch.cat([e2x, e2y], dim=1)

    energy = lam1 + lam2
    C = (lam1 - lam2) / (energy + eps)
    # Isotropic / empty neighborhoods: force low coherence when total energy is tiny
    C = torch.where(energy < 1e-8, torch.zeros_like(C), C)
    C = torch.clamp(C, 0.0, 1.0)

    if pool == "mean":
        Cbar = C.mean(dim=(1, 2, 3))
    elif pool == "none":
        Cbar = C.mean(dim=(1, 2, 3))  # still provide scalar per image
    else:
        raise ValueError(f"unknown pool={pool}")
    Cbar = Cbar.detach()  # stopgrad

    theta = torch.atan2(e1y, e1x)

    return StructureTensorResult(
        e1=e1,
        e2=e2,
        lam1=lam1,
        lam2=lam2,
        C=C,
        Cbar=Cbar,
        theta=theta,
    )


def pooled_frame_angle(
    result: StructureTensorResult,
    weight_by_coherence: bool = True,
) -> torch.Tensor:
    """Coherence-weighted pooled orientation (B,) in radians for global frame actions."""
    # Double-angle trick for orientation averaging
    th = result.theta  # (B,1,H,W)
    if weight_by_coherence:
        w = result.C
    else:
        w = torch.ones_like(result.C)
    c2 = torch.cos(2.0 * th)
    s2 = torch.sin(2.0 * th)
    wsum = w.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    mc = (w * c2).sum(dim=(1, 2, 3)) / wsum
    ms = (w * s2).sum(dim=(1, 2, 3)) / wsum
    return 0.5 * torch.atan2(ms, mc)
