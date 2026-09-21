"""Equivariance maps rho(g): fixed SO(2)-on-2D (default) or learned FiLM."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from sifess.geometry.frame_transforms import rotate_features_so2


class SoftEquivarianceHead(nn.Module):
    """Learned FiLM-style rho(g) (legacy / ablation). Prefer FixedSO2Action.

    rho(g): R^d -> R^d via scale/shift conditioned on (cos φ, sin φ, reflect):
      z' = (1 + s(g)) * z + b(g)

    Identity init (s≈0, b≈0) + DINO invariance made Day-0 V2 L_eq collapse to 0.
    """

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * dim),
        )
        # Init near identity: s≈0, b≈0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        z: torch.Tensor,
        angles: torch.Tensor,
        reflect: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply rho(g) to embeddings z (B, D)."""
        if reflect is None:
            reflect = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        else:
            reflect = reflect.float()
        g_feat = torch.stack(
            [torch.cos(angles), torch.sin(angles), reflect],
            dim=-1,
        )
        sb = self.net(g_feat)
        s, b = sb.chunk(2, dim=-1)
        return (1.0 + s) * z + b


class FixedSO2Action(nn.Module):
    """Non-learned SO(2)/O(2) on the first 2 embedding dimensions.

    Guarantees a material feature-space action whenever φ ≢ 0 (mod 2π) and the
    SO(2) plane has energy — unlike FiLM, it cannot collapse to identity.
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim

    def forward(
        self,
        z: torch.Tensor,
        angles: torch.Tensor,
        reflect: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return rotate_features_so2(z, angles, reflect)


def build_rho(eq_action: str, dim: int = 256) -> nn.Module:
    """Factory: ``so2_2d`` (default) or ``film`` (legacy learned)."""
    key = (eq_action or "so2_2d").lower()
    if key in ("so2_2d", "so2", "fixed_so2"):
        return FixedSO2Action(dim=dim)
    if key in ("film", "learned", "soft"):
        return SoftEquivarianceHead(dim=dim)
    raise ValueError(f"unknown eq_action={eq_action!r}; use so2_2d|film")


def coherence_weighted_residual(
    z_tgt: torch.Tensor,
    z_src_transformed: torch.Tensor,
    cbar: torch.Tensor,
    p: float = 2.0,
    stopgrad_target: bool = True,
) -> torch.Tensor:
    """L_eq = mean( Cbar * || sg(z_tgt) - rho(g) z_src ||^p ).

    **Gradient / stop-grad choice (BYOL-style, documented):**
    We *predict the framed embedding from the source* via rho:
      target = stopgrad(z(g·x)),  prediction = rho(g) z(x).
    Gradients flow through ``z_src`` and rho (if learned), not through the
    framed student embed. This blocks the trivial joint collapse
    z(g·x)≈z(x)∧rho→I that zeroed Day-0 V2 L_eq after epoch 1.

    Uses the vector Lp-norm (sum over feature dim), not a per-dim mean — the
    latter diluted the loss by ~D and made logged L_eq print as 0.0000.

    cbar: (B,) stopgrad coherence gate. Ablation no_c_gating sets cbar=1.
    """
    if stopgrad_target:
        z_tgt = z_tgt.detach()
    diff = z_tgt - z_src_transformed
    # ||v||_2^p  (p=2 → squared Euclidean); do NOT mean over feature dim
    if p == 2.0:
        per = (diff ** 2).sum(dim=-1)
    else:
        per = diff.norm(p=2, dim=-1).pow(p)
    w = cbar.reshape(-1).to(dtype=per.dtype)
    return (w * per).mean()


def so2_plane_energy(z: torch.Tensor) -> torch.Tensor:
    """Mean energy in the first 2 dims (SO(2) plane)."""
    return (z[:, 0] ** 2 + z[:, 1] ** 2).mean()
