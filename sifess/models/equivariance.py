"""Learned soft equivariance map rho(g) + C-weighted residual utilities."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class SoftEquivarianceHead(nn.Module):
    """Maps group parameters (angle, reflect) to a linear action on embeddings.

    rho(g): R^d -> R^d implemented as a predicted residual transform:
      z' = z + MLP([cos φ, sin φ, reflect])  (additive) or
      z' = W(g) z with low-rank W — we use FiLM-style scale/shift for stability:
      z' = (1 + s(g)) * z + b(g)
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


def coherence_weighted_residual(
    z_tgt: torch.Tensor,
    z_src_transformed: torch.Tensor,
    cbar: torch.Tensor,
    p: float = 2.0,
) -> torch.Tensor:
    """L_eq = mean( Cbar * || z_tgt - rho(g) z_src ||^p ).

    cbar: (B,) stopgrad coherence gate. Ablation no_c_gating sets cbar=1.
    """
    diff = z_tgt - z_src_transformed
    if p == 2.0:
        per = (diff ** 2).mean(dim=-1)
    else:
        per = diff.abs().pow(p).mean(dim=-1)
    w = cbar.reshape(-1).to(dtype=per.dtype)
    return (w * per).mean()
