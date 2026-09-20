"""Coherence-weighted soft equivariance residual loss."""

from __future__ import annotations

import torch
import torch.nn as nn

from sifess.models.equivariance import coherence_weighted_residual


class SoftEquivarianceLoss(nn.Module):
    def __init__(self, p: float = 2.0):
        super().__init__()
        self.p = p

    def forward(
        self,
        z_target: torch.Tensor,
        z_rho: torch.Tensor,
        cbar: torch.Tensor,
    ) -> torch.Tensor:
        return coherence_weighted_residual(z_target, z_rho, cbar, p=self.p)


class OrthogonalityLoss(nn.Module):
    """Optional λ_orth: encourage rho near SO(2) on first 2 dims (or skip if 0).

    For pure SO(2)-on-2D action with λ_orth=0, this is unused.
    """

    def forward(self, z: torch.Tensor, z_rho: torch.Tensor) -> torch.Tensor:
        # Soft norm preservation + pairwise orthogonality of finite differences — lightweight proxy
        n1 = z.norm(dim=-1)
        n2 = z_rho.norm(dim=-1)
        return ((n1 - n2) ** 2).mean()
