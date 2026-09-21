"""Coherence-weighted soft equivariance residual + SO(2) plane energy floor."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sifess.models.equivariance import coherence_weighted_residual, so2_plane_energy


class SoftEquivarianceLoss(nn.Module):
    """L_eq with optional stop-grad on the framed target (default: on)."""

    def __init__(self, p: float = 2.0, stopgrad_target: bool = True):
        super().__init__()
        self.p = p
        self.stopgrad_target = stopgrad_target

    def forward(
        self,
        z_target: torch.Tensor,
        z_rho: torch.Tensor,
        cbar: torch.Tensor,
    ) -> torch.Tensor:
        return coherence_weighted_residual(
            z_target,
            z_rho,
            cbar,
            p=self.p,
            stopgrad_target=self.stopgrad_target,
        )


class PlaneEnergyLoss(nn.Module):
    """Keep energy in the SO(2) plane so fixed rho cannot be vacuously identity.

    L_plane = ReLU(plane_min - mean(z0^2 + z1^2)). Without this, an invariant
    encoder can dump all mass into dims >=2 and drive L_eq->0 even with fixed SO(2).
    """

    def __init__(self, plane_min: float = 0.05):
        super().__init__()
        self.plane_min = float(plane_min)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        energy = so2_plane_energy(z)
        return F.relu(self.plane_min - energy)


class OrthogonalityLoss(nn.Module):
    """Optional lambda_orth: encourage ||rho z|| ≈ ||z|| (norm preservation).

    For pure SO(2)-on-2D action with lambda_orth=0, this is unused (SO(2) is isometric
    on the plane and identity elsewhere, so norms match automatically when
    reflect is orthogonal).
    """

    def forward(self, z: torch.Tensor, z_rho: torch.Tensor) -> torch.Tensor:
        n1 = z.norm(dim=-1)
        n2 = z_rho.norm(dim=-1)
        return ((n1 - n2) ** 2).mean()
