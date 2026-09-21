"""Combined objective: L = L_DINO + lambda_eq L_eq (+ lambda_orth + lambda_plane)."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from sifess.losses.dino import DINOLoss
from sifess.losses.equivariance import OrthogonalityLoss, PlaneEnergyLoss, SoftEquivarianceLoss


class SIFESSLoss(nn.Module):
    def __init__(
        self,
        out_dim: int = 4096,
        lambda_eq: float = 1.0,
        lambda_orth: float = 0.0,
        lambda_plane: float = 0.1,
        plane_min: float = 0.05,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        stopgrad_target: bool = True,
    ):
        super().__init__()
        self.lambda_eq = lambda_eq
        self.lambda_orth = lambda_orth
        self.lambda_plane = lambda_plane
        self.dino = DINOLoss(out_dim=out_dim, teacher_temp=teacher_temp, student_temp=student_temp)
        self.eq = SoftEquivarianceLoss(stopgrad_target=stopgrad_target)
        self.orth = OrthogonalityLoss()
        self.plane = PlaneEnergyLoss(plane_min=plane_min)

    def forward(
        self,
        student_logits: Sequence[torch.Tensor],
        teacher_logits: Sequence[torch.Tensor],
        z_src: Optional[torch.Tensor] = None,
        z_tgt: Optional[torch.Tensor] = None,
        z_rho: Optional[torch.Tensor] = None,
        cbar: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        l_dino = self.dino(student_logits, teacher_logits)
        out: Dict[str, torch.Tensor] = {"l_dino": l_dino, "loss": l_dino}

        if (
            self.lambda_eq > 0
            and z_src is not None
            and z_tgt is not None
            and z_rho is not None
            and cbar is not None
        ):
            l_eq = self.eq(z_tgt, z_rho, cbar)
            out["l_eq"] = l_eq
            out["loss"] = out["loss"] + self.lambda_eq * l_eq
            if self.lambda_orth > 0:
                l_orth = self.orth(z_src, z_rho)
                out["l_orth"] = l_orth
                out["loss"] = out["loss"] + self.lambda_orth * l_orth
            if self.lambda_plane > 0:
                l_plane = self.plane(z_src)
                out["l_plane"] = l_plane
                out["loss"] = out["loss"] + self.lambda_plane * l_plane
        return out
