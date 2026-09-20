"""DINO-style student–teacher with optional SIFESS soft-equivariance head.

Single backbone family: ResNet-18 (smoke) / ResNet-50 (default full runs).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def build_backbone(name: str = "resnet50", pretrained: bool = False) -> Tuple[nn.Module, int]:
    name = name.lower()
    weights = None
    if name == "resnet18":
        m = models.resnet18(weights=weights)
        dim = m.fc.in_features
        m.fc = nn.Identity()
    elif name == "resnet50":
        m = models.resnet50(weights=weights)
        dim = m.fc.in_features
        m.fc = nn.Identity()
    else:
        raise ValueError(f"unsupported backbone: {name} (use resnet18|resnet50)")
    return m, dim


class DINOHead(nn.Module):
    """Standard DINO projection head (MLP + weight-norm prototype)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 4096,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last.weight_g.data.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last(x)


class DINOSifessModel(nn.Module):
    """Student–teacher pair + projector + soft equivariance head on bottleneck feats."""

    def __init__(
        self,
        backbone: str = "resnet50",
        out_dim: int = 4096,
        use_equivariance_head: bool = True,
        momentum: float = 0.996,
    ):
        super().__init__()
        self.momentum = momentum
        student_bb, feat_dim = build_backbone(backbone)
        teacher_bb, _ = build_backbone(backbone)
        self.student_backbone = student_bb
        self.teacher_backbone = teacher_bb
        self.feat_dim = feat_dim

        self.student_head = DINOHead(feat_dim, out_dim=out_dim)
        self.teacher_head = DINOHead(feat_dim, out_dim=out_dim)

        # Equivariance acts on L2-normalized bottleneck (pre-prototype) features
        self.student_bottleneck = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.use_equivariance_head = use_equivariance_head
        if use_equivariance_head:
            from sifess.models.equivariance import SoftEquivarianceHead

            self.rho = SoftEquivarianceHead(256)
        else:
            self.rho = None

        self._init_teacher_from_student()
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

    def _init_teacher_from_student(self) -> None:
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())

    @torch.no_grad()
    def update_teacher(self) -> None:
        m = self.momentum
        for ps, pt in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)
        for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)

    def embed_student(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.student_backbone(x)
        logits = self.student_head(feat)
        z = F.normalize(self.student_bottleneck(feat), dim=-1)
        return logits, z

    @torch.no_grad()
    def embed_teacher(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.teacher_backbone(x)
        return self.teacher_head(feat)

    def apply_rho(
        self,
        z: torch.Tensor,
        angles: torch.Tensor,
        reflect: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.rho is None:
            return z
        return self.rho(z, angles, reflect)

    def forward_multicrop(
        self,
        student_crops: Sequence[torch.Tensor],
        teacher_crops: Sequence[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """student_crops: all views; teacher_crops: typically global views only."""
        s_logits = []
        s_z = []
        for crop in student_crops:
            logits, z = self.embed_student(crop)
            s_logits.append(logits)
            s_z.append(z)
        t_logits = [self.embed_teacher(crop) for crop in teacher_crops]
        return {
            "student_logits": s_logits,
            "teacher_logits": t_logits,
            "student_z": s_z,
        }
