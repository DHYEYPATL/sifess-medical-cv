"""DINO-style student-teacher with optional SIFESS soft-equivariance head.

Single backbone family: ResNet-18 (smoke) / ResNet-50 (default full runs).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

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
    """Student-teacher pair + projector + equivariance map on bottleneck feats.

    The student bottleneck packs a dedicated SO(2) plane into the first 2
    dims with fixed energy fraction plane_alpha. That prevents the vacuous
    collapse where an invariant encoder dumps all mass into dims >=2 and drives
    fixed-SO(2) L_eq -> 0 (Day-0 V2 failure mode).
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        out_dim: int = 4096,
        use_equivariance_head: bool = True,
        momentum: float = 0.996,
        eq_action: str = "so2_2d",
        embed_dim: int = 256,
        plane_alpha: float = 0.1,
    ):
        super().__init__()
        self.momentum = momentum
        self.eq_action = eq_action
        self.plane_alpha = float(plane_alpha)
        student_bb, feat_dim = build_backbone(backbone)
        teacher_bb, _ = build_backbone(backbone)
        self.student_backbone = student_bb
        self.teacher_backbone = teacher_bb
        self.feat_dim = feat_dim

        self.student_head = DINOHead(feat_dim, out_dim=out_dim)
        self.teacher_head = DINOHead(feat_dim, out_dim=out_dim)

        self.student_bottleneck = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.use_equivariance_head = use_equivariance_head
        if use_equivariance_head:
            from sifess.models.equivariance import build_rho

            self.plane_proj = nn.Linear(feat_dim, 2)
            self.rho = build_rho(eq_action, dim=embed_dim)
        else:
            self.plane_proj = None
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

    def _pack_embedding(self, feat: torch.Tensor) -> torch.Tensor:
        """Build unit embedding with fixed energy in the SO(2) plane (first 2 dims)."""
        h = self.student_bottleneck(feat)
        rest = F.normalize(h[:, 2:], dim=-1)
        plane = F.normalize(self.plane_proj(feat), dim=-1)
        a = self.plane_alpha
        z = torch.cat([plane * (a ** 0.5), rest * ((1.0 - a) ** 0.5)], dim=-1)
        return z

    def embed_student(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.student_backbone(x)
        logits = self.student_head(feat)
        if self.use_equivariance_head:
            z = self._pack_embedding(feat)
        else:
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
