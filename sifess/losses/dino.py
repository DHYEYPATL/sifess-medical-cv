"""DINO cross-entropy loss with teacher centering / sharpening."""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim: int = 4096,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_logits: Sequence[torch.Tensor]) -> None:
        # teacher_logits: list of (B, D)
        batch_center = torch.cat(list(teacher_logits), dim=0).mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1.0 - self.center_momentum)

    def forward(
        self,
        student_logits: Sequence[torch.Tensor],
        teacher_logits: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Standard DINO: each teacher view supervises all student views except same index when aligned."""
        student = [s / self.student_temp for s in student_logits]
        teacher = [(t - self.center) / self.teacher_temp for t in teacher_logits]
        teacher = [F.softmax(t, dim=-1) for t in teacher]

        total = 0.0
        n = 0
        n_t = len(teacher)
        for i, t in enumerate(teacher):
            for j, s in enumerate(student):
                # Skip matching global view index when student list starts with same globals
                if j < n_t and i == j:
                    continue
                loss = torch.sum(-t * F.log_softmax(s, dim=-1), dim=-1).mean()
                total = total + loss
                n += 1
        self.update_center(teacher_logits)
        return total / max(n, 1)
