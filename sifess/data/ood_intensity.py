"""OOD intensity shifts: brightness / contrast / gamma.

Used by eval_ood to measure AUROC / delta under photometric shift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


ShiftName = Literal["brightness", "contrast", "gamma", "none"]


@dataclass
class OODConfig:
    shift: ShiftName = "none"
    severity: float = 0.5  # interpretation depends on shift


def apply_ood_intensity(images: torch.Tensor, cfg: OODConfig) -> torch.Tensor:
    """Apply intensity OOD to (B,C,H,W) tensors roughly in [-1,1] or [0,1].

    Operates in a loosely normalized space: we map to [0,1], transform, map back.
    """
    if cfg.shift == "none" or cfg.severity == 0:
        return images

    # Detect range
    amin, amax = images.min().item(), images.max().item()
    if amin >= -1.1 and amax <= 1.1:
        x01 = (images + 1.0) * 0.5
        back = lambda t: t * 2.0 - 1.0
    else:
        x01 = images.clamp(0, 1)
        back = lambda t: t

    s = float(cfg.severity)
    if cfg.shift == "brightness":
        # severity in ~[0,1] -> delta in [-0.5, 0.5] scaled
        x01 = (x01 + (s - 0.5)).clamp(0, 1)
    elif cfg.shift == "contrast":
        # contrast factor around 1
        factor = 0.5 + s  # 0.5..1.5 for s in 0..1
        mean = x01.mean(dim=(-2, -1), keepdim=True)
        x01 = ((x01 - mean) * factor + mean).clamp(0, 1)
    elif cfg.shift == "gamma":
        gamma = 0.5 + s  # 0.5..1.5
        x01 = x01.clamp(1e-6, 1.0) ** gamma
    else:
        raise ValueError(f"unknown shift {cfg.shift}")
    return back(x01)
