"""Losses: DINO + coherence-weighted soft equivariance (+ optional orthogonality)."""

from sifess.losses.dino import DINOLoss
from sifess.losses.equivariance import SoftEquivarianceLoss
from sifess.losses.combined import SIFESSLoss

__all__ = ["DINOLoss", "SoftEquivarianceLoss", "SIFESSLoss"]
